from __future__ import annotations

import asyncio
import argparse
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import parse_qs, unquote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from .legacy_jobs import (
    create_automatic_job,
    create_continuous_library_holdout_loop_job,
    create_dataset_build_job,
    create_deepphy_eval_job,
    create_library_holdout_job,
    create_pooltool_reward_optimization_job,
    create_pooltool_reward_single_scene_job,
    create_pooltool_reward_value_model_job,
    create_supervised_consistency_analysis_job,
    create_supervised_job,
    create_unsupervised_job,
)
from .input_sync import build_job_input_bundle, build_job_input_plan
from .input_sync import build_job_input_bundle_from_plan
from .models import (
    ClientKind,
    ClientSettingsUpdateRequest,
    CombineSupervisedRunsRequest,
    CreateJobRequest,
    CreateJobResponse,
    HeartbeatRequest,
    HideRepeatLearningRunsRequest,
    JobArtifactUploadResponse,
    JobLogChunkRequest,
    JobUpdateRequest,
    JobWorkersUpdateRequest,
    PollResponse,
    PreferredClientUpdateRequest,
    PublicAccessUpdateRequest,
    QueueSettings,
    QueueSettingsUpdateRequest,
    RegisterClientRequest,
    RegisterClientResponse,
)
from public_connection_public_access import PublicAccessController
from .state import ServerStateStore
from .supervised_label_importance import compute_supervised_label_importance, resolve_latest_supervised_library


def _coerce_form_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "on", "yes"}


def _resolve_next_url(raw_value: str | None, *, default: str = "/") -> str:
    candidate = str(raw_value or "").strip()
    if not candidate.startswith("/") or candidate.startswith("//"):
        return default
    return candidate


def _resolve_project_file(project_root: Path, raw_path: str) -> Path:
    candidate = (project_root / unquote(raw_path).lstrip("/")).resolve()
    project_root_resolved = project_root.resolve()
    if not str(candidate).startswith(str(project_root_resolved)):
        raise HTTPException(status_code=400, detail="Path escapes project root")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return candidate


def _build_job_response(
    state_store: ServerStateStore,
    builder,
    payload: dict,
) -> CreateJobResponse:
    try:
        request = builder(payload)
    except (ValidationError, ValueError, FileNotFoundError, RuntimeError) as exc:
        detail = exc.errors() if isinstance(exc, ValidationError) else str(exc)
        raise HTTPException(status_code=422, detail=detail) from exc
    return CreateJobResponse(job=state_store.create_job(request))


def create_app(
    project_root: Path | None = None,
    *,
    private_port: int = 8788,
    public_https_port: int = 8443,
    public_access_controller: PublicAccessController | None = None,
) -> FastAPI:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    state_store = ServerStateStore(project_root / "state", project_root=project_root)
    public_local_target_url = str(
        os.environ.get("PATTERN_PUBLIC_LOCAL_TARGET_URL", f"http://127.0.0.1:{int(private_port)}")
    ).strip()
    public_access = public_access_controller or PublicAccessController(
        project_root / "state",
        local_target_url=public_local_target_url,
        https_port=int(public_https_port),
    )
    input_bundle_cache: dict[str, tuple[float, bytes]] = {}
    input_bundle_cache_ttl_seconds = 120.0
    
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        public_access.start()
        scheduler_stop = asyncio.Event()

        async def _scheduler_loop() -> None:
            while not scheduler_stop.is_set():
                try:
                    state_store.run_maintenance_cycle()
                    state_store.schedule_pending_jobs()
                    _dispatch_direct_clients()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(scheduler_stop.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

        scheduler_task = asyncio.create_task(_scheduler_loop())
        try:
            yield
        finally:
            scheduler_stop.set()
            scheduler_task.cancel()
            try:
                await scheduler_task
            except BaseException:
                pass
            public_access.stop()

    app = FastAPI(title="Pattern Learning V2 Server", lifespan=lifespan)

    def _configured_server_api_key() -> str:
        try:
            payload = json.loads((project_root / "state" / "server_public_access.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        return str(payload.get("api_key") or "").strip()

    def _request_server_api_key(request: Request) -> str:
        authorization = str(request.headers.get("authorization") or "").strip()
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return str(request.headers.get("x-api-key") or "").strip()

    def _require_server_api_key(request: Request) -> None:
        configured_key = _configured_server_api_key()
        if not configured_key:
            return
        presented_key = _request_server_api_key(request)
        if presented_key and secrets.compare_digest(presented_key, configured_key):
            return
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing server API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _public_access_payload() -> dict[str, object]:
        state = public_access.get_state()
        payload = state.model_dump(mode="json")
        payload["api_key"] = ""
        payload["api_key_configured"] = bool(str(state.api_key or "").strip())
        return payload

    def _dispatch_direct_clients() -> None:
        direct_clients = [
            client
            for client in state_store.list_clients_snapshot()
            if client.kind == ClientKind.direct and client.status == "online" and str(client.direct_url or "").strip()
        ]
        if not direct_clients:
            return
        for direct_client in direct_clients:
            try:
                poll_response = state_store.poll(direct_client.client_id)
            except KeyError:
                continue
            base_url = str(direct_client.direct_url or "").rstrip("/")
            for command in poll_response.commands:
                try:
                    _post_direct_json(
                        f"{base_url}/api/v1/direct/jobs/{command.job_id}/command",
                        command.model_dump(mode="json"),
                    )
                except Exception:
                    continue
            for client_command in poll_response.client_commands:
                try:
                    _post_direct_json(
                        f"{base_url}/api/v1/direct/client/command",
                        client_command.model_dump(mode="json"),
                    )
                except Exception:
                    continue
            for assignment in poll_response.assignments:
                try:
                    _post_direct_json(
                        f"{base_url}/api/v1/direct/jobs/start",
                        assignment.model_dump(mode="json"),
                    )
                except Exception as exc:
                    state_store.release_job_assignment(
                        assignment.job_id,
                        message=f"Direct dispatch failed for client {direct_client.client_id}: {exc}",
                    )

    def _post_direct_json(url: str, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=15) as response:
                status = int(getattr(response, "status", 200) or 200)
        except urllib_error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Unable to reach direct client at {url}: {exc.reason}") from exc
        if status >= 400:
            raise RuntimeError(f"HTTP {status} from {url}")

    def _job_response_with_direct_dispatch(job_response: CreateJobResponse) -> CreateJobResponse:
        _schedule_and_dispatch_direct_clients()
        return job_response

    def _schedule_and_dispatch_direct_clients() -> None:
        state_store.run_maintenance_cycle()
        state_store.schedule_pending_jobs()
        _dispatch_direct_clients()

    def _input_bundle_cache_key(job_id: str, plan: dict[str, object]) -> str:
        return f"{job_id}:{json.dumps(plan, sort_keys=True, separators=(',', ':'))}"

    def _cached_input_bundle(job_id: str, plan: dict[str, object]) -> bytes:
        now = time.monotonic()
        cache_key = _input_bundle_cache_key(job_id, plan)
        cached = input_bundle_cache.get(cache_key)
        if cached is not None and (now - cached[0]) <= input_bundle_cache_ttl_seconds:
            return cached[1]
        payload = build_job_input_bundle_from_plan(plan)
        input_bundle_cache[cache_key] = (now, payload)
        stale_keys = [
            key
            for key, (created_at, _payload) in input_bundle_cache.items()
            if key != cache_key and (now - created_at) > input_bundle_cache_ttl_seconds
        ]
        for key in stale_keys:
            input_bundle_cache.pop(key, None)
        return payload

    @app.get("/")
    async def api_root() -> JSONResponse:
        return JSONResponse(
            {
                "name": "pattern-server",
                "mode": "api-only",
                "project_root": str(project_root),
                "endpoints": {
                    "health": "/healthz",
                    "server_info": "/api/v1/server/info",
                    "clients": "/api/v1/clients",
                    "jobs": "/api/v1/jobs",
                    "public_status": "/public/v1/status",
                    "public_jobs": "/public/v1/jobs",
                },
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/api/v1/server/info")
    async def server_info() -> JSONResponse:
        return JSONResponse(
            {
                "name": "pattern-server",
                "project_root": str(project_root),
                "artifact_roots": {
                    "configs": str(project_root / "configs"),
                    "data": str(project_root / "data"),
                    "evaluations": str(project_root / "evaluations"),
                    "logs": str(project_root / "logs"),
                    "state": str(project_root / "state"),
                },
            }
        )

    @app.get("/api/v1/public-access")
    async def get_public_access() -> JSONResponse:
        return JSONResponse(_public_access_payload())

    @app.post("/api/v1/public-access")
    async def update_public_access(
        http_request: Request,
        request: PublicAccessUpdateRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        public_access.update_state(
            enabled=request.enabled,
            api_key=request.api_key,
        )
        return JSONResponse(_public_access_payload())

    def _public_status_response() -> JSONResponse:
        state = public_access.get_state()
        jobs = state_store.list_jobs_snapshot()
        active_count = sum(1 for job in jobs if job.status.value in {"assigned", "starting", "running", "pausing", "resuming", "stopping"})
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job.status.value] = counts.get(job.status.value, 0) + 1
        return JSONResponse(
            {
                "server": "pattern-server",
                "job_counts": counts,
                "active_jobs": active_count,
                "public_url": state.public_url,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    def _public_jobs_response() -> JSONResponse:
        jobs = state_store.list_jobs_snapshot()
        return JSONResponse(
            {
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "title": str(job.spec.metadata.get("title") or job.spec.action),
                        "env_id": job.spec.metadata.get("env_id"),
                        "status": job.status.value,
                        "progress": job.spec.metadata.get("progress"),
                        "updated_at": job.updated_at.isoformat(),
                    }
                    for job in jobs
                ]
            }
        )

    @app.get("/public/v1/status")
    async def public_status() -> JSONResponse:
        return _public_status_response()

    @app.get("/public/v1/jobs")
    async def public_jobs() -> JSONResponse:
        return _public_jobs_response()

    @app.get("/v1/status")
    async def public_status_funnel_alias(request: Request) -> JSONResponse:
        return _public_status_response()

    @app.get("/v1/jobs")
    async def public_jobs_funnel_alias(request: Request) -> JSONResponse:
        return _public_jobs_response()

    @app.get("/api/v1/clients")
    async def list_clients() -> JSONResponse:
        return JSONResponse(
            {
                "clients": [
                    client.model_dump(mode="json")
                    for client in state_store.list_clients_snapshot()
                ]
            }
        )

    @app.post("/api/v1/clients/register", response_model=RegisterClientResponse)
    async def register_client(
        http_request: Request,
        request: RegisterClientRequest,
    ) -> RegisterClientResponse:
        _require_server_api_key(http_request)
        response = RegisterClientResponse(client=await run_in_threadpool(state_store.register_client, request))
        return response

    @app.post("/api/v1/clients/{client_id}/heartbeat")
    async def heartbeat(client_id: str, http_request: Request, request: HeartbeatRequest) -> JSONResponse:
        _require_server_api_key(http_request)
        try:
            client = await run_in_threadpool(
                state_store.heartbeat,
                client_id,
                current_jobs=request.current_jobs,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown client {client_id}"
            ) from exc
        if client.kind == ClientKind.direct:
            _schedule_and_dispatch_direct_clients()
        return JSONResponse({"client": client.model_dump(mode="json")})

    @app.post("/api/v1/clients/{client_id}/settings")
    async def update_client_settings(
        client_id: str,
        http_request: Request,
        request: ClientSettingsUpdateRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        try:
            client = await run_in_threadpool(
                state_store.update_client_max_concurrency,
                client_id,
                request.max_concurrency,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown client {client_id}") from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"client": client.model_dump(mode="json")})

    @app.post("/api/v1/clients/{client_id}/shutdown")
    async def shutdown_client(client_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            client = await run_in_threadpool(state_store.shutdown_client, client_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown client {client_id}") from exc
        return JSONResponse({"client": client.model_dump(mode="json")})

    @app.post("/api/v1/jobs/{job_id}/preferred-client")
    async def update_preferred_client(
        job_id: str,
        http_request: Request,
        request: PreferredClientUpdateRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        try:
            job = state_store.set_job_preferred_client(job_id, request.preferred_client_id or None)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.post("/api/v1/jobs/{job_id}/workers")
    async def update_job_workers(
        job_id: str,
        http_request: Request,
        request: JobWorkersUpdateRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        try:
            job = state_store.update_job_workers(job_id, request.workers)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.post("/api/v1/env/{env_id}/compute-supervised-importance")
    async def compute_supervised_importance(env_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        normalized = str(env_id or "").strip().lower()
        library = resolve_latest_supervised_library(project_root, normalized)
        if library is None:
            raise HTTPException(status_code=404, detail=f"No supervised library found for {normalized}")
        output = compute_supervised_label_importance(library.root)
        return JSONResponse({"ok": True, "output_path": str(output)})

    @app.post("/api/v1/clients/{client_id}/poll", response_model=PollResponse)
    async def poll(client_id: str, request: Request) -> PollResponse:
        _require_server_api_key(request)
        try:
            return await run_in_threadpool(state_store.poll, client_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown client {client_id}"
            ) from exc

    @app.post("/api/v1/clients/{client_id}/commands/{command_id}/ack")
    async def acknowledge_client_command(client_id: str, command_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            client = state_store.acknowledge_client_command(client_id, command_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown client {client_id}") from exc
        return JSONResponse({"client": client.model_dump(mode="json")})

    @app.post("/api/v1/jobs/{job_id}/commands/{command_id}/ack")
    async def acknowledge_command(job_id: str, command_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            job = state_store.acknowledge_command(job_id, command_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown job") from exc
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.get("/api/v1/jobs")
    async def list_jobs(full: bool = False) -> JSONResponse:
        jobs = state_store.list_jobs_snapshot()
        if full:
            return JSONResponse(
                {"jobs": [job.model_dump(mode="json") for job in jobs]}
            )
        return JSONResponse(
            {
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "status": job.status.value,
                        "created_at": job.created_at.isoformat(),
                        "updated_at": job.updated_at.isoformat(),
                        "started_at": (
                            job.started_at.isoformat()
                            if job.started_at is not None
                            else None
                        ),
                        "finished_at": (
                            job.finished_at.isoformat()
                            if job.finished_at is not None
                            else None
                        ),
                        "assigned_client_id": job.assigned_client_id,
                        "exit_code": job.exit_code,
                        "message": job.message,
                        "action": job.spec.action,
                        "env_id": job.spec.metadata.get("env_id"),
                        "progress": job.spec.metadata.get("progress"),
                    }
                    for job in jobs
                ]
            }
        )

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        try:
            job = state_store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        return JSONResponse(job.model_dump(mode="json"))

    @app.get("/api/v1/jobs/{job_id}/inputs")
    async def get_job_inputs(job_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            job = state_store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        assigned_client = None
        if job.assigned_client_id:
            try:
                assigned_client = state_store.get_client(job.assigned_client_id)
            except KeyError:
                assigned_client = None
        return JSONResponse(build_job_input_plan(job, project_root=project_root, assigned_client=assigned_client))

    @app.get("/api/v1/jobs/{job_id}/input-bundle")
    async def get_job_input_bundle(job_id: str, request: Request) -> Response:
        _require_server_api_key(request)
        try:
            job = state_store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        assigned_client = None
        if job.assigned_client_id:
            try:
                assigned_client = state_store.get_client(job.assigned_client_id)
            except KeyError:
                assigned_client = None
        plan = build_job_input_plan(job, project_root=project_root, assigned_client=assigned_client)
        payload = await run_in_threadpool(_cached_input_bundle, job_id, plan)
        return Response(
            content=payload,
            media_type="application/gzip",
            headers={"content-disposition": f'attachment; filename="{job_id}-inputs.tar.gz"'},
        )

    @app.get("/api/v1/jobs/{job_id}/logs")
    async def get_job_logs(job_id: str, limit: int = 200) -> JSONResponse:
        return JSONResponse({"lines": state_store.read_job_logs(job_id, limit=limit)})

    @app.post("/api/v1/jobs", response_model=CreateJobResponse)
    async def create_job(http_request: Request, request: CreateJobRequest) -> CreateJobResponse:
        _require_server_api_key(http_request)
        return _job_response_with_direct_dispatch(CreateJobResponse(job=state_store.create_job(request)))

    @app.post("/api/v1/submit/dataset-build", response_model=CreateJobResponse)
    async def submit_dataset_build(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(state_store, create_dataset_build_job, payload))

    @app.post("/api/v1/submit/supervised", response_model=CreateJobResponse)
    async def submit_supervised(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(state_store, create_supervised_job, payload))

    @app.post("/api/v1/submit/automatic", response_model=CreateJobResponse)
    async def submit_automatic(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(state_store, create_automatic_job, payload))

    @app.post("/api/v1/submit/unsupervised", response_model=CreateJobResponse)
    async def submit_unsupervised(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(state_store, create_unsupervised_job, payload))

    @app.post("/api/v1/submit/deepphy-eval", response_model=CreateJobResponse)
    async def submit_deepphy_eval(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(state_store, create_deepphy_eval_job, payload))

    @app.post("/api/v1/submit/library-holdout-eval", response_model=CreateJobResponse)
    async def submit_library_holdout_eval(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(state_store, create_library_holdout_job, payload))

    @app.post("/api/v1/submit/library-holdout-loop", response_model=CreateJobResponse)
    async def submit_library_holdout_loop(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(
            _build_job_response(
                state_store,
                create_continuous_library_holdout_loop_job,
                payload,
            )
        )

    @app.post("/api/v1/submit/supervised-consistency-analysis", response_model=CreateJobResponse)
    async def submit_supervised_consistency_analysis(request: Request) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(
            _build_job_response(state_store, create_supervised_consistency_analysis_job, payload)
        )

    @app.post("/api/v1/supervised-runs/combine", response_model=CreateJobResponse)
    async def combine_supervised_runs(
        http_request: Request,
        request: CombineSupervisedRunsRequest,
    ) -> CreateJobResponse:
        _require_server_api_key(http_request)
        try:
            job = await run_in_threadpool(
                lambda: state_store.combine_supervised_runs(
                    request.job_ids,
                    env_id=request.env_id,
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {exc.args[0]}") from exc
        except (ValueError, FileNotFoundError, RuntimeError, FileExistsError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _job_response_with_direct_dispatch(CreateJobResponse(job=job))

    @app.post("/api/v1/repeat-learning-runs/hide")
    async def hide_repeat_learning_runs(
        http_request: Request,
        request: HideRepeatLearningRunsRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        try:
            jobs = await run_in_threadpool(state_store.hide_repeat_learning_runs, request.job_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {exc.args[0]}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"jobs": [job.model_dump(mode="json") for job in jobs]})

    @app.post(
        "/api/v1/submit/pooltool-reward-optimize",
        response_model=CreateJobResponse,
    )
    async def submit_pooltool_reward_optimization(
        request: Request,
    ) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(
            state_store,
            create_pooltool_reward_optimization_job,
            payload,
        ))

    @app.post(
        "/api/v1/submit/pooltool-reward-value-model",
        response_model=CreateJobResponse,
    )
    async def submit_pooltool_reward_value_model(
        request: Request,
    ) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(
            state_store,
            create_pooltool_reward_value_model_job,
            payload,
        ))

    @app.post(
        "/api/v1/submit/pooltool-reward-single-scene",
        response_model=CreateJobResponse,
    )
    async def submit_pooltool_reward_single_scene(
        request: Request,
    ) -> CreateJobResponse:
        _require_server_api_key(request)
        payload = await request.json()
        return _job_response_with_direct_dispatch(_build_job_response(
            state_store,
            create_pooltool_reward_single_scene_job,
            payload,
        ))

    @app.post("/api/v1/jobs/{job_id}/pause")
    async def pause_job(job_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            job = state_store.pause_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.post("/api/v1/jobs/{job_id}/resume")
    async def resume_job(job_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            job = state_store.resume_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.post("/api/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            job = state_store.cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.post("/api/v1/jobs/{job_id}/requeue")
    async def requeue_job(job_id: str, request: Request) -> JSONResponse:
        _require_server_api_key(request)
        try:
            job = state_store.requeue_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.get("/api/v1/queue/settings")
    async def get_queue_settings() -> JSONResponse:
        settings = state_store.get_queue_settings()
        return JSONResponse({"queue_settings": settings.model_dump(mode="json")})

    @app.post("/api/v1/queue/settings")
    async def update_queue_settings(
        http_request: Request,
        request: QueueSettingsUpdateRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        current = state_store.get_queue_settings()
        updated = current.model_copy(
            update={
                "pooltool_attempt_limit": (
                    request.pooltool_attempt_limit
                    if request.pooltool_attempt_limit is not None
                    else current.pooltool_attempt_limit
                )
            }
        )
        settings = state_store.update_queue_settings(updated)
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"queue_settings": settings.model_dump(mode="json")})

    @app.put("/api/v1/jobs/{job_id}/status")
    async def update_job_status(
        job_id: str,
        http_request: Request,
        request: JobUpdateRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        try:
            job = await run_in_threadpool(state_store.update_job, job_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _schedule_and_dispatch_direct_clients()
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.post("/api/v1/jobs/{job_id}/logs")
    async def append_job_logs(
        job_id: str,
        http_request: Request,
        request: JobLogChunkRequest,
    ) -> JSONResponse:
        _require_server_api_key(http_request)
        try:
            job = await run_in_threadpool(state_store.append_job_logs, job_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"job": job.model_dump(mode="json")})

    @app.post(
        "/api/v1/jobs/{job_id}/artifact-bundle",
        response_model=JobArtifactUploadResponse,
    )
    async def upload_artifact_bundle(
        job_id: str,
        request: Request,
    ) -> JobArtifactUploadResponse:
        _require_server_api_key(request)
        try:
            body = await request.body()
            job, extracted = await run_in_threadpool(
                state_store.extract_artifact_bundle,
                job_id,
                archive_bytes=body,
                client_id=request.headers.get("x-client-id"),
            )
        except ClientDisconnect:
            raise HTTPException(status_code=499, detail="Client disconnected during artifact upload")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}") from exc
        except ValueError as exc:
            detail = str(exc)
            status_code = 409 if "leased" in detail or "assigned to client" in detail else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
        return JobArtifactUploadResponse(job=job, extracted_paths=extracted)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--public-https-port", type=int, default=8443)
    args = parser.parse_args()

    import uvicorn

    app = create_app(private_port=args.port, public_https_port=args.public_https_port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
