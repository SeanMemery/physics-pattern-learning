from __future__ import annotations

import copy
import io
import json
import os
import re
import shutil
import tarfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .imported_runs import MIGRATION_KEY, discover_imported_jobs
from .input_sync import _resolve_source_for_bundle
from .models import (
    ClientCommand,
    ClientCommandType,
    ClientKind,
    ClientRecord,
    ClientStatus,
    CreateJobRequest,
    JobCommand,
    JobCommandType,
    JobLogChunkRequest,
    JobRecord,
    JobStatus,
    JobUpdateRequest,
    PollResponse,
    QueueSettings,
    RegisterClientRequest,
)
from .supervised_run_combiner import combine_supervised_run_artifacts


MAX_RECENT_LOG_LINES = 200
MAX_STORED_ARTIFACT_RELPATHS = 64
MAX_STORED_UPLOADED_ARTIFACT_PREVIEW = 8
CLIENT_PRUNE_TIMEOUT = timedelta(
    seconds=max(
        30,
        int(os.environ.get("PATTERN_SERVER_CLIENT_PRUNE_TIMEOUT_SECONDS", "180")),
    )
)
CLIENT_HEARTBEAT_TIMEOUT = timedelta(
    seconds=max(
        30,
        int(
            os.environ.get(
                "PATTERN_SERVER_CLIENT_HEARTBEAT_TIMEOUT_SECONDS",
                str(int(CLIENT_PRUNE_TIMEOUT.total_seconds())),
            )
        ),
    )
)
CLIENT_POLL_RECORD_INTERVAL_SECONDS = max(
    5,
    int(os.environ.get("PATTERN_SERVER_CLIENT_POLL_RECORD_INTERVAL_SECONDS", "30")),
)
REMOTE_IDLE_POLL_DELAY_SECONDS = max(
    5.0,
    float(os.environ.get("PATTERN_SERVER_REMOTE_IDLE_POLL_DELAY_SECONDS", "15")),
)
REMOTE_BUSY_POLL_DELAY_SECONDS = max(
    1.0,
    float(os.environ.get("PATTERN_SERVER_REMOTE_BUSY_POLL_DELAY_SECONDS", "5")),
)
DIRECT_POLL_DELAY_SECONDS = max(
    1.0,
    float(os.environ.get("PATTERN_SERVER_DIRECT_POLL_DELAY_SECONDS", "2")),
)
ACTIVE_JOB_STATUSES = {
    JobStatus.assigned,
    JobStatus.starting,
    JobStatus.running,
    JobStatus.pausing,
    JobStatus.resuming,
    JobStatus.stopping,
}
ARTIFACT_UPLOAD_ALLOWED_STATUSES = {
    JobStatus.assigned,
    JobStatus.starting,
    JobStatus.running,
}
QUEUE_DISPATCH_PRIORITY = {
    "supervised-learn-and-evaluate": 0,
    "evaluate-deepphy": 0,
    "evaluate-library-case": 0,
    "analyze-supervised-consistency": 0,
    "pooltool-reward-optimize": 1,
    "pooltool-reward-value-model": 1,
    "pooltool-reward-single-scene": 1,
    "automatic-launch": 1,
    "unsupervised-launch": 2,
}
DIRECT_CLIENT_ACTIONS = {
    "evaluate-library-case",
    "analyze-supervised-consistency",
    "pooltool-reward-optimize",
    "pooltool-reward-value-model",
    "pooltool-reward-single-scene",
}
REMOTE_FORBIDDEN_ACTIONS = {
    "analyze-supervised-consistency",
    "pooltool-reward-optimize",
    "pooltool-reward-value-model",
    "pooltool-reward-single-scene",
}
AUTO_RESUME_PAUSE_PREFIXES = (
    "Auto-paused after client slot limit changed to ",
    "Auto-paused after client exceeded configured slot limit",
)
TRANSIENT_JOB_MESSAGE_PREFIXES = (
    "Recovered after client ",
    "Capacity available; re-queued automatically",
    "Dependency ready; re-queued automatically",
    "Recovered after missing client assignment",
)
JOB_REPORT_GRACE_SECONDS = 90
JOB_MISSING_REPORT_REQUEUE_SECONDS = max(
    120,
    int(os.environ.get("PATTERN_SERVER_MISSING_REPORT_REQUEUE_SECONDS", "600")),
)
JOB_REDISPATCH_GUARD_SECONDS = max(
    5,
    int(os.environ.get("PATTERN_SERVER_JOB_REDISPATCH_GUARD_SECONDS", "60")),
)
PREREQUISITE_REASON_CACHE_SECONDS = max(
    1,
    int(os.environ.get("PATTERN_SERVER_PREREQUISITE_REASON_CACHE_SECONDS", "5")),
)
PERIODIC_RECONCILE_SECONDS = max(
    1,
    int(os.environ.get("PATTERN_SERVER_PERIODIC_RECONCILE_SECONDS", "600")),
)
STALE_ACTIVE_JOB_TIMEOUT = timedelta(
    seconds=max(
        60,
        int(os.environ.get("PATTERN_SERVER_STALE_ACTIVE_JOB_TIMEOUT_SECONDS", "1800")),
    )
)
REPEAT_LEARNING_TARGET_RUNS = 3
REPEAT_EVALS_PER_RUN = 1
IMAGE_ONLY_EVAL_TARGET_RUNS = 8
PHYRE_IMAGE_ONLY_COMPLETION_TASK_TARGET = 125
SERVER_MIN_CANDIDATE_EVALUATIONS = 100
SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS = 25
SERVER_MAX_CANDIDATE_EVALUATIONS = 100
SERVER_POPULATION_SIZE = 25
REMOTE_DEFAULT_EXECUTION_ROOT = Path("/opt/pattern-learning-stack")
SERVER_POLICY_CONFIG_RELATIVE_PATH = Path("configs/generated/pattern_learning_server_policy.yaml")
REPEAT_RUN_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)-r(?P<index>\d+)$")
REPEAT_EVAL_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)-repeat-(?P<index>\d+)$")
REPEAT_LEARNING_ACTIONS = {
    "supervised-learn-and-evaluate": "supervised",
    "unsupervised-launch": "unsupervised",
    "automatic-launch": "automatic",
}
LEARNING_STOP_POLICY_ACTIONS = {
    "supervised-learn-and-evaluate",
    "unsupervised-launch",
    "automatic-launch",
}
RESUMABLE_EXISTING_RUN_ACTIONS = {
    "unsupervised-launch",
    "automatic-launch",
}
REPEAT_COUNTED_STATUSES = {
    JobStatus.queued,
    JobStatus.waiting_for_dependencies,
    JobStatus.assigned,
    JobStatus.starting,
    JobStatus.running,
    JobStatus.pausing,
    JobStatus.paused,
    JobStatus.resuming,
    JobStatus.stopping,
    JobStatus.succeeded,
}
HIDDEN_REQUIRED_RUN_METADATA_KEY = "hidden_required_run"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ServerStateStore:
    def __init__(self, root: Path, *, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = self.root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "server_state.json"
        self._lock = threading.RLock()
        self._snapshot_lock = threading.Lock()
        self._clients_snapshot: tuple[ClientRecord, ...] = ()
        self._jobs_snapshot: tuple[JobRecord, ...] = ()
        self._state_payload_cache: dict[str, Any] | None = None
        self._state_payload_mtime_ns: int | None = None
        self._prerequisite_reason_cache: dict[tuple[str, str, str], tuple[float, str | None]] = {}
        self._last_periodic_reconcile_monotonic = 0.0
        if not self.path.exists():
            self._write(
                {
                    "clients": {},
                    "client_commands": {},
                    "jobs": {},
                    "meta": {"migrations": {}},
                    "queue_settings": QueueSettings().model_dump(mode="json"),
                }
            )
        else:
            self._load_snapshot_from_disk()

    def _client_execution_root(self, client: ClientRecord) -> Path:
        metadata = client.capabilities.metadata or {}
        raw_execution_root = str(metadata.get("execution_root") or "").strip()
        if raw_execution_root:
            return Path(raw_execution_root)
        return REMOTE_DEFAULT_EXECUTION_ROOT

    def _rewrite_string_path_prefixes(
        self,
        value: Any,
        *,
        replacements: list[tuple[str, str]],
    ) -> Any:
        if isinstance(value, str):
            rewritten = value
            for source_prefix, target_prefix in replacements:
                rewritten = rewritten.replace(source_prefix, target_prefix)
            return rewritten
        if isinstance(value, list):
            return [
                self._rewrite_string_path_prefixes(item, replacements=replacements)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: self._rewrite_string_path_prefixes(item, replacements=replacements)
                for key, item in value.items()
            }
        return value

    def _rewrite_job_paths_for_client(
        self,
        job: JobRecord,
        client: ClientRecord,
    ) -> tuple[JobRecord, bool]:
        if client.kind != ClientKind.remote:
            return job, False
        execution_root = self._client_execution_root(client)
        project_root = self.project_root.resolve()
        if project_root.name == "pattern-learning-server":
            local_server_root = project_root
            local_services_root = project_root.parent
        else:
            local_server_root = (project_root / "pattern-learning-server").resolve()
            local_services_root = project_root
        local_client_root = (local_services_root / "pattern-learning-client").resolve()
        local_server_data_root = (local_server_root / "data").resolve()
        replacements = [
            (str(local_server_data_root), str(execution_root / "data")),
            (str(local_client_root), str(execution_root)),
            (
                str(local_services_root),
                str(execution_root / local_services_root.name),
            ),
        ]
        changed = False
        rewritten_argv = self._rewrite_string_path_prefixes(job.spec.argv, replacements=replacements)
        if rewritten_argv != job.spec.argv:
            job.spec.argv = [str(part) for part in rewritten_argv]
            changed = True
        action = str(job.spec.action or "").strip()
        if action in {"evaluate-deepphy", "evaluate-library-case"}:
            # Keep canonical server-side metadata paths for eval jobs so
            # prerequisite checks and input synchronization can source files.
            return job, changed
        metadata = dict(job.spec.metadata or {})
        rewritten_metadata = self._rewrite_string_path_prefixes(metadata, replacements=replacements)
        if rewritten_metadata != metadata:
            job.spec.metadata = rewritten_metadata
            changed = True
        return job, changed
        self.migrate_imported_runs()

    def _normalize_client_concurrency_defaults(self, client: ClientRecord) -> bool:
        changed = False
        if client.kind == ClientKind.remote:
            normalized = max(self._minimum_remote_client_concurrency(client), int(client.max_concurrency or 0))
            if normalized != client.max_concurrency:
                client.max_concurrency = normalized
                changed = True
            caps_max = max(
                self._minimum_remote_client_concurrency(client),
                int(client.capabilities.max_concurrency or 0),
            )
            if caps_max != client.capabilities.max_concurrency:
                client.capabilities.max_concurrency = caps_max
                changed = True
        return changed

    def _minimum_remote_client_concurrency(self, client: ClientRecord) -> int:
        _ = {
            str(tag).strip().lower()
            for tag in (client.capabilities.tags or [])
            if str(tag).strip()
        }
        return 4

    def register_client(self, request: RegisterClientRequest) -> ClientRecord:
        with self._lock:
            state = self._read()
            now = _utc_now()
            capabilities = request.capabilities.model_copy(deep=True)
            if request.kind == ClientKind.remote:
                min_concurrency = 4
                capabilities.max_concurrency = max(min_concurrency, int(capabilities.max_concurrency or 0))
            client = ClientRecord(
                client_id=uuid.uuid4().hex,
                display_name=request.display_name,
                kind=request.kind,
                direct_url=request.direct_url,
                version=request.version,
                hostname=request.hostname,
                platform=request.platform,
                capabilities=capabilities,
                max_concurrency=max(1, int(capabilities.max_concurrency or 4)),
                last_heartbeat=now,
            )
            state["clients"][client.client_id] = client.model_dump(mode="json")
            state.setdefault("client_commands", {})[client.client_id] = []
            self._adopt_replacement_client_state(
                state,
                replacement_client_id=client.client_id,
                replacement_client=client,
            )
            self._write(state)
            return client

    def _adopt_replacement_client_state(
        self,
        state: dict[str, Any],
        *,
        replacement_client_id: str,
        replacement_client: ClientRecord,
    ) -> bool:
        replacement_raw = state["clients"].get(replacement_client_id)
        if replacement_raw is None:
            return False
        replacement_key = self._client_instance_key(replacement_raw)
        if not replacement_key:
            return False
        changed = False
        adopted_jobs: set[str] = set(replacement_client.current_jobs or [])
        adopted_commands = list(state.setdefault("client_commands", {}).get(replacement_client_id, []))
        previous_client_ids = [
            client_id
            for client_id, raw_client in state.get("clients", {}).items()
            if client_id != replacement_client_id and self._client_instance_key(raw_client) == replacement_key
        ]
        for previous_client_id in previous_client_ids:
            previous_commands = state.setdefault("client_commands", {}).get(previous_client_id, [])
            if previous_commands:
                adopted_commands.extend(previous_commands)
                changed = True
            previous_jobs = {
                job_id
                for job_id, raw_job in state.get("jobs", {}).items()
                if (
                    JobRecord.model_validate(raw_job).status in ACTIVE_JOB_STATUSES
                    and str(JobRecord.model_validate(raw_job).assigned_client_id or "").strip() == previous_client_id
                )
            }
            for job_id in previous_jobs:
                raw_job = state["jobs"].get(job_id)
                if raw_job is None:
                    continue
                job = JobRecord.model_validate(raw_job)
                job.assigned_client_id = replacement_client_id
                state["jobs"][job_id] = job.model_dump(mode="json")
                adopted_jobs.add(job_id)
                changed = True
            state["clients"].pop(previous_client_id, None)
            state.setdefault("client_commands", {}).pop(previous_client_id, None)
            changed = True
        if not changed:
            return False
        replacement_client.current_jobs = sorted(adopted_jobs)
        state["clients"][replacement_client_id] = replacement_client.model_dump(mode="json")
        if adopted_commands:
            state["client_commands"][replacement_client_id] = adopted_commands
        return True

    def heartbeat(self, client_id: str, *, current_jobs: list[str]) -> ClientRecord:
        with self._lock:
            state = self._read()
            raw = state["clients"].get(client_id)
            if raw is None:
                raise KeyError(client_id)
            client = ClientRecord.model_validate(raw)
            self._normalize_client_concurrency_defaults(client)
            client.last_heartbeat = _utc_now()
            reported_current_jobs = set(current_jobs)
            if client.status == ClientStatus.draining:
                # Once a shutdown has been requested, worker-side heartbeats can still
                # arrive briefly from threads that are winding down. Do not let those
                # late heartbeats reclaim jobs or repopulate the draining client.
                client.current_jobs = []
                stale_reported_jobs = set(reported_current_jobs)
                reported_current_jobs = set()
            else:
                reported_current_jobs, stale_reported_jobs = self._split_reported_current_jobs(
                    state,
                    client_id,
                    reported_current_jobs=reported_current_jobs,
                )
                client.current_jobs = sorted(reported_current_jobs)
                client.status = ClientStatus.online
            state["clients"][client_id] = client.model_dump(mode="json")
            if stale_reported_jobs:
                self._queue_stale_job_commands(
                    state,
                    client_id,
                    stale_job_ids=stale_reported_jobs,
                )
            self._reconcile_client_reported_jobs(
                state,
                client_id,
                reported_current_jobs=reported_current_jobs,
            )
            self._prune_redundant_requeue_commands(state)
            self._write(state)
            return client

    def _restore_pruned_client_from_active_jobs(
        self,
        state: dict[str, Any],
        client_id: str,
        *,
        current_jobs: list[str],
    ) -> dict[str, Any] | None:
        reported_current_jobs = {str(job_id).strip() for job_id in current_jobs if str(job_id).strip()}
        if not reported_current_jobs:
            return None
        owned_job_ids: set[str] = set()
        for job_id in reported_current_jobs:
            raw_job = state["jobs"].get(job_id)
            if raw_job is None:
                continue
            job = JobRecord.model_validate(raw_job)
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            if str(job.assigned_client_id or "").strip() != client_id:
                continue
            owned_job_ids.add(job_id)
        if not owned_job_ids:
            return None
        now = _utc_now()
        client = ClientRecord(
            client_id=client_id,
            display_name=f"Recovered {client_id[:8]}",
            status=ClientStatus.online,
            hostname="unknown",
            platform="unknown",
            max_concurrency=max(1, len(owned_job_ids)),
            last_heartbeat=now,
            current_jobs=sorted(owned_job_ids),
        )
        state["clients"][client_id] = client.model_dump(mode="json")
        state.setdefault("client_commands", {})[client_id] = []
        return state["clients"][client_id]

    def list_clients(self) -> list[ClientRecord]:
        with self._lock:
            state = self._read()
            clients = [
                ClientRecord.model_validate(item)
                for item in state["clients"].values()
            ]
            return sorted(
                clients,
                key=lambda item: (
                    0 if item.status == ClientStatus.online else 1,
                    item.display_name.lower(),
                    item.client_id,
                ),
            )

    def list_clients_snapshot(self) -> list[ClientRecord]:
        self._refresh_snapshot_from_disk_if_needed()
        with self._snapshot_lock:
            return list(self._clients_snapshot)

    def get_client(self, client_id: str) -> ClientRecord:
        with self._lock:
            state = self._read()
            raw = state["clients"].get(client_id)
            if raw is None:
                raise KeyError(client_id)
            return ClientRecord.model_validate(raw)

    def update_client_max_concurrency(self, client_id: str, max_concurrency: int) -> ClientRecord:
        with self._lock:
            state = self._read()
            raw = state["clients"].get(client_id)
            if raw is None:
                raise KeyError(client_id)
            client = ClientRecord.model_validate(raw)
            self._normalize_client_concurrency_defaults(client)
            client.max_concurrency = max(1, int(max_concurrency))
            if client.kind == ClientKind.remote:
                client.max_concurrency = max(
                    self._minimum_remote_client_concurrency(client),
                    client.max_concurrency,
                )
                client.capabilities.max_concurrency = client.max_concurrency
            state["clients"][client_id] = client.model_dump(mode="json")
            active_jobs = self._active_jobs_from_state(state)
            self._enforce_client_slot_limit(
                state,
                client_id,
                reason=f"Auto-paused after client slot limit changed to {client.max_concurrency}",
                active_jobs=active_jobs,
            )
            self._write(state)
            return client

    def update_job_workers(self, job_id: str, workers: int) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            desired_workers = self._normalize_job_workers(job, workers)
            updated = self._apply_job_worker_update(job, desired_workers)
            updated.updated_at = _utc_now()
            state["jobs"][job_id] = updated.model_dump(mode="json")
            self._write(state)
            return updated

    def shutdown_client(self, client_id: str) -> ClientRecord:
        with self._lock:
            state = self._read()
            raw = state["clients"].get(client_id)
            if raw is None:
                raise KeyError(client_id)
            client = ClientRecord.model_validate(raw)
            client.status = ClientStatus.draining
            client.current_jobs = []
            state["clients"][client_id] = client.model_dump(mode="json")
            self._requeue_jobs_for_shutdown(state, client_id)
            pending = [
                ClientCommand.model_validate(item)
                for item in state.setdefault("client_commands", {}).get(client_id, [])
            ]
            if not any(command.command == ClientCommandType.shutdown for command in pending):
                pending.append(
                    ClientCommand(
                        command_id=uuid.uuid4().hex,
                        command=ClientCommandType.shutdown,
                        created_at=_utc_now(),
                    )
                )
            state["client_commands"][client_id] = [
                command.model_dump(mode="json")
                for command in pending
            ]
            self._write(state)
            return client

    def get_queue_settings(self) -> QueueSettings:
        with self._lock:
            state = self._read()
            return QueueSettings.model_validate(state.get("queue_settings") or {})

    def update_queue_settings(self, settings: QueueSettings) -> QueueSettings:
        with self._lock:
            state = self._read()
            state["queue_settings"] = settings.model_dump(mode="json")
            self._write(state)
            # Apply queue-setting-dependent reconciliations immediately.
            reconciled = self._read(force_reconcile=True, allow_periodic_reconcile=True)
            if reconciled != state:
                self._write(reconciled)
            return settings

    def create_job(self, request: CreateJobRequest) -> JobRecord:
        with self._lock:
            state = self._read()
            job = self._create_job_in_state(state, request)
            self._write(state)
            return job

    def _create_job_in_state(self, state: dict[str, Any], request: CreateJobRequest) -> JobRecord:
        now = _utc_now()
        spec = request.spec.model_copy(deep=True)
        metadata = dict(spec.metadata or {})
        metadata.setdefault("queue_requested", bool(request.enqueue))
        metadata.setdefault("launch_mode", "queue" if request.enqueue else "run_now")
        if request.enqueue:
            metadata.setdefault("queue_order", self._next_queue_order(state))
        spec.metadata = metadata
        job = JobRecord(
            job_id=uuid.uuid4().hex,
            status=JobStatus.queued,
            created_at=now,
            updated_at=now,
            spec=spec,
        )
        state["jobs"][job.job_id] = job.model_dump(mode="json")
        self._job_root(job.job_id).mkdir(parents=True, exist_ok=True)
        return job

    def _resolve_supervised_source_run_dir(
        self,
        run_dir_text: str,
        *,
        env_id: str,
    ) -> Path:
        run_name = Path(run_dir_text).name.strip()
        if run_name and env_id:
            canonical = (self.project_root / "data" / "patterns" / env_id / run_name).resolve()
            if canonical.is_dir():
                return canonical

        candidate = _resolve_source_for_bundle(Path(run_dir_text), project_root=self.project_root)
        if not candidate.is_absolute():
            candidate = (self.project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate

        if run_name:
            direct_fallback = (self.project_root / "data" / "patterns" / env_id / run_name).resolve()
            if direct_fallback.is_dir():
                return direct_fallback
            wildcard_fallbacks = sorted(
                path.resolve()
                for path in (self.project_root / "data" / "patterns").glob(f"*/{run_name}")
                if path.is_dir()
            )
            if len(wildcard_fallbacks) == 1:
                return wildcard_fallbacks[0]

        raise FileNotFoundError(f"Supervised run directory not found: {candidate}")

    def combine_supervised_runs(self, job_ids: list[str], *, env_id: str | None = None) -> JobRecord:
        normalized_job_ids = []
        seen_job_ids: set[str] = set()
        for value in job_ids:
            job_id = str(value or "").strip()
            if not job_id or job_id in seen_job_ids:
                continue
            normalized_job_ids.append(job_id)
            seen_job_ids.add(job_id)
        if len(normalized_job_ids) < 2:
            raise ValueError("Select at least two supervised runs to combine")

        source_run_dirs: list[Path] = []
        source_jobs: list[JobRecord] = []
        source_payloads: dict[str, dict[str, Any]] = {}

        with self._lock:
            state = self._read()
            for job_id in normalized_job_ids:
                raw_job = state["jobs"].get(job_id)
                if raw_job is None:
                    raise KeyError(job_id)
                job = JobRecord.model_validate(raw_job)
                metadata = dict(job.spec.metadata or {})
                if str(job.spec.action or "").strip() != "supervised-learn-and-evaluate":
                    raise ValueError(f"Job {job_id} is not a supervised learning run")
                if str(metadata.get("workflow_kind") or "").strip().lower() not in {"", "supervised"}:
                    raise ValueError(f"Job {job_id} is not a supervised learning run")
                if metadata.get("combined_into_job_id"):
                    raise ValueError(f"Job {job_id} has already been combined")
                job_env_id = str(metadata.get("env_id") or "").strip().lower()
                if env_id is not None and job_env_id != str(env_id).strip().lower():
                    raise ValueError(f"Job {job_id} does not belong to {env_id}")
                run_dir_text = str(metadata.get("run_dir") or "").strip()
                if not run_dir_text:
                    output_path = str(metadata.get("output_path") or "").strip()
                    if output_path:
                        run_dir_text = str(Path(output_path).parent)
                if not run_dir_text:
                    raise ValueError(f"Job {job_id} has no run_dir")
                resolve_env_id = job_env_id or str(env_id or "").strip().lower()
                run_dir = self._resolve_supervised_source_run_dir(
                    run_dir_text,
                    env_id=resolve_env_id,
                )
                source_run_dirs.append(run_dir)
                source_jobs.append(job)
                source_payloads[job.job_id] = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}

            effective_env_id = str(env_id or (source_jobs[0].spec.metadata.get("env_id") if source_jobs else "") or "").strip().lower()
            if not effective_env_id:
                raise ValueError("Unable to determine environment for selected supervised runs")

            now = _utc_now()
            for job in source_jobs:
                latest = JobRecord.model_validate(state["jobs"][job.job_id])
                metadata = dict(latest.spec.metadata or {})
                metadata["combine_pending"] = True
                metadata["combine_pending_at"] = now.isoformat()
                metadata["combine_pending_with"] = normalized_job_ids
                latest.spec.metadata = metadata
                if latest.status in ACTIVE_JOB_STATUSES:
                    latest.status = JobStatus.stopping
                    latest.updated_at = now
                    latest.message = "Stopping for supervised run combination"
                    has_cancel = any(command.command == JobCommandType.cancel for command in latest.pending_commands)
                    if not has_cancel:
                        self._append_command(latest, JobCommandType.cancel)
                elif latest.status not in {JobStatus.cancelled, JobStatus.succeeded, JobStatus.stopped}:
                    latest.status = JobStatus.cancelled
                    latest.finished_at = now
                    latest.updated_at = now
                    latest.message = "Superseded by supervised run combination"
                state["jobs"][latest.job_id] = latest.model_dump(mode="json")
            self._write(state)

        source_run_names = [path.name for path in source_run_dirs]
        combined_run_id = self._combined_supervised_run_id(
            env_id=effective_env_id,
            source_run_names=source_run_names,
        )
        combined_run_dir = self.project_root / "data" / "patterns" / effective_env_id / combined_run_id

        try:
            combined = combine_supervised_run_artifacts(
                project_root=self.project_root,
                env_id=effective_env_id,
                source_run_dirs=source_run_dirs,
                target_run_dir=combined_run_dir,
                min_candidate_evaluations=SERVER_MIN_CANDIDATE_EVALUATIONS,
                no_improvement_patience_evaluations=SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS,
                max_candidate_evaluations=SERVER_MAX_CANDIDATE_EVALUATIONS,
            )
        except Exception:
            with self._lock:
                state = self._read()
                for job_id in normalized_job_ids:
                    raw_job = state["jobs"].get(job_id)
                    if raw_job is None:
                        continue
                    job = JobRecord.model_validate(raw_job)
                    metadata = dict(job.spec.metadata or {})
                    metadata.pop("combine_pending", None)
                    metadata.pop("combine_pending_at", None)
                    metadata.pop("combine_pending_with", None)
                    job.spec.metadata = metadata
                    state["jobs"][job.job_id] = job.model_dump(mode="json")
                self._write(state)
            shutil.rmtree(combined_run_dir, ignore_errors=True)
            raise

        base_source_job = sorted(
            source_jobs,
            key=lambda item: (
                item.created_at,
                item.job_id,
            ),
        )[0]
        local_client_root = (self.project_root.parent / "pattern-learning-client").resolve()
        reverse_replacements = [
            ("/opt/pattern-learning-stack/configs", str((self.project_root / "configs").resolve())),
            ("/opt/pattern-learning-stack/data", str((self.project_root / "data").resolve())),
            ("/opt/pattern-learning-stack/evaluations", str((self.project_root / "evaluations").resolve())),
            ("/var/opt/pattern-learning-stack/configs", str((self.project_root / "configs").resolve())),
            ("/var/opt/pattern-learning-stack/data", str((self.project_root / "data").resolve())),
            ("/var/opt/pattern-learning-stack/evaluations", str((self.project_root / "evaluations").resolve())),
            ("/var/opt/pattern-learning-stack", str(local_client_root)),
            ("/opt/pattern-learning-stack", str(local_client_root)),
        ]

        base_metadata = dict(base_source_job.spec.metadata or {})
        base_request_payload = dict(source_payloads.get(base_source_job.job_id) or {})
        if not base_request_payload:
            request_payload = base_metadata.get("request")
            if isinstance(request_payload, dict):
                base_request_payload = dict(request_payload)
        canonical_request_payload_raw = self._rewrite_string_path_prefixes(
            base_request_payload,
            replacements=reverse_replacements,
        )
        canonical_request_payload = (
            dict(canonical_request_payload_raw)
            if isinstance(canonical_request_payload_raw, dict)
            else {}
        )
        combined_output_path = combined_run_dir / "pattern_library.json"
        canonical_request_payload["env_id"] = effective_env_id
        canonical_request_payload["output_path"] = str(combined_output_path)
        canonical_request_payload["run_timestamp"] = combined_run_id
        canonical_request_payload = self._normalize_learning_stop_policy_request(
            canonical_request_payload
        )

        base_resume_argv = list(base_metadata.get("resume_argv") or base_source_job.spec.argv or [])
        canonical_argv_raw = self._rewrite_string_path_prefixes(
            base_resume_argv,
            replacements=reverse_replacements,
        )
        canonical_argv = [str(part) for part in canonical_argv_raw] if isinstance(canonical_argv_raw, list) else []
        if not canonical_argv:
            raise RuntimeError("Unable to build resume command for combined supervised run")
        learning_python = local_client_root / "venv" / "learning" / "bin" / "python"
        if learning_python.is_file():
            canonical_argv[0] = str(learning_python)
        canonical_argv = self._replace_or_add_flag(
            canonical_argv,
            "--output-path",
            str(combined_output_path),
        )
        canonical_argv = self._replace_or_add_flag(
            canonical_argv,
            "--env",
            effective_env_id,
        )
        canonical_argv = self._ensure_flag(canonical_argv, "--resume")
        canonical_argv = self._rewrite_learning_stop_policy_argv(
            canonical_argv,
            action="supervised-learn-and-evaluate",
        )

        combined_spec = base_source_job.spec.model_copy(deep=True)
        combined_spec.action = "supervised-learn-and-evaluate"
        combined_spec.argv = canonical_argv
        combined_spec.metadata = dict(base_metadata)
        combined_spec.metadata["action"] = "supervised-learn-and-evaluate"
        combined_spec.metadata["env_id"] = effective_env_id
        combined_spec.metadata["workflow_kind"] = "supervised"
        combined_spec.metadata["supports_pause"] = True
        combined_spec.metadata["run_dir"] = str(combined_run_dir)
        combined_spec.metadata["output_path"] = str(combined_output_path)
        combined_spec.metadata["resume_argv"] = canonical_argv
        combined_spec.metadata["request"] = canonical_request_payload

        combined_request = CreateJobRequest(
            enqueue=True,
            spec=combined_spec,
        )
        combined_request.spec.metadata = dict(combined_request.spec.metadata or {})
        combined_request.spec.metadata["combined_source_job_ids"] = normalized_job_ids
        combined_request.spec.metadata["combined_source_run_dirs"] = [str(path) for path in source_run_dirs]
        combined_request.spec.metadata["combined_candidate_count"] = int(combined.candidate_count)
        combined_request.spec.metadata["combined_completed_label_count"] = int(combined.completed_label_count)
        combined_request.spec.metadata["combined_total_label_count"] = int(combined.total_label_count)

        with self._lock:
            state = self._read()
            combined_job = self._create_job_in_state(state, combined_request)
            if combined.fully_completed:
                combined_job.status = JobStatus.succeeded
                combined_job.finished_at = _utc_now()
                combined_job.updated_at = combined_job.finished_at
                combined_job.message = (
                    f"Combined from {len(normalized_job_ids)} supervised runs "
                    f"({combined.completed_label_count}/{combined.total_label_count} labels complete)"
                )
                state["jobs"][combined_job.job_id] = combined_job.model_dump(mode="json")
            else:
                combined_job.message = (
                    f"Combined from {len(normalized_job_ids)} supervised runs "
                    f"({combined.completed_label_count}/{combined.total_label_count} labels complete)"
                )
                state["jobs"][combined_job.job_id] = combined_job.model_dump(mode="json")

            now = _utc_now()
            for source_job_id in normalized_job_ids:
                raw_job = state["jobs"].get(source_job_id)
                if raw_job is None:
                    continue
                source_job = JobRecord.model_validate(raw_job)
                metadata = dict(source_job.spec.metadata or {})
                metadata.pop("combine_pending", None)
                metadata.pop("combine_pending_at", None)
                metadata.pop("combine_pending_with", None)
                metadata["combined_into_job_id"] = combined_job.job_id
                metadata["combined_into_run_dir"] = str(combined_run_dir)
                metadata["source_combined_hidden"] = True
                source_job.spec.metadata = metadata
                if source_job.status not in ACTIVE_JOB_STATUSES:
                    source_job.status = JobStatus.cancelled
                    source_job.finished_at = now
                source_job.updated_at = now
                if not source_job.message:
                    source_job.message = f"Superseded by combined run {combined_job.job_id}"
                state["jobs"][source_job.job_id] = source_job.model_dump(mode="json")
            self._write(state)
            return combined_job

    def _combined_supervised_run_id(self, *, env_id: str, source_run_names: list[str]) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        normalized_names = [re.sub(r"-r\d+$", "", name) for name in source_run_names if name]
        prefix = os.path.commonprefix(normalized_names).strip("-_") if normalized_names else ""
        base = prefix or f"{env_id}-supervised"
        return f"{base}-combined-{timestamp}"

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            state = self._read(force_reconcile=False, allow_periodic_reconcile=True)
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            return JobRecord.model_validate(raw)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            state = self._read(force_reconcile=False, allow_periodic_reconcile=True)
            jobs = [JobRecord.model_validate(item) for item in state["jobs"].values()]
            return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def list_jobs_snapshot(self) -> list[JobRecord]:
        self._refresh_snapshot_from_disk_if_needed()
        with self._snapshot_lock:
            return list(self._jobs_snapshot)

    def _has_transient_job_message(self, job: JobRecord) -> bool:
        message = str(job.message or "")
        return any(message.startswith(prefix) for prefix in TRANSIENT_JOB_MESSAGE_PREFIXES)

    def _require_job_client_lease(self, job: JobRecord, client_id: str | None) -> None:
        normalized = str(client_id or "").strip()
        assigned = str(job.assigned_client_id or "").strip()
        if not normalized:
            if assigned:
                return
            raise ValueError(f"Job {job.job_id} is not currently leased to any client")
        if assigned == normalized:
            return
        raise ValueError(f"Job {job.job_id} is no longer assigned to client {normalized}")

    def _attach_job_to_client_record(
        self,
        state: dict[str, Any],
        client_id: str,
        job_id: str,
    ) -> bool:
        raw_client = state["clients"].get(client_id)
        if raw_client is None:
            return False
        client = ClientRecord.model_validate(raw_client)
        current = list(client.current_jobs or [])
        if job_id in current:
            return False
        current.append(job_id)
        client.current_jobs = sorted(set(current))
        state["clients"][client_id] = client.model_dump(mode="json")
        return True

    def _maybe_take_over_lease(
        self,
        state: dict[str, Any],
        *,
        job: JobRecord,
        client_id: str,
        now: datetime,
    ) -> bool:
        assigned_client_id = str(job.assigned_client_id or "").strip()
        if assigned_client_id == client_id:
            return False
        if not assigned_client_id:
            job.assigned_client_id = client_id
            state["jobs"][job.job_id] = job.model_dump(mode="json")
            self._attach_job_to_client_record(state, client_id, job.job_id)
            return True
        previous_raw = state["clients"].get(assigned_client_id)
        can_take_over = False
        if previous_raw is None:
            can_take_over = True
        else:
            previous_client = ClientRecord.model_validate(previous_raw)
            previous_jobs = set(previous_client.current_jobs or [])
            previous_last_activity = previous_client.last_poll or previous_client.last_heartbeat
            if previous_client.status in {ClientStatus.draining, ClientStatus.offline}:
                can_take_over = True
            elif self._is_stale_client_record(previous_raw, now=now):
                can_take_over = True
            elif (
                job.job_id not in previous_jobs
                and previous_last_activity is not None
                and (now - previous_last_activity).total_seconds() >= JOB_MISSING_REPORT_REQUEUE_SECONDS
            ):
                can_take_over = True
        if not can_take_over:
            return False
        self._detach_job_from_client_record(state, assigned_client_id, job.job_id)
        job.assigned_client_id = client_id
        job.updated_at = now
        if job.status in {JobStatus.assigned, JobStatus.starting}:
            job.status = JobStatus.running
            job.started_at = job.started_at or now
        state["jobs"][job.job_id] = job.model_dump(mode="json")
        self._attach_job_to_client_record(state, client_id, job.job_id)
        return True

    def _ensure_job_client_lease(
        self,
        state: dict[str, Any],
        *,
        job: JobRecord,
        client_id: str | None,
    ) -> JobRecord:
        normalized = str(client_id or "").strip()
        assigned = str(job.assigned_client_id or "").strip()
        if not normalized:
            if assigned:
                return job
            raise ValueError(f"Job {job.job_id} is not currently leased to any client")
        if assigned == normalized:
            return job
        now = _utc_now()
        if self._maybe_take_over_lease(state, job=job, client_id=normalized, now=now):
            raw = state["jobs"].get(job.job_id)
            if raw is not None:
                return JobRecord.model_validate(raw)
        raise ValueError(f"Job {job.job_id} is no longer assigned to client {normalized}")

    def migrate_imported_runs(self) -> int:
        with self._lock:
            state = self._read()
            meta = state.setdefault("meta", {})
            migrations = meta.setdefault("migrations", {})
            if migrations.get(MIGRATION_KEY):
                return 0
            stale_ids: list[str] = []
            for job_id, raw_job in state["jobs"].items():
                metadata = (
                    (raw_job.get("spec") or {}).get("metadata") or {}
                    if isinstance(raw_job, dict)
                    else {}
                )
                if metadata.get("legacy_imported") or metadata.get("migration_key") == MIGRATION_KEY:
                    stale_ids.append(job_id)
            for job_id in stale_ids:
                state["jobs"].pop(job_id, None)
                shutil.rmtree(self._job_root(job_id), ignore_errors=True)
            discovered = {
                imported.record.job_id: imported
                for imported in discover_imported_jobs(self.project_root)
            }
            changed = False
            for job_id, imported in discovered.items():
                current = state["jobs"].get(job_id)
                payload = imported.record.model_dump(mode="json")
                if current != payload:
                    state["jobs"][job_id] = payload
                    changed = True
                if imported.log_lines:
                    job_root = self._job_root(job_id)
                    job_root.mkdir(parents=True, exist_ok=True)
                    self._job_log_path(job_id).write_text(
                        "\n".join(imported.log_lines) + "\n",
                        encoding="utf-8",
                    )
            migrations[MIGRATION_KEY] = _utc_now().isoformat()
            changed = True
            if changed:
                self._write(state)
            return len(discovered)

    def poll(self, client_id: str) -> PollResponse:
        with self._lock:
            state = self._read()
            raw_client = state["clients"].get(client_id)
            if raw_client is None:
                raise KeyError(client_id)
            client = ClientRecord.model_validate(raw_client)
            poll_started_at = _utc_now()
            persist_poll_timestamp = self._should_persist_client_poll(client, now=poll_started_at)
            state_changed = False
            queue_settings = QueueSettings.model_validate(state.get("queue_settings") or {})
            assignments: list[JobRecord] = []
            commands: list[JobCommand] = []
            client_commands = [
                ClientCommand.model_validate(item)
                for item in state.setdefault("client_commands", {}).get(client_id, [])
            ]
            reported_current_jobs = set(client.current_jobs)
            commands.extend(
                self._commands_for_reported_job_mismatches(
                    state,
                    client_id,
                    reported_current_jobs=reported_current_jobs,
                )
            )

            for raw_job in state["jobs"].values():
                job = JobRecord.model_validate(raw_job)
                if (
                    job.assigned_client_id != client_id
                    and job.job_id not in reported_current_jobs
                ):
                    continue
                if (
                    job.job_id not in reported_current_jobs
                    and job.status not in ACTIVE_JOB_STATUSES
                ):
                    continue
                commands.extend(job.pending_commands)

            if client_commands:
                if persist_poll_timestamp or state_changed:
                    client.last_poll = poll_started_at
                    state["clients"][client_id] = client.model_dump(mode="json")
                    self._write(state)
                return PollResponse(
                    assignments=[],
                    commands=commands,
                    client_commands=client_commands,
                    queue_settings=queue_settings,
                    client_max_concurrency=self._effective_client_max_concurrency(client),
                    suggested_poll_delay_seconds=self._suggested_poll_delay_seconds(
                        client,
                        assignments=[],
                        commands=commands,
                        client_commands=client_commands,
                    ),
                )

            pending_assignments = self._pending_assignments_for_client(
                state,
                client_id,
                reported_current_jobs=reported_current_jobs,
            )
            for pending_job in pending_assignments:
                pending_job, changed = self._rewrite_job_paths_for_client(pending_job, client)
                if changed:
                    state["jobs"][pending_job.job_id] = pending_job.model_dump(mode="json")
                    state_changed = True
                assignments.append(pending_job)
            if persist_poll_timestamp or state_changed:
                client.last_poll = poll_started_at
                state["clients"][client_id] = client.model_dump(mode="json")
                self._write(state)
            return PollResponse(
                assignments=assignments,
                commands=commands,
                client_commands=client_commands,
                queue_settings=queue_settings,
                client_max_concurrency=self._effective_client_max_concurrency(client),
                suggested_poll_delay_seconds=self._suggested_poll_delay_seconds(
                    client,
                    assignments=assignments,
                    commands=commands,
                    client_commands=client_commands,
                ),
            )

    def schedule_pending_jobs(self) -> bool:
        with self._lock:
            state = self._read()
            changed = False
            clients = self._clients_from_state(state)
            active_jobs = self._active_jobs_from_state(state)
            if self._normalize_repeat_learning_queue(state):
                changed = True
            if self._enforce_all_client_slot_limits(
                state,
                reason="Auto-paused after client exceeded configured slot limit",
                active_jobs=active_jobs,
                clients=clients,
            ):
                changed = True
                active_jobs = self._active_jobs_from_state(state)
            if self._requeue_capacity_paused_jobs(state):
                changed = True
            if self._prune_redundant_requeue_commands(state):
                changed = True

            queued_jobs = sorted(
                (
                    JobRecord.model_validate(raw_job)
                    for raw_job in state["jobs"].values()
                ),
                key=self._dispatch_sort_key,
            )
            now = _utc_now()
            for job in queued_jobs:
                if job.status != JobStatus.queued:
                    continue
                best_client_id = self._best_client_id_for_job(
                    job,
                    active_jobs=active_jobs,
                    clients=clients,
                )
                if not best_client_id:
                    continue
                raw_client = state["clients"].get(best_client_id)
                if raw_client is None:
                    continue
                job.status = JobStatus.assigned
                job.assigned_client_id = best_client_id
                job.updated_at = now
                if self._has_transient_job_message(job):
                    job.message = None
                state["jobs"][job.job_id] = job.model_dump(mode="json")
                active_jobs.append(job)
                changed = True

            if changed:
                self._write(state)
            return changed

    def _should_persist_client_poll(self, client: ClientRecord, *, now: datetime) -> bool:
        last_poll = client.last_poll
        if last_poll is None:
            return True
        return (now - last_poll).total_seconds() >= CLIENT_POLL_RECORD_INTERVAL_SECONDS

    def _suggested_poll_delay_seconds(
        self,
        client: ClientRecord,
        *,
        assignments: list[JobRecord],
        commands: list[JobCommand],
        client_commands: list[ClientCommand],
    ) -> float:
        if client.kind == ClientKind.direct:
            return DIRECT_POLL_DELAY_SECONDS
        if assignments or commands or client_commands:
            return REMOTE_BUSY_POLL_DELAY_SECONDS
        if client.current_jobs:
            return REMOTE_BUSY_POLL_DELAY_SECONDS
        return REMOTE_IDLE_POLL_DELAY_SECONDS

    def acknowledge_client_command(self, client_id: str, command_id: str) -> ClientRecord:
        with self._lock:
            state = self._read()
            raw = state["clients"].get(client_id)
            if raw is None:
                raise KeyError(client_id)
            pending = [
                ClientCommand.model_validate(item)
                for item in state.setdefault("client_commands", {}).get(client_id, [])
            ]
            state["client_commands"][client_id] = [
                command.model_dump(mode="json")
                for command in pending
                if command.command_id != command_id
            ]
            client = ClientRecord.model_validate(raw)
            self._write(state)
            return client

    def acknowledge_command(self, job_id: str, command_id: str) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            job.pending_commands = [
                command
                for command in job.pending_commands
                if command.command_id != command_id
            ]
            job.updated_at = _utc_now()
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def pause_job(self, job_id: str) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            if job.status == JobStatus.queued:
                job.status = JobStatus.paused
                job.updated_at = _utc_now()
            elif job.status in ACTIVE_JOB_STATUSES:
                job.status = JobStatus.pausing
                job.updated_at = _utc_now()
                self._append_command(job, JobCommandType.pause)
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def cancel_job(self, job_id: str) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            if job.status in {JobStatus.queued, JobStatus.paused, JobStatus.failed}:
                job.status = JobStatus.cancelled
                job.finished_at = _utc_now()
                job.updated_at = _utc_now()
            elif job.status in ACTIVE_JOB_STATUSES:
                job.status = JobStatus.stopping
                job.updated_at = _utc_now()
                self._append_command(job, JobCommandType.cancel)
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def hide_repeat_learning_runs(self, job_ids: list[str]) -> list[JobRecord]:
        normalized_job_ids: list[str] = []
        seen_ids: set[str] = set()
        for raw_job_id in job_ids:
            job_id = str(raw_job_id or "").strip()
            if not job_id or job_id in seen_ids:
                continue
            seen_ids.add(job_id)
            normalized_job_ids.append(job_id)
        if not normalized_job_ids:
            raise ValueError("Select at least one run")

        with self._lock:
            state = self._read()
            now = _utc_now()
            updated_jobs: list[JobRecord] = []
            for job_id in normalized_job_ids:
                raw = state["jobs"].get(job_id)
                if raw is None:
                    raise KeyError(job_id)
                job = JobRecord.model_validate(raw)
                action = str(job.spec.action or "").strip().lower()
                if action not in REPEAT_LEARNING_ACTIONS:
                    raise ValueError(
                        f"Run deletion is only supported for learning runs; {job_id} is {job.spec.action}"
                    )
                metadata = dict(job.spec.metadata or {})
                metadata[HIDDEN_REQUIRED_RUN_METADATA_KEY] = True
                job.spec.metadata = metadata
                if job.status == JobStatus.queued:
                    job.status = JobStatus.paused
                    job.assigned_client_id = None
                    job.finished_at = now
                    job.pending_commands = []
                    job.message = "Hidden from required runs view"
                elif job.status in ACTIVE_JOB_STATUSES:
                    job.status = JobStatus.pausing
                    self._append_command(job, JobCommandType.pause)
                    job.message = "Hidden from required runs view; pausing run"
                else:
                    job.status = JobStatus.paused
                    job.assigned_client_id = None
                    job.finished_at = now
                    job.pending_commands = []
                    job.message = "Hidden from required runs view"
                job.updated_at = now
                state["jobs"][job_id] = job.model_dump(mode="json")
                updated_jobs.append(job)
            self._write(state)
            return updated_jobs

    def requeue_job(self, job_id: str) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            if job.status in {JobStatus.failed, JobStatus.paused, JobStatus.cancelled, JobStatus.stopped}:
                job.status = JobStatus.queued
                job.updated_at = _utc_now()
                job.finished_at = None
                job.exit_code = None
                job.assigned_client_id = None
                job.message = None
                metadata = dict(job.spec.metadata or {})
                metadata.pop("progress", None)
                job.spec.metadata = metadata
                resume_argv = job.spec.metadata.get("resume_argv")
                if isinstance(resume_argv, list) and resume_argv:
                    job.spec.argv = [str(part) for part in resume_argv]
            elif job.status in ACTIVE_JOB_STATUSES:
                job.status = JobStatus.stopping
                job.updated_at = _utc_now()
                self._append_command(job, JobCommandType.requeue)
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def release_job_assignment(self, job_id: str, *, message: str | None = None) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            if job.status in ACTIVE_JOB_STATUSES or job.status == JobStatus.assigned:
                metadata = dict(job.spec.metadata or {})
                resume_argv = metadata.get("resume_argv")
                if isinstance(resume_argv, list) and resume_argv:
                    job.spec.argv = [str(part) for part in resume_argv]
                job.status = JobStatus.queued
                job.assigned_client_id = None
                job.updated_at = _utc_now()
                job.started_at = None
                job.finished_at = None
                job.exit_code = None
                job.pending_commands = []
                if message is not None:
                    job.message = message
                state["jobs"][job_id] = job.model_dump(mode="json")
                self._write(state)
            return job

    def set_job_preferred_client(self, job_id: str, client_id: str | None) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            metadata = dict(job.spec.metadata or {})
            normalized_client_id = (client_id or "").strip()
            if normalized_client_id:
                raw_client = state["clients"].get(normalized_client_id)
                if raw_client is None:
                    raise KeyError(normalized_client_id)
                client = ClientRecord.model_validate(raw_client)
                metadata["preferred_client_id"] = client.client_id
                metadata["preferred_client_name"] = client.display_name
            else:
                metadata.pop("preferred_client_id", None)
                metadata.pop("preferred_client_name", None)
            job.spec.metadata = metadata
            job.updated_at = _utc_now()
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def resume_job(self, job_id: str) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            if job.status not in {JobStatus.paused, JobStatus.failed, JobStatus.stopped}:
                return job
            job.status = JobStatus.queued
            job.updated_at = _utc_now()
            job.finished_at = None
            job.exit_code = None
            job.message = None
            job.assigned_client_id = None
            metadata = dict(job.spec.metadata or {})
            metadata.pop("progress", None)
            job.spec.metadata = metadata
            resume_argv = job.spec.metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def update_job(self, job_id: str, request: JobUpdateRequest) -> JobRecord:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            job = self._ensure_job_client_lease(state, job=job, client_id=request.client_id)
            previous_status = job.status
            job.status = request.status
            job.updated_at = _utc_now()
            job.exit_code = request.exit_code
            if request.message is not None:
                job.message = request.message
            if request.metadata_patch:
                job.spec.metadata.update(request.metadata_patch)
            if request.status in {JobStatus.starting, JobStatus.running} and job.started_at is None:
                job.started_at = _utc_now()
            if request.status in {JobStatus.starting, JobStatus.running}:
                if request.message is None and self._has_transient_job_message(job):
                    job.message = None
            if request.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
                JobStatus.paused,
                JobStatus.stopped,
            }:
                job.finished_at = _utc_now()
                job.assigned_client_id = None
            elif request.status == JobStatus.queued:
                job.finished_at = None
                job.started_at = None
                job.assigned_client_id = None
                job.exit_code = None
            elif previous_status in {
                JobStatus.failed,
                JobStatus.succeeded,
                JobStatus.paused,
                JobStatus.stopped,
            } and request.status in {JobStatus.resuming, JobStatus.running, JobStatus.starting}:
                job.finished_at = None
            if request.status in {JobStatus.queued, JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled, JobStatus.paused, JobStatus.stopped}:
                job.pending_commands = []
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def append_job_logs(self, job_id: str, request: JobLogChunkRequest) -> JobRecord:
        if not request.lines:
            return self.get_job(job_id)
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            self._ensure_job_client_lease(state, job=job, client_id=request.client_id)
        normalized_lines = [str(line).rstrip("\n") for line in request.lines]
        log_path = self._job_log_path(job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            for line in normalized_lines:
                handle.write(line + "\n")
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            job = self._ensure_job_client_lease(state, job=job, client_id=request.client_id)
            job.recent_logs.extend(normalized_lines)
            if len(job.recent_logs) > MAX_RECENT_LOG_LINES:
                job.recent_logs = job.recent_logs[-MAX_RECENT_LOG_LINES:]
            job.updated_at = _utc_now()
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job

    def read_job_logs(self, job_id: str, *, limit: int = 200) -> list[str]:
        log_path = self._job_log_path(job_id)
        if not log_path.is_file():
            return []
        if limit <= 0:
            return []
        return self._tail_text_lines(log_path, limit=limit)

    def extract_artifact_bundle(
        self,
        job_id: str,
        *,
        archive_bytes: bytes,
        client_id: str | None = None,
    ) -> tuple[JobRecord, list[str]]:
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            job = self._ensure_job_client_lease(state, job=job, client_id=client_id)
            if job.status not in ARTIFACT_UPLOAD_ALLOWED_STATUSES:
                raise ValueError(
                    f"Job {job.job_id} does not accept artifact uploads while status is {job.status.value}"
                )
        extracted = self._extract_bundle_bytes(archive_bytes)
        with self._lock:
            state = self._read()
            raw = state["jobs"].get(job_id)
            if raw is None:
                raise KeyError(job_id)
            job = JobRecord.model_validate(raw)
            job = self._ensure_job_client_lease(state, job=job, client_id=client_id)
            if job.status not in ARTIFACT_UPLOAD_ALLOWED_STATUSES:
                raise ValueError(
                    f"Job {job.job_id} does not accept artifact uploads while status is {job.status.value}"
                )
            known = set(job.artifact_relpaths)
            compacted = self._compact_artifact_relpaths(extracted)
            for relpath in compacted:
                if relpath not in known:
                    job.artifact_relpaths.append(relpath)
                    known.add(relpath)
            job.updated_at = _utc_now()
            state["jobs"][job_id] = job.model_dump(mode="json")
            self._write(state)
            return job, extracted

    def _append_command(self, job: JobRecord, command: JobCommandType) -> None:
        for existing in job.pending_commands:
            if existing.command == command:
                return
        job.pending_commands.append(
            JobCommand(
                command_id=uuid.uuid4().hex,
                job_id=job.job_id,
                command=command,
                created_at=_utc_now(),
            )
        )

    def _extract_bundle_bytes(self, archive_bytes: bytes) -> list[str]:
        extracted_paths: list[str] = []
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                member_path = member.name.strip().lstrip("/")
                if not member_path:
                    continue
                target_path = (self.project_root / member_path).resolve()
                if not str(target_path).startswith(str(self.project_root.resolve())):
                    raise ValueError(f"Refusing to extract outside project root: {member.name}")
                extracted_paths.append(member_path)
            tar.extractall(path=self.project_root, filter="data")
        return extracted_paths

    def _job_root(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _job_log_path(self, job_id: str) -> Path:
        return self._job_root(job_id) / "job.log"

    def _tail_text_lines(self, path: Path, *, limit: int) -> list[str]:
        if limit <= 0:
            return []
        with path.open("rb") as handle:
            handle.seek(0, io.SEEK_END)
            remaining = handle.tell()
            if remaining <= 0:
                return []
            chunks: list[bytes] = []
            newline_count = 0
            while remaining > 0 and newline_count <= limit:
                chunk_size = min(8192, remaining)
                remaining -= chunk_size
                handle.seek(remaining)
                chunk = handle.read(chunk_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
        text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
        return text.splitlines()[-limit:]

    def _is_pooltool_job(self, job: JobRecord) -> bool:
        action = str(job.spec.action or "")
        env_id = str(job.spec.metadata.get("env_id") or "")
        return action.startswith("pooltool-") or env_id == "pooltool"

    def _is_reward_job(self, job: JobRecord) -> bool:
        return str(job.spec.action or "").startswith("pooltool-reward-")

    def _client_is_reward_reserved(self, client: ClientRecord) -> bool:
        tags = {str(tag).strip().lower() for tag in (client.capabilities.tags or [])}
        display_name = str(client.display_name or "").strip().lower()
        return "reward" in tags or display_name.startswith("reward-")

    def _client_is_required_reserved(self, client: ClientRecord) -> bool:
        tags = {str(tag).strip().lower() for tag in (client.capabilities.tags or [])}
        display_name = str(client.display_name or "").strip().lower()
        return "required" in tags or display_name.startswith("required-")

    def _client_is_ablation_reserved(self, client: ClientRecord) -> bool:
        tags = {str(tag).strip().lower() for tag in (client.capabilities.tags or [])}
        display_name = str(client.display_name or "").strip().lower()
        return "ablation" in tags or display_name.startswith("ablation-")

    def _client_is_analysis_reserved(self, client: ClientRecord) -> bool:
        tags = {str(tag).strip().lower() for tag in (client.capabilities.tags or [])}
        display_name = str(client.display_name or "").strip().lower()
        return "analysis" in tags or display_name.startswith("analysis-")

    def _client_is_preferred_only(self, client: ClientRecord) -> bool:
        tags = {str(tag).strip().lower() for tag in (client.capabilities.tags or [])}
        display_name = str(client.display_name or "").strip().lower()
        return "preferred-only" in tags or display_name.startswith("preferred-only-")

    def _is_required_run_job(self, job: JobRecord) -> bool:
        metadata = dict(job.spec.metadata or {})
        if bool(metadata.get(HIDDEN_REQUIRED_RUN_METADATA_KEY)):
            return False
        if self._repeat_learning_descriptor(job) is not None:
            return True
        if self._repeat_eval_descriptor(job) is not None:
            return True
        if self._is_phyre_required_image_only_eval(job):
            return True
        return False

    def _is_ablation_job(self, job: JobRecord) -> bool:
        return str(job.spec.action or "").strip() == "evaluate-library-case"

    def _is_analysis_job(self, job: JobRecord) -> bool:
        return str(job.spec.action or "").strip() == "analyze-supervised-consistency"

    def _dispatch_sort_key(self, job: JobRecord) -> tuple[int, int, int, datetime]:
        metadata = dict(job.spec.metadata or {})
        launch_mode = str(metadata.get("launch_mode") or "")
        immediate_rank = 0 if launch_mode == "run_now" else 1
        action = str(job.spec.action or "")
        priority = int(QUEUE_DISPATCH_PRIORITY.get(action, 99))
        queue_order_raw = metadata.get("queue_order")
        try:
            queue_order = int(queue_order_raw)
        except (TypeError, ValueError):
            queue_order = 0
        return (immediate_rank, priority, queue_order, job.created_at)

    def _client_active_jobs(self, client_id: str, active_jobs: list[JobRecord]) -> list[JobRecord]:
        return [job for job in active_jobs if job.assigned_client_id == client_id]

    def _job_counts_toward_slot_limit(self, job: JobRecord) -> bool:
        return job.status not in {JobStatus.pausing, JobStatus.stopping}

    def _client_effective_busy_count(
        self,
        client: ClientRecord,
        active_jobs: list[JobRecord],
    ) -> int:
        # Prefer the higher of server-side active assignments and the client's own
        # reported current jobs so routing does not overfill a worker when state
        # reconciliation is briefly behind live execution.
        active_count = sum(
            1
            for job in self._client_active_jobs(client.client_id, active_jobs)
            if self._job_counts_toward_slot_limit(job)
        )
        reported_count = len(client.current_jobs or [])
        return max(active_count, reported_count)

    def _client_has_active_pooltool_job(
        self,
        client_id: str,
        active_jobs: list[JobRecord],
    ) -> bool:
        return any(
            self._is_pooltool_job(job) and self._job_counts_toward_slot_limit(job)
            for job in self._client_active_jobs(client_id, active_jobs)
        )

    def _effective_client_max_concurrency(self, client: ClientRecord) -> int:
        return max(1, int(client.max_concurrency or 1))

    def _client_busy_key(self, client: ClientRecord, active_jobs: list[JobRecord]) -> tuple[float, int, str, str]:
        active_count = self._client_effective_busy_count(client, active_jobs)
        max_concurrency = self._effective_client_max_concurrency(client)
        utilization = active_count / max_concurrency
        return (utilization, active_count, client.display_name.lower(), client.client_id)

    def _active_jobs_from_state(self, state: dict[str, Any]) -> list[JobRecord]:
        jobs = [JobRecord.model_validate(item) for item in state["jobs"].values()]
        return [job for job in jobs if job.status in ACTIVE_JOB_STATUSES]

    def _clients_from_state(self, state: dict[str, Any]) -> list[ClientRecord]:
        return [
            ClientRecord.model_validate(item)
            for item in state.get("clients", {}).values()
        ]

    def _client_can_run(
        self,
        client: ClientRecord,
        job: JobRecord,
        *,
        active_jobs: list[JobRecord] | None = None,
    ) -> bool:
        if not self._client_hard_compatible_with_job(client, job):
            return False
        active_jobs = active_jobs or []
        if self._client_effective_busy_count(client, active_jobs) >= self._effective_client_max_concurrency(client):
            return False
        if (
            self._is_pooltool_job(job)
            and not self._is_reward_job(job)
            and self._client_has_active_pooltool_job(client.client_id, active_jobs)
        ):
            return False
        return True

    def _client_hard_compatible_with_job(self, client: ClientRecord, job: JobRecord) -> bool:
        constraints = job.spec.constraints
        caps = client.capabilities
        metadata = dict(job.spec.metadata or {})
        preferred_client_id = str(metadata.get("preferred_client_id") or "").strip()
        if client.status != ClientStatus.online:
            return False
        if self._job_avoids_client(job, client.client_id):
            return False
        action = str(job.spec.action or "").strip()
        if client.kind == ClientKind.direct:
            if action not in DIRECT_CLIENT_ACTIONS:
                return False
            if self._client_is_required_reserved(client) and not self._is_required_run_job(job):
                return False
        if action in REMOTE_FORBIDDEN_ACTIONS and client.kind != ClientKind.direct:
            return False
        if self._client_is_reward_reserved(client) and not self._is_reward_job(job):
            return False
        if self._client_is_ablation_reserved(client) and not self._is_ablation_job(job):
            return False
        if self._client_is_analysis_reserved(client) and not self._is_analysis_job(job):
            return False
        if self._client_is_preferred_only(client) and preferred_client_id != client.client_id:
            return False
        if preferred_client_id and preferred_client_id != client.client_id:
            return False
        if constraints.gpu_required and not caps.gpu:
            return False
        if constraints.required_environments:
            # Some older/remotely-managed clients can temporarily report an empty
            # environment list even though they are otherwise healthy. Treat
            # "no environments reported" as unknown rather than incompatible so
            # runnable queued work can still dispatch.
            if caps.environments and not set(constraints.required_environments).issubset(
                set(caps.environments)
            ):
                return False
        if constraints.required_tags:
            if not set(constraints.required_tags).issubset(set(caps.tags)):
                return False
        return True

    def _mark_job_avoid_client(
        self,
        job: JobRecord,
        *,
        client_id: str | None,
        now: datetime | None = None,
    ) -> None:
        normalized = str(client_id or "").strip()
        metadata = dict(job.spec.metadata or {})
        if not normalized:
            metadata.pop("avoid_client_id", None)
            metadata.pop("avoid_client_until", None)
            job.spec.metadata = metadata
            return
        reference = now or _utc_now()
        metadata["avoid_client_id"] = normalized
        metadata["avoid_client_until"] = (
            reference + timedelta(seconds=JOB_REDISPATCH_GUARD_SECONDS)
        ).isoformat()
        job.spec.metadata = metadata

    def _job_avoids_client(self, job: JobRecord, client_id: str | None) -> bool:
        normalized = str(client_id or "").strip()
        if not normalized:
            return False
        metadata = dict(job.spec.metadata or {})
        avoid_client_id = str(metadata.get("avoid_client_id") or "").strip()
        if avoid_client_id != normalized:
            return False
        raw_until = str(metadata.get("avoid_client_until") or "").strip()
        if not raw_until:
            return True
        try:
            avoid_until = datetime.fromisoformat(raw_until.replace("Z", "+00:00"))
        except ValueError:
            return True
        if avoid_until.tzinfo is None:
            avoid_until = avoid_until.replace(tzinfo=timezone.utc)
        return _utc_now() < avoid_until

    def _queued_job_can_be_reclaimed_by_reporting_client(
        self,
        job: JobRecord,
        *,
        client_id: str,
    ) -> bool:
        if job.status != JobStatus.queued:
            return False
        metadata = dict(job.spec.metadata or {})
        avoid_client_id = str(metadata.get("avoid_client_id") or "").strip()
        if not avoid_client_id or avoid_client_id == str(client_id or "").strip():
            return False
        message = str(job.message or "")
        return message.startswith("Recovered after client ") or message.startswith(
            "Re-queued after client "
        )

    def _best_client_id_for_job(
        self,
        job: JobRecord,
        *,
        active_jobs: list[JobRecord],
        clients: list[ClientRecord],
    ) -> str | None:
        compatible_clients: list[ClientRecord] = []
        for client in clients:
            if self._client_can_run(
                client,
                job,
                active_jobs=active_jobs,
            ):
                compatible_clients.append(client)
        if not compatible_clients:
            return None
        if self._is_reward_job(job):
            reserved_clients = [
                client for client in compatible_clients if self._client_is_reward_reserved(client)
            ]
            if reserved_clients:
                compatible_clients = reserved_clients
        compatible_clients.sort(key=lambda client: self._client_busy_key(client, active_jobs))
        return compatible_clients[0].client_id

    def _enforce_all_client_slot_limits(
        self,
        state: dict[str, Any],
        *,
        reason: str,
        active_jobs: list[JobRecord] | None = None,
        clients: list[ClientRecord] | None = None,
    ) -> bool:
        changed = False
        active_jobs = active_jobs if active_jobs is not None else self._active_jobs_from_state(state)
        client_records = clients if clients is not None else self._clients_from_state(state)
        for client in client_records:
            if self._enforce_client_slot_limit(
                state,
                client.client_id,
                reason=reason,
                active_jobs=active_jobs,
            ):
                changed = True
        return changed

    def _enforce_client_slot_limit(
        self,
        state: dict[str, Any],
        client_id: str,
        *,
        reason: str,
        active_jobs: list[JobRecord] | None = None,
    ) -> bool:
        raw_client = state["clients"].get(client_id)
        if raw_client is None:
            return False
        client = ClientRecord.model_validate(raw_client)
        limit = self._effective_client_max_concurrency(client)
        relevant_active_jobs = active_jobs if active_jobs is not None else self._active_jobs_from_state(state)
        active_jobs = [
            job
            for job in self._client_active_jobs(client_id, relevant_active_jobs)
            if self._job_counts_toward_slot_limit(job)
        ]
        if len(active_jobs) <= limit:
            return False
        overflow_count = len(active_jobs) - limit
        overflow_jobs = sorted(
            active_jobs,
            key=lambda job: (self._job_recency_key(job), job.job_id),
            reverse=True,
        )[:overflow_count]
        changed = False
        for job in overflow_jobs:
            if self._auto_pause_overflow_job(state, job.job_id, reason=reason):
                changed = True
        return changed

    def _auto_pause_overflow_job(self, state: dict[str, Any], job_id: str, *, reason: str) -> bool:
        raw = state["jobs"].get(job_id)
        if raw is None:
            return False
        job = JobRecord.model_validate(raw)
        if job.status == JobStatus.queued:
            job.status = JobStatus.paused
            job.assigned_client_id = None
            job.finished_at = _utc_now()
            job.updated_at = _utc_now()
            job.message = reason
            job.pending_commands = []
        elif job.status in ACTIVE_JOB_STATUSES:
            job.status = JobStatus.pausing
            job.updated_at = _utc_now()
            job.message = reason
            self._append_command(job, JobCommandType.pause)
        else:
            return False
        state["jobs"][job_id] = job.model_dump(mode="json")
        return True

    def _job_recency_key(self, job: JobRecord) -> tuple[int, datetime]:
        metadata = dict(job.spec.metadata or {})
        try:
            queue_order = int(metadata.get("queue_order") or 0)
        except (TypeError, ValueError):
            queue_order = 0
        return (queue_order, job.created_at)

    def _reconcile_dependency_blocked_jobs(self, state: dict[str, Any]) -> bool:
        changed = False
        for job_id, raw_job in list(state.get("jobs", {}).items()):
            job = JobRecord.model_validate(raw_job)
            blocked_reason = self._job_prerequisite_missing_reason(job)
            waiting_prefix = "Waiting for prerequisite: "
            message_text = str(job.message or "").lower()
            if blocked_reason:
                if job.status == JobStatus.queued:
                    job.status = JobStatus.paused
                    job.assigned_client_id = None
                    job.finished_at = None
                    job.updated_at = _utc_now()
                    job.message = waiting_prefix + blocked_reason
                    job.pending_commands = []
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                    continue
                if job.status == JobStatus.paused and str(job.message or "").startswith(waiting_prefix):
                    continue
                if (
                    job.status == JobStatus.cancelled
                    and self._repeat_eval_descriptor(job) is not None
                    and (
                        "standalone repeat evaluation" in message_text
                        or "embedded auto-evals" in message_text
                    )
                ):
                    continue
                if job.status in {JobStatus.failed, JobStatus.stopped, JobStatus.cancelled}:
                    job.status = JobStatus.paused
                    job.assigned_client_id = None
                    job.finished_at = None
                    job.updated_at = _utc_now()
                    job.exit_code = None
                    job.message = waiting_prefix + blocked_reason
                    job.pending_commands = []
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                    continue
                if job.status in {JobStatus.assigned, JobStatus.starting}:
                    job.status = JobStatus.paused
                    job.assigned_client_id = None
                    job.finished_at = _utc_now()
                    job.updated_at = _utc_now()
                    job.message = waiting_prefix + blocked_reason
                    job.pending_commands = []
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                    continue
                if job.status in ACTIVE_JOB_STATUSES:
                    if job.status == JobStatus.pausing and not job.pending_commands:
                        job.status = JobStatus.paused
                        job.assigned_client_id = None
                        job.finished_at = _utc_now()
                        job.updated_at = _utc_now()
                        job.message = waiting_prefix + blocked_reason
                        state["jobs"][job_id] = job.model_dump(mode="json")
                        changed = True
                        continue
                    job.status = JobStatus.pausing
                    job.updated_at = _utc_now()
                    job.message = waiting_prefix + blocked_reason
                    self._append_command(job, JobCommandType.pause)
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                    continue
            if (
                not blocked_reason
                and job.status == JobStatus.paused
                and str(job.message or "").startswith(waiting_prefix)
            ):
                job.status = JobStatus.queued
                job.updated_at = _utc_now()
                job.finished_at = None
                job.message = "Dependency ready; re-queued automatically"
                metadata = dict(job.spec.metadata or {})
                metadata.pop("progress", None)
                job.spec.metadata = metadata
                state["jobs"][job_id] = job.model_dump(mode="json")
                changed = True
        return changed

    def _requeue_capacity_paused_jobs(self, state: dict[str, Any]) -> bool:
        changed = False
        clients = self._clients_from_state(state)
        active_jobs = self._active_jobs_from_state(state)
        paused_jobs = sorted(
            (
                JobRecord.model_validate(raw_job)
                for raw_job in state.get("jobs", {}).values()
            ),
            key=self._dispatch_sort_key,
        )
        for job in paused_jobs:
            if not self._is_capacity_paused_job(job):
                continue
            if self._job_prerequisite_missing_reason(job):
                continue
            best_client_id = self._best_client_id_for_job(
                job,
                active_jobs=active_jobs,
                clients=clients,
            )
            if best_client_id is None:
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            job.status = JobStatus.queued
            job.assigned_client_id = None
            job.updated_at = _utc_now()
            job.finished_at = None
            job.exit_code = None
            job.pending_commands = []
            job.message = "Capacity available; re-queued automatically"
            state["jobs"][job.job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _is_capacity_paused_job(self, job: JobRecord) -> bool:
        if job.status != JobStatus.paused:
            return False
        message = str(job.message or "")
        return any(message.startswith(prefix) for prefix in AUTO_RESUME_PAUSE_PREFIXES)

    def _results_payload_has_reasoning_traces(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        containers: list[dict[str, Any]] = []
        results = payload.get("results")
        if isinstance(results, list):
            containers.extend(item for item in results if isinstance(item, dict))
        records = payload.get("records")
        if isinstance(records, list):
            containers.append(payload)
        for container in containers:
            container_records = container.get("records")
            if not isinstance(container_records, list):
                continue
            for record in container_records:
                if not isinstance(record, dict):
                    continue
                if isinstance(record.get("raw_model_output"), str) and isinstance(
                    record.get("trace_description"), str
                ):
                    return True
        return False

    def _results_path_has_reasoning_traces(self, results_path: Path) -> bool:
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if self._results_payload_has_reasoning_traces(payload):
            return True
        tasks_root = results_path.parent / "tasks"
        if not tasks_root.is_dir():
            return False
        for task_result_path in sorted(tasks_root.glob("*/*/task_result.json")):
            try:
                task_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if self._results_payload_has_reasoning_traces(task_payload):
                return True
        return False

    def _candidate_pattern_run_dirs_for_prerequisite(self, pattern_run_dir: str) -> list[Path]:
        raw = str(pattern_run_dir or "").strip()
        if not raw:
            return []
        normalized = raw.replace("\\", "/")
        candidates: list[Path] = [Path(raw)]
        project_root = self.project_root.resolve()
        if project_root.name == "pattern-learning-server":
            local_server_root = project_root
            local_services_root = project_root.parent
        else:
            local_server_root = (project_root / "pattern-learning-server").resolve()
            local_services_root = project_root
        local_client_root = (local_services_root / "pattern-learning-client").resolve()
        remaps = [
            ("/opt/pattern-learning-stack/pattern-learning-server/data/", (local_server_root / "data").resolve()),
            ("/var/opt/pattern-learning-stack/pattern-learning-server/data/", (local_server_root / "data").resolve()),
            ("/opt/pattern-learning-stack/pattern-learning-server/", local_server_root),
            ("/var/opt/pattern-learning-stack/pattern-learning-server/", local_server_root),
            ("/opt/pattern-learning-stack/pattern-learning-client/data/", (local_client_root / "data").resolve()),
            ("/var/opt/pattern-learning-stack/pattern-learning-client/data/", (local_client_root / "data").resolve()),
            ("/opt/pattern-learning-stack/pattern-learning-client/", local_client_root),
            ("/var/opt/pattern-learning-stack/pattern-learning-client/", local_client_root),
            ("/opt/pattern-learning-stack/data/", (local_server_root / "data").resolve()),
            ("/var/opt/pattern-learning-stack/data/", (local_server_root / "data").resolve()),
            ("/opt/pattern-learning-stack/", local_client_root),
            ("/var/opt/pattern-learning-stack/", local_client_root),
        ]
        for source_prefix, target_root in remaps:
            if not normalized.startswith(source_prefix):
                continue
            suffix = normalized[len(source_prefix):].lstrip("/")
            if not suffix:
                continue
            remapped = (target_root / Path(suffix)).resolve()
            if remapped not in candidates:
                candidates.append(remapped)
        work_root_mappings = [
            ("/Services/Work/pattern-learning-server/data/", (self.project_root / "data").resolve()),
            ("/Services/Work/pattern-learning-server/", self.project_root.resolve()),
            ("/Services/Work/pattern-learning-client/data/", (local_client_root / "data").resolve()),
            ("/Services/Work/pattern-learning-client/", local_client_root),
        ]
        for marker, target_root in work_root_mappings:
            if marker not in normalized:
                continue
            suffix = normalized.split(marker, 1)[1].lstrip("/")
            remapped = (target_root / Path(suffix)).resolve()
            if remapped not in candidates:
                candidates.append(remapped)
        return candidates

    def _candidate_supervised_source_run_dirs_for_automatic(self, source_run_dir: str) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def add(path: Path) -> None:
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            candidates.append(resolved)

        for candidate in self._candidate_pattern_run_dirs_for_prerequisite(source_run_dir):
            add(candidate)
            parts = list(candidate.parts)
            if "patterns" in parts and "supervised" in parts:
                supervised_index = parts.index("supervised")
                shortened = Path(*parts[:supervised_index], *parts[supervised_index + 1 :])
                add(shortened)

        expanded: list[Path] = []
        expanded_seen: set[Path] = set()
        for candidate in candidates:
            if candidate not in expanded_seen:
                expanded.append(candidate)
                expanded_seen.add(candidate)
            if candidate.exists():
                for sibling in sorted(candidate.parent.glob(f"{candidate.name}.__resume_sync__*")):
                    resolved = sibling.resolve()
                    if resolved in expanded_seen:
                        continue
                    expanded.append(resolved)
                    expanded_seen.add(resolved)
        return expanded

    def _pattern_library_exists(self, pattern_run_dir: str) -> bool:
        for run_dir in self._candidate_pattern_run_dirs_for_prerequisite(pattern_run_dir):
            if (run_dir / "pattern_library.json").is_file():
                return True
        return False

    def _job_prerequisite_missing_reason(self, job: JobRecord) -> str | None:
        action = str(job.spec.action or "").strip()
        metadata = dict(job.spec.metadata or {})
        if action == "evaluate-deepphy":
            pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip()
            if not pattern_run_dir:
                request = metadata.get("request")
                if isinstance(request, dict):
                    pattern_run_dir = str(request.get("pattern_run_dir") or "").strip()
            if not pattern_run_dir:
                return None
            if not self._pattern_library_exists(pattern_run_dir):
                missing_path = Path(pattern_run_dir) / "pattern_library.json"
                return f"{missing_path} not found"
        if action == "evaluate-library-case":
            pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip()
            if not pattern_run_dir:
                request = metadata.get("request")
                if isinstance(request, dict):
                    pattern_run_dir = str(request.get("pattern_run_dir") or "").strip()
            if not pattern_run_dir:
                return None
            if not self._pattern_library_exists(pattern_run_dir):
                missing_path = Path(pattern_run_dir) / "pattern_library.json"
                return f"{missing_path} not found"
        if action == "analyze-supervised-consistency":
            request = metadata.get("request")
            request_dict = request if isinstance(request, dict) else {}
            raw_trace_paths = metadata.get("trace_paths")
            if not isinstance(raw_trace_paths, list):
                raw_trace_paths = request_dict.get("trace_paths")
            trace_paths = (
                [str(item or "").strip() for item in raw_trace_paths if str(item or "").strip()]
                if isinstance(raw_trace_paths, list)
                else []
            )
            if not trace_paths:
                trace_path = str(metadata.get("trace_path") or request_dict.get("trace_path") or "").strip()
                if trace_path:
                    trace_paths = [trace_path]
            for trace_path in trace_paths:
                if not Path(trace_path).is_file():
                    return f"{Path(trace_path)} not found"
            raw_run_dirs = request_dict.get("run_dirs")
            if isinstance(raw_run_dirs, list):
                for raw_run_dir in raw_run_dirs:
                    run_dir_text = str(raw_run_dir or "").strip()
                    if not run_dir_text:
                        continue
                    library_path = Path(run_dir_text) / "pattern_library.json"
                    label_glob = list(Path(run_dir_text).glob("label_*"))
                    if library_path.is_file() or label_glob:
                        continue
                    return f"{Path(run_dir_text)} missing supervised run artifacts"
        if action == "automatic-launch":
            source_run_dir = str(metadata.get("source_supervised_run_dir") or "").strip()
            if not source_run_dir:
                request = metadata.get("request")
                if isinstance(request, dict):
                    source_run_dir = str(request.get("source_supervised_run_dir") or "").strip()
            if not source_run_dir:
                return None
            source_root = Path(source_run_dir)
            run_json_path = source_root / "run.json"
            requested_run_id = str(metadata.get("source_evaluation_run_id") or "").strip()
            if not requested_run_id:
                request = metadata.get("request")
                if isinstance(request, dict):
                    requested_run_id = str(request.get("source_evaluation_run_id") or "").strip()
            cache_key = (action, source_run_dir, requested_run_id)
            cached = self._prerequisite_reason_cache.get(cache_key)
            now_mono = time.monotonic()
            if cached is not None and now_mono - cached[0] <= PREREQUISITE_REASON_CACHE_SECONDS:
                return cached[1]
            unusable_reason: str | None = None
            searched_eval_roots: list[Path] = []
            for candidate_root in self._candidate_supervised_source_run_dirs_for_automatic(source_run_dir):
                deepphy_eval_root = candidate_root / "deepphy_eval"
                searched_eval_roots.append(deepphy_eval_root)
                run_json_path = candidate_root / "run.json"
                if requested_run_id and requested_run_id != "__auto_source_eval__":
                    results_path = deepphy_eval_root / requested_run_id / "results.json"
                    eval_run_root = results_path.parent
                    if results_path.is_file() or eval_run_root.is_dir():
                        if self._results_path_has_reasoning_traces(results_path):
                            self._prerequisite_reason_cache[cache_key] = (now_mono, None)
                            return None
                        unusable_reason = (
                            f"No usable reasoning traces found in {results_path}; "
                            "rerun the supervised DeepPHY evaluation"
                        )
                    elif unusable_reason is None:
                        unusable_reason = f"{results_path} not found"
                if run_json_path.is_file():
                    try:
                        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        payload = {}
                    if isinstance(payload, dict):
                        auto_run_id = payload.get("auto_evaluation_run_id")
                        if isinstance(auto_run_id, str) and auto_run_id.strip():
                            candidate = deepphy_eval_root / auto_run_id / "results.json"
                            if candidate.is_file() or candidate.parent.is_dir():
                                if self._results_path_has_reasoning_traces(candidate):
                                    self._prerequisite_reason_cache[cache_key] = (now_mono, None)
                                    return None
                                if unusable_reason is None:
                                    unusable_reason = (
                                        f"No usable reasoning traces found in {candidate}; "
                                        "rerun the supervised DeepPHY evaluation"
                                    )
                if deepphy_eval_root.is_dir():
                    found_eval_runs = False
                    for eval_run_root in sorted(deepphy_eval_root.iterdir()):
                        if not eval_run_root.is_dir():
                            continue
                        found_eval_runs = True
                        candidate = eval_run_root / "results.json"
                        if self._results_path_has_reasoning_traces(candidate):
                            self._prerequisite_reason_cache[cache_key] = (now_mono, None)
                            return None
                        if unusable_reason is None:
                            unusable_reason = (
                                f"No usable reasoning traces found in {candidate}; "
                                "rerun the supervised DeepPHY evaluation"
                            )
                    if found_eval_runs:
                        if unusable_reason is not None:
                            self._prerequisite_reason_cache[cache_key] = (now_mono, unusable_reason)
                            return unusable_reason
                        reason = (
                            f"No usable reasoning traces found under {deepphy_eval_root}; "
                            "rerun the supervised DeepPHY evaluation"
                        )
                        self._prerequisite_reason_cache[cache_key] = (now_mono, reason)
                        return reason
            if unusable_reason is not None:
                self._prerequisite_reason_cache[cache_key] = (now_mono, unusable_reason)
                return unusable_reason
            search_root = searched_eval_roots[0] if searched_eval_roots else source_root / "deepphy_eval"
            reason = f"Supervised evaluation results not found under {search_root}"
            self._prerequisite_reason_cache[cache_key] = (now_mono, reason)
            return reason
        return None

    def _job_supports_workers(self, job: JobRecord) -> bool:
        return str(job.spec.action or "").strip() in {
            "supervised-learn-and-evaluate",
            "unsupervised-launch",
            "automatic-launch",
            "evaluate-deepphy",
            "evaluate-library-case",
        }

    def _normalize_job_workers(self, job: JobRecord, workers: int) -> int:
        if not self._job_supports_workers(job):
            raise ValueError("Job does not support worker configuration.")
        desired = max(1, int(workers))
        env_id = str((job.spec.metadata or {}).get("env_id") or "").strip().lower()
        if env_id == "pooltool":
            desired = 1
        return desired

    def _apply_job_worker_update(self, job: JobRecord, workers: int) -> JobRecord:
        metadata = dict(job.spec.metadata or {})
        request_payload = metadata.get("request")
        if isinstance(request_payload, dict):
            request_payload = {**request_payload, "workers": workers}
            metadata["request"] = request_payload
        metadata["workers"] = workers
        metadata["requested_workers"] = workers
        job.spec.metadata = metadata
        job.spec.argv = self._replace_or_add_flag(job.spec.argv, "--workers", str(workers))
        resume_argv = metadata.get("resume_argv")
        if isinstance(resume_argv, list) and resume_argv:
            metadata["resume_argv"] = self._replace_or_add_flag(
                [str(part) for part in resume_argv],
                "--workers",
                str(workers),
            )
            job.spec.metadata = metadata
        run_dir = self._job_run_dir_path(job)
        if run_dir is not None and run_dir.exists() and self._job_uses_learning_run_control(job):
            applied_workers = self._update_learning_run_worker_target(run_dir, workers)
            if applied_workers != workers:
                metadata["workers"] = applied_workers
                metadata["requested_workers"] = applied_workers
                if isinstance(request_payload, dict):
                    request_payload["workers"] = applied_workers
                    metadata["request"] = request_payload
                job.spec.metadata = metadata
                job.spec.argv = self._replace_or_add_flag(job.spec.argv, "--workers", str(applied_workers))
                resume_argv = metadata.get("resume_argv")
                if isinstance(resume_argv, list) and resume_argv:
                    metadata["resume_argv"] = self._replace_or_add_flag(
                        [str(part) for part in resume_argv],
                        "--workers",
                        str(applied_workers),
                    )
                    job.spec.metadata = metadata
        return job

    def _replace_or_add_flag(self, argv: list[str], flag: str, value: str) -> list[str]:
        rewritten = [str(part) for part in argv]
        index = 0
        while index < len(rewritten):
            if rewritten[index] == flag:
                if index + 1 < len(rewritten):
                    rewritten[index + 1] = value
                else:
                    rewritten.append(value)
                return rewritten
            index += 1
        rewritten.extend([flag, value])
        return rewritten

    def _remove_flag(self, argv: list[str], flag: str) -> list[str]:
        rewritten: list[str] = []
        normalized = [str(part) for part in argv]
        index = 0
        while index < len(normalized):
            if normalized[index] == flag:
                index += 2
                continue
            rewritten.append(normalized[index])
            index += 1
        return rewritten

    def _ensure_flag(self, argv: list[str], flag: str) -> list[str]:
        rewritten = [str(part) for part in argv]
        if flag not in rewritten:
            rewritten.append(flag)
        return rewritten

    def _server_learning_stop_policy_config_path(self) -> Path:
        target = (self.project_root / SERVER_POLICY_CONFIG_RELATIVE_PATH).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "\n".join(
                [
                    'config_version: "1"',
                    "split:",
                    "  - train",
                    "validation_split:",
                    "  - validation",
                "generator_retries: 3",
                "iteration_count: 8",
                "seed: 42",
                f"population_size: {SERVER_POPULATION_SIZE}",
                "elite_fraction: 0.25",
                "children_per_elite: 2",
                "crossover_rate: 1.0",
                    f"min_candidate_evaluations: {SERVER_MIN_CANDIDATE_EVALUATIONS}",
                    (
                        "no_improvement_patience_evaluations: "
                        f"{SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS}"
                    ),
                    f"max_candidate_evaluations: {SERVER_MAX_CANDIDATE_EVALUATIONS}",
                    "accept_margin: 0.001",
                    "max_repairs: 2",
                    "mutate_temperature: 0.7",
                    "mutate_max_tokens: 8192",
                    "losses:",
                    "  - kind: correlation_structure",
                    "    weight: 1.0",
                    "    histogram_bins: 12",
                    "    pair_limit: 500",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return target

    def _job_supports_learning_stop_policy(self, job: JobRecord) -> bool:
        return str(job.spec.action or "").strip() in LEARNING_STOP_POLICY_ACTIONS

    def _rewrite_learning_stop_policy_argv(
        self,
        argv: list[str],
        *,
        action: str | None = None,
    ) -> list[str]:
        rewritten = self._remove_flag(argv, "--patience-factor")
        rewritten = self._remove_flag(rewritten, "--population-size")
        rewritten = self._remove_flag(rewritten, "--min-candidate-evaluations")
        rewritten = self._remove_flag(rewritten, "--no-improvement-patience-evaluations")
        rewritten = self._remove_flag(rewritten, "--max-candidate-evaluations")
        if str(action or "").strip() == "automatic-launch":
            return self._remove_flag(rewritten, "--config")
        rewritten = self._replace_or_add_flag(
            rewritten,
            "--config",
            str(self._server_learning_stop_policy_config_path()),
        )
        return rewritten

    def _normalize_learning_stop_policy_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized.pop("patience_factor", None)
        normalized["population_size"] = SERVER_POPULATION_SIZE
        normalized["min_candidate_evaluations"] = SERVER_MIN_CANDIDATE_EVALUATIONS
        normalized["no_improvement_patience_evaluations"] = SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS
        normalized["max_candidate_evaluations"] = SERVER_MAX_CANDIDATE_EVALUATIONS
        normalized["config_path"] = str(self._server_learning_stop_policy_config_path())
        return normalized

    def _resolved_job_run_dir_path(self, job: JobRecord) -> Path | None:
        run_dir = self._job_run_dir_path(job)
        if run_dir is None:
            return None
        resolved = _resolve_source_for_bundle(run_dir, project_root=self.project_root)
        resume_sync_root = resolved.parent / f"{resolved.name}.__resume_sync__{job.job_id}"
        if resume_sync_root.exists():
            if self._job_uses_learning_run_control(job) and resolved.exists():
                canonical_completion = self._completed_learning_run_payload(job, resolved)
                resume_incomplete_reason = self._incomplete_learning_run_reason(job, resume_sync_root)
                if canonical_completion is not None and resume_incomplete_reason is not None:
                    return resolved
            return resume_sync_root
        if resolved.exists():
            return resolved
        return None

    def _synchronize_learning_run_progress_metadata(self, state: dict[str, Any]) -> bool:
        changed = False
        for raw_job in state.get("jobs", {}).values():
            job = JobRecord.model_validate(raw_job)
            if not self._job_uses_learning_run_control(job):
                continue
            effective_run_root = self._resolved_job_run_dir_path(job)
            if effective_run_root is None or not effective_run_root.is_dir():
                continue
            effective_payload = self._load_json_payload(effective_run_root / "run.json")
            effective_completed, effective_total, _mtime = self._learning_run_completion_counts(
                effective_run_root,
                effective_payload,
            )
            if self._write_learning_run_progress_metadata(
                effective_run_root,
                completed_labels=effective_completed,
                total_labels=effective_total,
            ):
                changed = True
            canonical_run_root = self._job_run_dir_path(job)
            if canonical_run_root is None:
                continue
            canonical_run_root = _resolve_source_for_bundle(
                canonical_run_root,
                project_root=self.project_root,
            )
            if canonical_run_root == effective_run_root or not canonical_run_root.is_dir():
                continue
            canonical_payload = self._load_json_payload(canonical_run_root / "run.json")
            canonical_completed, canonical_total, _mtime = self._learning_run_completion_counts(
                canonical_run_root,
                canonical_payload,
            )
            if self._write_learning_run_progress_metadata(
                canonical_run_root,
                completed_labels=max(canonical_completed, effective_completed),
                total_labels=max(canonical_total, effective_total),
            ):
                changed = True
        return changed

    def _load_json_payload(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_learning_run_progress_metadata(
        self,
        run_root: Path,
        *,
        completed_labels: int,
        total_labels: int,
    ) -> bool:
        run_json_path = run_root / "run.json"
        payload = self._load_json_payload(run_json_path)
        if not payload:
            return False
        changed = False
        try:
            existing_completed = int(payload.get("completed_label_count") or 0)
        except (TypeError, ValueError):
            existing_completed = 0
        if completed_labels != existing_completed:
            payload["completed_label_count"] = completed_labels
            changed = True
        try:
            existing_total = int(payload.get("primitive_label_count") or 0)
        except (TypeError, ValueError):
            existing_total = 0
        if total_labels > existing_total:
            payload["primitive_label_count"] = total_labels
            changed = True
        if changed:
            run_json_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return changed

    def _job_should_force_resume_existing_run(self, job: JobRecord) -> bool:
        if str(job.spec.action or "").strip() not in RESUMABLE_EXISTING_RUN_ACTIONS:
            return False
        run_dir = self._resolved_job_run_dir_path(job)
        if run_dir is None or not run_dir.is_dir():
            return False
        if (run_dir / "proposed_labels.txt").is_file() and (run_dir / "proposed_labels.json").is_file():
            return True
        if (run_dir / "pattern_library.json").is_file():
            return True
        if any(run_dir.glob("label_*")):
            return True
        return False

    def _enforce_existing_run_resume_policy(self, state: dict[str, Any]) -> bool:
        changed = False
        for job_id, raw_job in list(state.get("jobs", {}).items()):
            job_changed = False
            try:
                job = JobRecord.model_validate(raw_job)
            except Exception:
                continue
            if not self._job_should_force_resume_existing_run(job):
                continue
            metadata = dict(job.spec.metadata or {})
            request_payload = metadata.get("request")
            metadata_changed = False
            if isinstance(request_payload, dict):
                normalized_request = dict(request_payload)
                if normalized_request.get("resume") is not True:
                    normalized_request["resume"] = True
                    metadata["request"] = normalized_request
                    metadata_changed = True
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                rewritten_resume_argv = self._ensure_flag([str(part) for part in resume_argv], "--resume")
            else:
                rewritten_resume_argv = self._ensure_flag([str(part) for part in job.spec.argv], "--resume")
            if rewritten_resume_argv != [str(part) for part in (resume_argv or [])]:
                metadata["resume_argv"] = rewritten_resume_argv
                metadata_changed = True
            if job.status in {JobStatus.queued, JobStatus.paused, JobStatus.assigned}:
                rewritten_argv = self._ensure_flag([str(part) for part in job.spec.argv], "--resume")
                if rewritten_argv != [str(part) for part in job.spec.argv]:
                    job.spec.argv = rewritten_argv
                    job_changed = True
            if metadata_changed:
                job.spec.metadata = metadata
                job_changed = True
            if job_changed:
                state["jobs"][job_id] = job.model_dump(mode="json")
                changed = True
        return changed

    def _update_learning_run_stop_policy(self, run_root: Path) -> bool:
        run_json_path = run_root / "run.json"
        if not run_json_path.is_file():
            return False
        try:
            run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(run_payload, dict):
            return False
        changed = False
        if run_payload.get("population_size") != SERVER_POPULATION_SIZE:
            run_payload["population_size"] = SERVER_POPULATION_SIZE
            changed = True
        if run_payload.get("min_candidate_evaluations") != SERVER_MIN_CANDIDATE_EVALUATIONS:
            run_payload["min_candidate_evaluations"] = SERVER_MIN_CANDIDATE_EVALUATIONS
            changed = True
        if (
            run_payload.get("no_improvement_patience_evaluations")
            != SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS
        ):
            run_payload["no_improvement_patience_evaluations"] = (
                SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS
            )
            changed = True
        if run_payload.get("patience_evaluations") != SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS:
            run_payload["patience_evaluations"] = SERVER_NO_IMPROVEMENT_PATIENCE_EVALUATIONS
            changed = True
        if run_payload.get("max_candidate_evaluations") != SERVER_MAX_CANDIDATE_EVALUATIONS:
            run_payload["max_candidate_evaluations"] = SERVER_MAX_CANDIDATE_EVALUATIONS
            changed = True
        if "patience_factor" in run_payload:
            run_payload.pop("patience_factor", None)
            changed = True
        if changed:
            run_json_path.write_text(
                json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return changed

    def _enforce_learning_stop_policy(self, state: dict[str, Any]) -> bool:
        changed = False
        for raw_job in state.get("jobs", {}).values():
            try:
                job = JobRecord.model_validate(raw_job)
            except Exception:
                continue
            if job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
                JobStatus.stopped,
            }:
                continue
            if not self._job_supports_learning_stop_policy(job):
                continue
            original_metadata = dict(job.spec.metadata or {})
            metadata = dict(original_metadata)
            request_payload = metadata.get("request")
            if isinstance(request_payload, dict):
                metadata["request"] = self._normalize_learning_stop_policy_request(request_payload)
            action = str(job.spec.action or "").strip()
            rewritten_argv = self._rewrite_learning_stop_policy_argv(
                job.spec.argv,
                action=action,
            )
            if rewritten_argv != job.spec.argv:
                job.spec.argv = rewritten_argv
                changed = True
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                rewritten_resume_argv = self._rewrite_learning_stop_policy_argv(
                    [str(part) for part in resume_argv],
                    action=action,
                )
                if rewritten_resume_argv != [str(part) for part in resume_argv]:
                    metadata["resume_argv"] = rewritten_resume_argv
                    changed = True
            run_dir = self._resolved_job_run_dir_path(job)
            if run_dir is not None and run_dir.exists():
                if self._update_learning_run_stop_policy(run_dir):
                    changed = True
            if metadata != original_metadata:
                job.spec.metadata = metadata
                changed = True
            state["jobs"][job.job_id] = job.model_dump(mode="json")
        return changed

    def _update_learning_run_evaluation_strategy(self, run_root: Path) -> bool:
        run_json_path = run_root / "run.json"
        if not run_json_path.is_file():
            return False
        try:
            run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(run_payload, dict):
            return False
        changed = False
        if "defer_evaluation" in run_payload:
            run_payload.pop("defer_evaluation", None)
            changed = True
        if str(run_payload.get("evaluation_strategy") or "").strip().lower() == "scheduled":
            run_payload.pop("evaluation_strategy", None)
            changed = True
        if changed:
            run_json_path.write_text(
                json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return changed

    def _clear_stale_learning_run_auto_eval(self, run_root: Path) -> bool:
        run_json_path = run_root / "run.json"
        if not run_json_path.is_file():
            return False
        try:
            run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(run_payload, dict):
            return False
        changed = False
        auto_run_id = str(run_payload.get("auto_evaluation_run_id") or "").strip()
        if auto_run_id:
            history_raw = run_payload.get("historical_auto_evaluation_run_ids")
            history = (
                [str(item).strip() for item in history_raw if str(item).strip()]
                if isinstance(history_raw, list)
                else []
            )
            if auto_run_id not in history:
                history.append(auto_run_id)
                run_payload["historical_auto_evaluation_run_ids"] = history
                changed = True
        if "auto_evaluation_run_id" in run_payload:
            run_payload.pop("auto_evaluation_run_id", None)
            changed = True
        if changed:
            run_json_path.write_text(
                json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return changed

    def _enforce_learning_auto_evaluation(self, state: dict[str, Any]) -> bool:
        changed = False
        for raw_job in state.get("jobs", {}).values():
            try:
                job = JobRecord.model_validate(raw_job)
            except Exception:
                continue
            if str(job.spec.action or "").strip() not in {
                "supervised-learn-and-evaluate",
                "unsupervised-launch",
            }:
                continue
            if job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
                JobStatus.stopped,
            }:
                continue
            original_metadata = dict(job.spec.metadata or {})
            metadata = dict(original_metadata)
            metadata.pop("defer_evaluation", None)
            if str(metadata.get("evaluation_strategy") or "").strip().lower() == "scheduled":
                metadata.pop("evaluation_strategy", None)
            request_payload = metadata.get("request")
            if isinstance(request_payload, dict):
                normalized_request = dict(request_payload)
                normalized_request.pop("defer_evaluation", None)
                if str(normalized_request.get("evaluation_strategy") or "").strip().lower() == "scheduled":
                    normalized_request.pop("evaluation_strategy", None)
                metadata["request"] = normalized_request
            rewritten_argv = self._remove_flag([str(part) for part in job.spec.argv], "--defer-evaluation")
            if rewritten_argv != [str(part) for part in job.spec.argv]:
                job.spec.argv = rewritten_argv
                changed = True
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                rewritten_resume_argv = self._remove_flag(
                    [str(part) for part in resume_argv],
                    "--defer-evaluation",
                )
                if rewritten_resume_argv != [str(part) for part in resume_argv]:
                    metadata["resume_argv"] = rewritten_resume_argv
                    changed = True
            run_dir = self._resolved_job_run_dir_path(job)
            if run_dir is not None and run_dir.exists():
                if self._update_learning_run_evaluation_strategy(run_dir):
                    changed = True
            if metadata != original_metadata:
                job.spec.metadata = metadata
                changed = True
            state["jobs"][job.job_id] = job.model_dump(mode="json")
        return changed

    def _job_run_dir_path(self, job: JobRecord) -> Path | None:
        metadata = dict(job.spec.metadata or {})
        for key in ("run_dir", "evaluation_run_dir", "library_evaluation_run_dir", "output_path"):
            raw = str(metadata.get(key) or "").strip()
            if raw:
                return Path(raw)
        return None

    def _job_uses_learning_run_control(self, job: JobRecord) -> bool:
        return str(job.spec.action or "").strip() in {
            "supervised-learn-and-evaluate",
            "unsupervised-launch",
            "automatic-launch",
        }

    def _read_learning_run_control(self, run_root: Path) -> dict[str, Any]:
        control_path = run_root / "run_control.json"
        if not control_path.is_file():
            return {}
        try:
            payload = json.loads(control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_learning_run_control(self, run_root: Path, payload: dict[str, Any]) -> None:
        control_path = run_root / "run_control.json"
        control_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _update_learning_run_worker_target(self, run_root: Path, workers: int) -> int:
        desired_workers = max(1, int(workers))
        primitive_label_count = None
        run_json_path = run_root / "run.json"
        if run_json_path.is_file():
            try:
                run_payload = json.loads(run_json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                run_payload = {}
            if isinstance(run_payload, dict):
                if str(run_payload.get("env_id") or "") == "pooltool":
                    desired_workers = 1
                primitive_label_count = run_payload.get("primitive_label_count")
                try:
                    max_workers = int(primitive_label_count)
                except (TypeError, ValueError):
                    max_workers = None
                if max_workers is not None and max_workers > 0:
                    desired_workers = min(desired_workers, max_workers)
                run_payload["workers"] = desired_workers
                run_json_path.write_text(
                    json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        control_payload = self._read_learning_run_control(run_root)
        configured_workers = desired_workers
        if isinstance(control_payload, dict):
            try:
                configured_workers = max(
                    1, int(control_payload.get("configured_workers") or desired_workers)
                )
            except (TypeError, ValueError):
                configured_workers = desired_workers
        control_payload.update(
            {
                "desired_workers": desired_workers,
                "configured_workers": configured_workers,
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        self._write_learning_run_control(run_root, control_payload)
        return desired_workers

    def _should_run_periodic_reconcile(self, *, force_reconcile: bool = False) -> bool:
        if force_reconcile:
            self._last_periodic_reconcile_monotonic = time.monotonic()
            return True
        now = time.monotonic()
        if (now - self._last_periodic_reconcile_monotonic) < PERIODIC_RECONCILE_SECONDS:
            return False
        self._last_periodic_reconcile_monotonic = now
        return True

    def _current_state_file_mtime_ns(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def _read(
        self,
        *,
        force_reconcile: bool = False,
        allow_periodic_reconcile: bool = False,
    ) -> dict[str, Any]:
        current_mtime_ns = self._current_state_file_mtime_ns()
        if (
            self._state_payload_cache is not None
            and current_mtime_ns is not None
            and self._state_payload_mtime_ns == current_mtime_ns
        ):
            state = copy.deepcopy(self._state_payload_cache)
        else:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            self._state_payload_cache = copy.deepcopy(state)
            self._state_payload_mtime_ns = current_mtime_ns
        state.setdefault("client_commands", {})
        changed = False
        if self._trim_oversized_job_state(state):
            changed = True
        if self._normalize_queue_settings(state):
            changed = True
        if self._prune_stale_idle_clients(state):
            changed = True
        if self._enforce_existing_run_resume_policy(state):
            changed = True
        if self._enforce_learning_stop_policy(state):
            changed = True
        if self._enforce_learning_auto_evaluation(state):
            changed = True
        if self._synchronize_learning_run_progress_metadata(state):
            changed = True
        if self._enforce_image_only_eval_caps(state):
            changed = True
        if self._reconcile_stale_image_only_eval_cancellations(state):
            changed = True
        if (
            (force_reconcile or allow_periodic_reconcile)
            and self._should_run_periodic_reconcile(force_reconcile=force_reconcile)
        ):
            if self._normalize_repeat_learning_queue(state):
                changed = True
            if self._recover_stale_clients(state):
                changed = True
            if self._recover_stale_active_jobs(state):
                changed = True
            if self._reconcile_dependency_blocked_jobs(state):
                changed = True
            if self._reconcile_missing_preferred_clients(state):
                changed = True
            if self._reconcile_incomplete_automatic_eval_jobs(state):
                changed = True
            if self._reconcile_incomplete_library_ablation_jobs(state):
                changed = True
            if self._reconcile_invalid_active_learning_auto_eval_jobs(state):
                changed = True
            if self._reconcile_invalid_completed_learning_jobs(state):
                changed = True
            if self._reconcile_completed_learning_jobs(state):
                changed = True
            if self._clear_pooltool_attempt_limit_job_failures(state):
                changed = True
            if self._reconcile_completed_deepphy_jobs(state):
                changed = True
            if self._reconcile_completed_library_ablation_jobs(state):
                changed = True
            if self._recover_orphan_active_jobs(state):
                changed = True
            if self._reconcile_client_current_job_views(state):
                changed = True
        if changed:
            self._write(state)
        return state

    def run_maintenance_cycle(self) -> bool:
        with self._lock:
            before_mtime = self._current_state_file_mtime_ns()
            self._read(force_reconcile=False, allow_periodic_reconcile=True)
            after_mtime = self._current_state_file_mtime_ns()
            return after_mtime != before_mtime

    def _image_only_repeat_index(self, job: JobRecord) -> int | None:
        if str(job.spec.action or "").strip() != "evaluate-deepphy":
            return None
        metadata = dict(job.spec.metadata or {})
        request = dict(metadata.get("request") or {})
        markers = [
            str(metadata.get("evaluation_run_id") or "").strip(),
            str(request.get("run_id") or "").strip(),
            str(metadata.get("evaluation_run_dir") or "").strip(),
            str(request.get("output_dir") or "").strip(),
            str(metadata.get("run_dir") or "").strip(),
        ]
        for marker in markers:
            if not marker:
                continue
            match = REPEAT_EVAL_SUFFIX_RE.match(Path(marker).name)
            if match is None:
                continue
            if "image-only-deepphy" not in Path(marker).name:
                continue
            try:
                return int(match.group("index"))
            except (TypeError, ValueError):
                return None
        return None

    def _job_exceeds_image_only_eval_target(self, job: JobRecord) -> bool:
        repeat_index = self._image_only_repeat_index(job)
        return repeat_index is not None and repeat_index > IMAGE_ONLY_EVAL_TARGET_RUNS

    def _is_phyre_required_image_only_eval(self, job: JobRecord) -> bool:
        if str(job.spec.action or "").strip() != "evaluate-deepphy":
            return False
        metadata = dict(job.spec.metadata or {})
        if str(metadata.get("env_id") or "").strip().lower() != "phyre":
            return False
        request = dict(metadata.get("request") or {})
        markers = [
            str(metadata.get("evaluation_run_id") or "").strip(),
            str(request.get("run_id") or "").strip(),
            str(metadata.get("evaluation_run_dir") or "").strip(),
            str(request.get("output_dir") or "").strip(),
            str(metadata.get("run_dir") or "").strip(),
        ]
        return any("required-20260420-phyre-image-only-deepphy" in Path(marker).name for marker in markers if marker)

    def _completed_threshold_deepphy_run_summary(
        self,
        job: JobRecord,
        run_root: Path,
    ) -> tuple[dict[str, Any], float] | None:
        if not self._is_phyre_required_image_only_eval(job):
            return None
        tasks_root = run_root / "tasks"
        if not tasks_root.is_dir():
            return None
        completed_task_paths = sorted(tasks_root.glob("*/*/task_result.json"))
        if len(completed_task_paths) < PHYRE_IMAGE_ONLY_COMPLETION_TASK_TARGET:
            return None
        latest_mtime = max(task_path.stat().st_mtime for task_path in completed_task_paths)
        return (
            {
                "task_count": PHYRE_IMAGE_ONLY_COMPLETION_TASK_TARGET,
                "completed_task_count": len(completed_task_paths),
                "completion_mode": "task-threshold",
            },
            latest_mtime,
        )

    def _enforce_image_only_eval_caps(self, state: dict[str, Any]) -> bool:
        changed = False
        now = _utc_now()
        for job_id, raw_job in list(state.get("jobs", {}).items()):
            job = JobRecord.model_validate(raw_job)
            if not self._job_exceeds_image_only_eval_target(job):
                continue
            if job.status in {JobStatus.succeeded, JobStatus.cancelled, JobStatus.failed, JobStatus.stopped}:
                continue
            if job.status != JobStatus.paused or str(job.assigned_client_id or "").strip():
                job.status = JobStatus.paused
                job.assigned_client_id = None
                job.updated_at = now
                job.finished_at = now
                job.pending_commands = []
                job.message = (
                    f"Permanently paused because required image-only target is capped at "
                    f"{IMAGE_ONLY_EVAL_TARGET_RUNS} runs per env; over-cap runs are paused "
                    f"starting with least progress."
                )
                state["jobs"][job_id] = job.model_dump(mode="json")
                changed = True
            self._detach_job_from_client_record(state, str(raw_job.get("assigned_client_id") or ""), job_id)
        return changed

    def _reconcile_stale_image_only_eval_cancellations(self, state: dict[str, Any]) -> bool:
        changed = False
        now = _utc_now()
        for job_id, raw_job in list(state.get("jobs", {}).items()):
            job = JobRecord.model_validate(raw_job)
            if str(job.spec.action or "").strip() != "evaluate-deepphy":
                continue
            repeat_index = self._image_only_repeat_index(job)
            if repeat_index is None or repeat_index > IMAGE_ONLY_EVAL_TARGET_RUNS:
                continue
            if job.status != JobStatus.cancelled:
                continue
            message_text = str(job.message or "").lower()
            if (
                "excess queued repeat evaluation" not in message_text
                and "target lowered to 3 per run" not in message_text
            ):
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            job.status = JobStatus.queued
            job.updated_at = now
            job.finished_at = None
            job.started_at = None
            job.assigned_client_id = None
            job.exit_code = None
            job.pending_commands = []
            job.message = "Re-queued after stale image-only repeat cancellation was cleared"
            progress = metadata.get("progress")
            if isinstance(progress, dict):
                progress = dict(progress)
                progress["phase"] = "Queued"
                progress["summary"] = "Re-queued after stale image-only repeat cancellation was cleared"
                progress["fraction"] = 0.0
                metadata["progress"] = progress
                job.spec.metadata = metadata
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _recover_stale_active_jobs(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            client_id = str(job.assigned_client_id or "").strip()
            if not client_id:
                continue
            stale_reference = job.updated_at or job.started_at or job.created_at
            if (now - stale_reference) < STALE_ACTIVE_JOB_TIMEOUT:
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            self._mark_job_avoid_client(job, client_id=client_id, now=now)
            job.status = JobStatus.queued
            job.assigned_client_id = None
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.exit_code = None
            job.pending_commands = []
            job.message = (
                f"Recovered after stale active lease on client {client_id} "
                f"exceeded {int(STALE_ACTIVE_JOB_TIMEOUT.total_seconds() // 60)} minutes"
            )
            state["jobs"][job_id] = job.model_dump(mode="json")
            raw_client = state.get("clients", {}).get(client_id)
            if raw_client is not None:
                client = ClientRecord.model_validate(raw_client)
                if job_id in client.current_jobs:
                    client.current_jobs = [
                        current_job_id
                        for current_job_id in client.current_jobs
                        if current_job_id != job_id
                    ]
                    state["clients"][client_id] = client.model_dump(mode="json")
            changed = True
        return changed

    def _load_snapshot_from_disk(self) -> None:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self._state_payload_cache = copy.deepcopy(state)
        self._state_payload_mtime_ns = self._current_state_file_mtime_ns()
        self._refresh_snapshot(state)

    def _refresh_snapshot_from_disk_if_needed(self) -> None:
        current_mtime_ns = self._current_state_file_mtime_ns()
        if current_mtime_ns is None:
            return
        with self._snapshot_lock:
            cached_mtime_ns = self._state_payload_mtime_ns
        if cached_mtime_ns == current_mtime_ns:
            return
        with self._lock:
            current_mtime_ns = self._current_state_file_mtime_ns()
            if current_mtime_ns is None or self._state_payload_mtime_ns == current_mtime_ns:
                return
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return
            self._state_payload_cache = copy.deepcopy(state)
            self._state_payload_mtime_ns = current_mtime_ns
            self._refresh_snapshot(state)

    def _refresh_snapshot(self, payload: dict[str, Any]) -> None:
        clients = [
            ClientRecord.model_validate(item)
            for item in payload.get("clients", {}).values()
        ]
        clients_sorted = tuple(
            sorted(
                clients,
                key=lambda item: (
                    0 if item.status == ClientStatus.online else 1,
                    item.display_name.lower(),
                    item.client_id,
                ),
            )
        )
        jobs = [
            JobRecord.model_validate(item)
            for item in payload.get("jobs", {}).values()
        ]
        jobs_sorted = tuple(sorted(jobs, key=lambda item: item.created_at, reverse=True))
        with self._snapshot_lock:
            self._clients_snapshot = clients_sorted
            self._jobs_snapshot = jobs_sorted

    def _reconcile_missing_preferred_clients(self, state: dict[str, Any]) -> bool:
        live_client_ids = set(state.get("clients", {}).keys())
        active_jobs = self._active_jobs_from_state(state)
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            metadata = dict(job.spec.metadata or {})
            preferred_client_id = str(metadata.get("preferred_client_id") or "").strip()
            if not preferred_client_id:
                continue
            if preferred_client_id in live_client_ids:
                if job.status == JobStatus.queued:
                    preferred_raw = state["clients"].get(preferred_client_id)
                    preferred_client = (
                        ClientRecord.model_validate(preferred_raw)
                        if preferred_raw is not None
                        else None
                    )
                    unpinned_job = job.model_copy(deep=True)
                    unpinned_metadata = dict(unpinned_job.spec.metadata or {})
                    unpinned_metadata.pop("preferred_client_id", None)
                    unpinned_metadata.pop("preferred_client_name", None)
                    unpinned_job.spec.metadata = unpinned_metadata
                    preferred_can_run = (
                        preferred_client is not None
                        and self._client_can_run(
                            preferred_client,
                            job,
                            active_jobs=active_jobs,
                        )
                    )
                    alternate_can_run = any(
                        other_client_id != preferred_client_id
                        and self._client_can_run(
                            ClientRecord.model_validate(other_raw_client),
                            unpinned_job,
                            active_jobs=active_jobs,
                        )
                        for other_client_id, other_raw_client in state["clients"].items()
                    )
                    if not preferred_can_run and alternate_can_run:
                        metadata.pop("preferred_client_id", None)
                        metadata.pop("preferred_client_name", None)
                        job.spec.metadata = metadata
                        job.updated_at = _utc_now()
                        state["jobs"][job_id] = job.model_dump(mode="json")
                        changed = True
                continue
            metadata.pop("preferred_client_id", None)
            metadata.pop("preferred_client_name", None)
            job.spec.metadata = metadata
            job.updated_at = _utc_now()
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _reconcile_completed_learning_jobs(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
                JobStatus.stopped,
            }:
                continue
            if not self._job_uses_learning_run_control(job):
                continue
            run_root = self._resolved_job_run_dir_path(job)
            if run_root is None:
                continue
            completion = self._completed_learning_run_payload(job, run_root)
            if completion is None:
                continue
            completion_payload, completion_mtime = completion
            if job.status not in ACTIVE_JOB_STATUSES:
                updated_at_ts = job.updated_at.timestamp()
                metadata = dict(job.spec.metadata or {})
                progress = dict(metadata.get("progress") or {})
                phase = str(progress.get("phase") or "").strip().lower()
                uploaded_artifacts = metadata.get("uploaded_artifacts")
                has_uploaded_artifacts = isinstance(uploaded_artifacts, list) and bool(uploaded_artifacts)
                if not has_uploaded_artifacts:
                    try:
                        has_uploaded_artifacts = int(metadata.get("uploaded_artifact_count") or 0) > 0
                    except (TypeError, ValueError):
                        has_uploaded_artifacts = False
                if completion_mtime <= updated_at_ts and phase != "completed" and not has_uploaded_artifacts:
                    continue
            job.status = JobStatus.succeeded
            job.updated_at = now
            job.finished_at = now
            job.assigned_client_id = None
            job.exit_code = 0
            job.pending_commands = []
            job.message = "Reconciled from completed run artifacts"
            metadata = dict(job.spec.metadata or {})
            metadata["progress"] = {
                "fraction": 1.0,
                "phase": "Completed",
                "summary": "Done",
            }
            auto_run_id = str(completion_payload.get("auto_evaluation_run_id") or "").strip()
            if auto_run_id:
                metadata["auto_evaluation_run_id"] = auto_run_id
            job.spec.metadata = metadata
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _reconcile_invalid_completed_learning_jobs(self, state: dict[str, Any]) -> bool:
        changed = False
        now = _utc_now()
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status != JobStatus.succeeded:
                continue
            if not self._job_uses_learning_run_control(job):
                continue
            run_root = self._resolved_job_run_dir_path(job)
            if run_root is None:
                continue
            incomplete_reason = self._incomplete_learning_run_reason(job, run_root)
            if incomplete_reason is None:
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            if "auto_evaluation_run_id" in metadata:
                metadata.pop("auto_evaluation_run_id", None)
            metadata.pop("progress", None)
            job.spec.metadata = metadata
            if self._clear_stale_learning_run_auto_eval(run_root):
                changed = True
            job.status = JobStatus.queued
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.assigned_client_id = None
            job.exit_code = None
            job.pending_commands = []
            job.message = f"False completion corrected; {incomplete_reason}"
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _reconcile_invalid_active_learning_auto_eval_jobs(self, state: dict[str, Any]) -> bool:
        changed = False
        now = _utc_now()
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
                JobStatus.stopped,
            }:
                continue
            if not self._job_uses_learning_run_control(job):
                continue
            run_root = self._resolved_job_run_dir_path(job)
            if run_root is None:
                continue
            run_json_path = run_root / "run.json"
            if not run_json_path.is_file():
                continue
            try:
                payload = json.loads(run_json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            auto_run_id = str(payload.get("auto_evaluation_run_id") or "").strip()
            if not auto_run_id:
                continue
            completed_labels, total_labels, _latest_checkpoint_mtime = self._learning_run_completion_counts(
                run_root,
                payload,
            )
            if total_labels <= 0 or completed_labels >= total_labels:
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            metadata.pop("auto_evaluation_run_id", None)
            metadata.pop("progress", None)
            job.spec.metadata = metadata
            self._mark_job_avoid_client(
                job,
                client_id=str(job.assigned_client_id or "").strip() or None,
                now=now,
            )
            if self._clear_stale_learning_run_auto_eval(run_root):
                changed = True
            job.status = JobStatus.queued
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.assigned_client_id = None
            job.exit_code = None
            job.pending_commands = []
            job.message = (
                "Recovered after invalid embedded eval on incomplete learning labels "
                f"({completed_labels}/{total_labels})"
            )
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _reconcile_incomplete_automatic_eval_jobs(self, state: dict[str, Any]) -> bool:
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            job_changed = False
            if str(job.spec.action or "").strip() != "automatic-launch":
                continue
            if job.status == JobStatus.succeeded:
                continue
            run_root = self._resolved_job_run_dir_path(job)
            if run_root is None:
                continue
            run_json_path = run_root / "run.json"
            library_path = run_root / "pattern_library.json"
            if not run_json_path.is_file() or not library_path.is_file():
                continue
            try:
                payload = json.loads(run_json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("status") or "").strip().lower() != "completed":
                continue
            try:
                completed_labels = int(payload.get("completed_label_count") or 0)
                total_labels = int(payload.get("primitive_label_count") or 0)
            except (TypeError, ValueError):
                continue
            if total_labels <= 0 or completed_labels < total_labels:
                continue
            metadata = dict(job.spec.metadata or {})
            progress = {
                "fraction": 1.0,
                "phase": "Automatic labels complete",
                "summary": "Embedded DeepPHY evaluation pending",
            }
            if metadata.get("progress") != progress:
                metadata["progress"] = progress
                job.spec.metadata = metadata
                changed = True
                job_changed = True
            message = str(job.message or "").strip()
            if (
                not message
                or "re-run requested" in message.lower()
                or "embedded evaluation artifacts cleared" in message.lower()
            ):
                desired_message = "Automatic labels complete; embedded DeepPHY evaluation pending"
                if message != desired_message:
                    job.message = desired_message
                    changed = True
                    job_changed = True
            if job_changed:
                state["jobs"][job_id] = job.model_dump(mode="json")
        return changed

    def _completed_learning_run_payload(self, job: JobRecord, run_root: Path) -> tuple[dict[str, Any], float] | None:
        run_json_path = run_root / "run.json"
        library_path = run_root / "pattern_library.json"
        if not run_json_path.is_file() or not library_path.is_file():
            return None
        try:
            payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        status = str(payload.get("status") or "").strip().lower()
        completed_labels, total_labels, latest_checkpoint_mtime = self._learning_run_completion_counts(
            run_root, payload
        )
        if status != "completed" and (total_labels <= 0 or completed_labels < total_labels):
            return None
        if total_labels <= 0 or completed_labels < total_labels:
            return None
        action = str(job.spec.action or "").strip()
        if action == "automatic-launch":
            combined_eval_root = run_root / "deepphy_eval" / "combined-eval"
            combined_eval_completion = self._completed_deepphy_run_summary(combined_eval_root)
            if combined_eval_completion is None:
                return None
            _combined_eval_payload, combined_eval_mtime = combined_eval_completion
            completion_mtime = max(
                run_json_path.stat().st_mtime,
                library_path.stat().st_mtime,
                latest_checkpoint_mtime,
                combined_eval_mtime,
            )
            return payload, completion_mtime
        metadata = dict(job.spec.metadata or {})
        request_payload = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
        defer_evaluation = bool(
            metadata.get("defer_evaluation")
            or payload.get("defer_evaluation")
            or request_payload.get("defer_evaluation")
            or str(metadata.get("evaluation_strategy") or "").strip().lower() == "scheduled"
            or str(payload.get("evaluation_strategy") or "").strip().lower() == "scheduled"
            or str(request_payload.get("evaluation_strategy") or "").strip().lower() == "scheduled"
        )
        if not defer_evaluation and not str(payload.get("auto_evaluation_run_id") or "").strip():
            inferred_auto_run_id = self._infer_completed_embedded_deepphy_eval_run_id(run_root)
            if inferred_auto_run_id:
                payload["auto_evaluation_run_id"] = inferred_auto_run_id
            else:
                return None
        completion_mtime = max(
            run_json_path.stat().st_mtime,
            library_path.stat().st_mtime,
            latest_checkpoint_mtime,
        )
        if not defer_evaluation:
            auto_run_id = str(payload.get("auto_evaluation_run_id") or "").strip()
            auto_eval_root = run_root / "deepphy_eval" / auto_run_id
            auto_eval_completion = self._completed_deepphy_run_summary(auto_eval_root)
            if auto_eval_completion is None:
                return None
            _auto_eval_payload, auto_eval_mtime = auto_eval_completion
            completion_mtime = max(completion_mtime, auto_eval_mtime)
        return payload, completion_mtime

    def _incomplete_learning_run_reason(self, job: JobRecord, run_root: Path) -> str | None:
        run_json_path = run_root / "run.json"
        if not run_json_path.is_file():
            return None
        try:
            payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        status = str(payload.get("status") or "").strip().lower()
        completed_labels, total_labels, _latest_checkpoint_mtime = self._learning_run_completion_counts(
            run_root, payload
        )
        if total_labels > 0 and completed_labels < total_labels:
            return f"learning labels incomplete ({completed_labels}/{total_labels})"
        if status and status != "completed" and (total_labels <= 0 or completed_labels < total_labels):
            return f"learning run status is {status}"
        if str(job.spec.action or "").strip() == "automatic-launch":
            combined_eval_root = run_root / "deepphy_eval" / "combined-eval"
            if self._completed_deepphy_run_summary(combined_eval_root) is None:
                return "embedded DeepPHY evaluation is incomplete"
            return None
        metadata = dict(job.spec.metadata or {})
        request_payload = metadata.get("request") if isinstance(metadata.get("request"), dict) else {}
        defer_evaluation = bool(
            metadata.get("defer_evaluation")
            or payload.get("defer_evaluation")
            or request_payload.get("defer_evaluation")
            or str(metadata.get("evaluation_strategy") or "").strip().lower() == "scheduled"
            or str(payload.get("evaluation_strategy") or "").strip().lower() == "scheduled"
            or str(request_payload.get("evaluation_strategy") or "").strip().lower() == "scheduled"
        )
        if defer_evaluation:
            return None
        auto_run_id = str(payload.get("auto_evaluation_run_id") or "").strip()
        if not auto_run_id:
            auto_run_id = self._infer_completed_embedded_deepphy_eval_run_id(run_root)
        if not auto_run_id:
            return "embedded DeepPHY evaluation is not recorded"
        auto_eval_root = run_root / "deepphy_eval" / auto_run_id
        if self._completed_deepphy_run_summary(auto_eval_root) is None:
            return f"embedded DeepPHY evaluation is incomplete ({auto_run_id})"
        return None

    def _infer_completed_embedded_deepphy_eval_run_id(self, run_root: Path) -> str:
        deepphy_root = run_root / "deepphy_eval"
        if not deepphy_root.is_dir():
            return ""
        best_run_id = ""
        best_mtime = 0.0
        for child in deepphy_root.iterdir():
            if not child.is_dir():
                continue
            completion = self._completed_deepphy_run_summary(child)
            if completion is None:
                continue
            _payload, completion_mtime = completion
            if completion_mtime >= best_mtime:
                best_run_id = child.name
                best_mtime = completion_mtime
        return best_run_id

    def _learning_run_completion_counts(
        self,
        run_root: Path,
        payload: dict[str, Any],
    ) -> tuple[int, int, float]:
        try:
            payload_completed_labels = int(payload.get("completed_label_count") or 0)
        except (TypeError, ValueError):
            payload_completed_labels = 0
        try:
            total_labels = int(payload.get("primitive_label_count") or 0)
        except (TypeError, ValueError):
            total_labels = 0
        if total_labels <= 0:
            label_runs = payload.get("label_runs")
            if isinstance(label_runs, list):
                total_labels = len(label_runs)
        checkpoint_count = 0
        latest_checkpoint_mtime = 0.0
        has_label_dirs = False
        for label_dir in run_root.glob("label_*"):
            if label_dir.is_dir():
                has_label_dirs = True
                break
        for label_result_path in run_root.glob("label_*/label_result.json"):
            if not label_result_path.is_file():
                continue
            checkpoint_count += 1
            try:
                latest_checkpoint_mtime = max(
                    latest_checkpoint_mtime,
                    label_result_path.stat().st_mtime,
                )
            except OSError:
                continue
        if has_label_dirs:
            completed_labels = checkpoint_count
        else:
            completed_labels = max(payload_completed_labels, checkpoint_count)
        if total_labels > 0:
            completed_labels = max(0, min(completed_labels, total_labels))
        return completed_labels, total_labels, latest_checkpoint_mtime

    def _pooltool_attempt_limit_violation(
        self,
        run_root: Path,
        *,
        attempt_limit: int,
    ) -> tuple[int, int, float] | None:
        tasks_root = run_root / "tasks"
        if not tasks_root.is_dir():
            return None
        violating_task_count = 0
        max_attempt_count = 0
        latest_mtime = 0.0
        for task_result_path in sorted(tasks_root.glob("*/*/task_result.json")):
            try:
                task_payload = json.loads(task_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(task_payload, dict):
                continue
            if str(task_payload.get("env_id") or "").strip().lower() != "pooltool":
                continue
            records = task_payload.get("records")
            if not isinstance(records, list):
                continue
            attempt_count = len(records)
            if attempt_count <= attempt_limit:
                continue
            violating_task_count += 1
            max_attempt_count = max(max_attempt_count, attempt_count)
            latest_mtime = max(latest_mtime, task_result_path.stat().st_mtime)
        if violating_task_count <= 0:
            return None
        return violating_task_count, max_attempt_count, latest_mtime

    def _clear_pooltool_attempt_limit_job_failures(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        changed = False
        attempt_limit = QueueSettings.model_validate(
            state.get("queue_settings") or {}
        ).pooltool_attempt_limit
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if str(job.spec.action or "").strip() != "evaluate-deepphy":
                continue
            if job.status in {JobStatus.cancelled, JobStatus.stopped}:
                continue
            if "PoolTool DeepPHY run failed server-side:" not in str(job.message or ""):
                continue
            run_root = self._resolved_job_run_dir_path(job)
            if run_root is None:
                continue
            completion = self._completed_deepphy_run_summary(run_root)
            if completion is not None:
                payload, completion_mtime = completion
                finished_at = datetime.fromtimestamp(completion_mtime, tz=timezone.utc)
                job.status = JobStatus.succeeded
                job.updated_at = finished_at
                job.finished_at = finished_at
                job.assigned_client_id = None
                job.exit_code = 0
                job.pending_commands = []
                job.message = (
                    f"Completed. PoolTool tasks over {attempt_limit} attempts remain task-level "
                    "failures only."
                )
                metadata = dict(job.spec.metadata or {})
                metadata["progress"] = {
                    "fraction": 1.0,
                    "phase": "Completed",
                    "summary": "PoolTool eval completed",
                }
                metadata["pooltool_attempt_limit"] = attempt_limit
                job.spec.metadata = metadata
                state["jobs"][job_id] = job.model_dump(mode="json")
                changed = True
                continue
            job.status = JobStatus.queued
            job.updated_at = now
            job.finished_at = None
            job.started_at = None
            job.assigned_client_id = None
            job.exit_code = None
            job.pending_commands = []
            job.message = (
                f"Re-queued after removing server-side PoolTool job failure policy. "
                f"Tasks over {attempt_limit} attempts remain task-level failures only."
            )
            metadata = dict(job.spec.metadata or {})
            metadata["progress"] = {
                "fraction": 0.0,
                "phase": "Queued",
                "summary": "Re-queued after task-level PoolTool attempt policy fix",
            }
            metadata["pooltool_attempt_limit"] = attempt_limit
            job.spec.metadata = metadata
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _reconcile_completed_deepphy_jobs(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
                JobStatus.stopped,
            }:
                continue
            if str(job.spec.action or "").strip() != "evaluate-deepphy":
                continue
            run_root = self._resolved_job_run_dir_path(job)
            if run_root is None:
                continue
            completion = self._completed_deepphy_run_summary(run_root)
            if completion is None:
                completion = self._completed_threshold_deepphy_run_summary(job, run_root)
            if completion is None:
                continue
            completion_payload, completion_mtime = completion
            if job.status not in ACTIVE_JOB_STATUSES and completion_mtime <= job.updated_at.timestamp():
                continue
            assigned_client_id = str(job.assigned_client_id or "").strip()
            job.status = JobStatus.succeeded
            job.updated_at = now
            job.finished_at = now
            job.assigned_client_id = None
            job.exit_code = 0
            job.pending_commands = []
            completion_mode = str(completion_payload.get("completion_mode") or "").strip().lower()
            if completion_mode == "task-threshold":
                completed_task_count = int(completion_payload.get("completed_task_count") or 0)
                task_count = int(completion_payload.get("task_count") or PHYRE_IMAGE_ONLY_COMPLETION_TASK_TARGET)
                job.message = (
                    "Marked complete after reaching the PHYRE image-only completion threshold "
                    f"({completed_task_count} completed tasks, target {task_count})."
                )
            else:
                task_count = int(completion_payload.get("task_count") or 0)
                success_count = int(completion_payload.get("success_count") or 0)
                failure_count = int(completion_payload.get("failure_count") or 0)
                job.message = (
                    f"Reconciled from completed DeepPHY results "
                    f"({success_count}/{task_count} success, {failure_count} fail)"
                )
            metadata = dict(job.spec.metadata or {})
            metadata["progress"] = {
                "fraction": 1.0,
                "phase": "Completed",
                "summary": "Done",
            }
            job.spec.metadata = metadata
            state["jobs"][job_id] = job.model_dump(mode="json")
            if assigned_client_id:
                self._detach_job_from_client_record(state, assigned_client_id, job_id)
            changed = True
        return changed

    def _completed_deepphy_run_summary(self, run_root: Path) -> tuple[dict[str, Any], float] | None:
        results_path = run_root / "results.json"
        if not results_path.is_file():
            return None
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        env_summaries = payload.get("env_summaries")
        if isinstance(env_summaries, dict):
            summaries = [item for item in env_summaries.values() if isinstance(item, dict)]
        elif isinstance(env_summaries, list):
            summaries = [item for item in env_summaries if isinstance(item, dict)]
        else:
            summaries = []
        if not summaries:
            return None
        task_count = 0
        success_count = 0
        failure_count = 0
        for summary in summaries:
            try:
                env_task_count = int(summary.get("task_count") or summary.get("total_tasks") or 0)
                env_success_count = int(summary.get("success_count") or summary.get("successful_tasks") or 0)
                env_failure_count = int(summary.get("failure_count") or summary.get("failed_tasks") or 0)
            except (TypeError, ValueError):
                return None
            if env_task_count <= 0:
                return None
            if env_success_count + env_failure_count < env_task_count:
                return None
            task_count += env_task_count
            success_count += env_success_count
            failure_count += env_failure_count
        return (
            {
                "task_count": task_count,
                "success_count": success_count,
                "failure_count": failure_count,
            },
            results_path.stat().st_mtime,
        )

    def _reconcile_incomplete_library_ablation_jobs(self, state: dict[str, Any]) -> bool:
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if str(job.spec.action or "").strip() != "evaluate-library-case":
                continue
            if job.status != JobStatus.succeeded:
                continue
            run_root = self._resolved_job_run_dir_path(job)
            if run_root is None:
                continue
            progress = self._library_ablation_summary_progress(run_root)
            if progress is not None:
                completed_seed_run_count, total_seed_run_count, _mtime = progress
                if total_seed_run_count > 0 and completed_seed_run_count == total_seed_run_count:
                    continue
            job.status = JobStatus.queued
            job.updated_at = _utc_now()
            job.finished_at = None
            job.assigned_client_id = None
            job.exit_code = None
            job.pending_commands = []
            prerequisite_reason = self._job_prerequisite_missing_reason(job)
            if prerequisite_reason:
                job.status = JobStatus.paused
                job.message = f"Waiting for prerequisite: {prerequisite_reason}"
            else:
                if progress is None:
                    job.message = "Library ablation missing usable summary; re-queued automatically"
                else:
                    completed_seed_run_count, total_seed_run_count, _mtime = progress
                    job.message = (
                        "Library ablation incomplete "
                        f"({completed_seed_run_count}/{total_seed_run_count} runs complete); "
                        "re-queued automatically"
                    )
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _reconcile_completed_library_ablation_jobs(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status in {
                JobStatus.succeeded,
                JobStatus.failed,
                JobStatus.cancelled,
                JobStatus.stopped,
            }:
                continue
            if str(job.spec.action or "").strip() != "evaluate-library-case":
                continue
            run_root = self._job_run_dir_path(job)
            if run_root is None:
                continue
            completion = self._completed_library_ablation_summary(run_root)
            if completion is None:
                continue
            completed_payload, completion_mtime = completion
            if job.status not in ACTIVE_JOB_STATUSES and completion_mtime <= job.updated_at.timestamp():
                continue
            job.status = JobStatus.succeeded
            job.updated_at = now
            job.finished_at = now
            job.assigned_client_id = None
            job.exit_code = 0
            job.pending_commands = []
            job.message = (
                "Reconciled from completed library ablation summary "
                f"({completed_payload['completed_seed_run_count']}/{completed_payload['total_seed_run_count']} runs)"
            )
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _completed_library_ablation_summary(self, run_root: Path) -> tuple[dict[str, Any], float] | None:
        progress = self._library_ablation_summary_progress(run_root)
        if progress is None:
            return None
        completed_seed_run_count, total_seed_run_count, mtime = progress
        if total_seed_run_count <= 0:
            return None
        if completed_seed_run_count <= 0:
            return None
        if completed_seed_run_count < total_seed_run_count:
            return None
        return (
            {
                "completed_seed_run_count": completed_seed_run_count,
                "total_seed_run_count": total_seed_run_count,
            },
            mtime,
        )

    def _library_ablation_summary_progress(self, run_root: Path) -> tuple[int, int, float] | None:
        summary_path = run_root / "summary.json"
        if not summary_path.is_file():
            return None
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            completed_seed_run_count = int(
                payload.get("completed_evaluation_run_count")
                or payload.get("completed_seed_run_count")
                or 0
            )
            total_seed_run_count = int(
                payload.get("total_evaluation_run_count")
                or payload.get("total_seed_run_count")
                or 0
            )
        except (TypeError, ValueError):
            return None
        return (completed_seed_run_count, total_seed_run_count, summary_path.stat().st_mtime)

    def _normalize_queue_settings(self, state: dict[str, Any]) -> bool:
        normalized = QueueSettings.model_validate(
            state.get("queue_settings") or {}
        ).model_dump(mode="json")
        if state.get("queue_settings") == normalized:
            return False
        state["queue_settings"] = normalized
        return True

    def _normalize_repeat_learning_queue(self, state: dict[str, Any]) -> bool:
        changed = False
        learning_groups: dict[tuple[str, str, str], dict[int, list[JobRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        learning_templates: dict[tuple[str, str, str], JobRecord] = {}
        canonical_learning_jobs: dict[tuple[str, str, str], dict[int, JobRecord]] = defaultdict(dict)

        for raw_job in state["jobs"].values():
            job = JobRecord.model_validate(raw_job)
            descriptor = self._repeat_learning_descriptor(job)
            if descriptor is None:
                continue
            group_key = (
                descriptor["env_id"],
                descriptor["family"],
                descriptor["batch_prefix"],
            )
            learning_groups[group_key][descriptor["run_index"]].append(job)
            if learning_templates.get(group_key) is None and self._job_request_payload(job):
                learning_templates[group_key] = job

        for group_key, jobs_by_index in learning_groups.items():
            for run_index, jobs in jobs_by_index.items():
                if group_key[1] == "automatic":
                    for automatic_job in jobs:
                        if self._retarget_repeat_automatic_job(
                            state,
                            job=automatic_job,
                            run_index=run_index,
                        ):
                            changed = True
                canonical = self._select_canonical_repeat_job(jobs)
                if canonical is not None and canonical.status in REPEAT_COUNTED_STATUSES:
                    canonical_learning_jobs[group_key][run_index] = canonical
                for extra in self._repeat_queued_duplicates(jobs, keep=canonical):
                    if self._cancel_inactive_repeat_job(
                        state,
                        extra,
                        message="Cancelled duplicate queued repeat learning job",
                    ):
                        changed = True
            if self._ensure_repeat_learning_jobs(
                state,
                group_key=group_key,
                canonical_jobs=canonical_learning_jobs[group_key],
                template_job=learning_templates.get(group_key),
            ):
                changed = True

        eval_groups: dict[tuple[str, str, str, int], dict[int, list[JobRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )
        eval_templates: dict[tuple[str, str, str], JobRecord] = {}
        for raw_job in state["jobs"].values():
            job = JobRecord.model_validate(raw_job)
            descriptor = self._repeat_eval_descriptor(job)
            if descriptor is None:
                continue
            family_key = (
                descriptor["env_id"],
                descriptor["family"],
                descriptor["batch_prefix"],
            )
            group_key = (*family_key, descriptor["run_index"])
            eval_groups[group_key][descriptor["eval_index"]].append(job)
            if eval_templates.get(family_key) is None and self._job_request_payload(job):
                eval_templates[family_key] = job

        for eval_group in eval_groups.values():
            for jobs in eval_group.values():
                canonical = self._select_canonical_repeat_job(jobs)
                for extra in self._repeat_queued_duplicates(jobs, keep=canonical):
                    if self._cancel_inactive_repeat_job(
                        state,
                        extra,
                        message="Cancelled duplicate queued repeat evaluation",
                    ):
                        changed = True
                if canonical is not None and self._cancel_inactive_repeat_job(
                    state,
                    canonical,
                    message="Cancelled standalone repeat evaluation; required runs now rely on embedded auto-evals",
                ):
                    changed = True

        for family_key, learning_jobs_by_index in canonical_learning_jobs.items():
            template_job = eval_templates.get(family_key)
            run_indices = sorted(
                set(range(1, REPEAT_LEARNING_TARGET_RUNS + 1)) | set(learning_jobs_by_index.keys())
            )
            for run_index in run_indices:
                learning_job = learning_jobs_by_index.get(run_index)
                eval_group = eval_groups.get((*family_key, run_index), {})
                for eval_index, jobs in eval_group.items():
                    if eval_index > REPEAT_EVALS_PER_RUN:
                        for queued_extra in jobs:
                            if self._cancel_inactive_repeat_job(
                                state,
                                queued_extra,
                                message="Cancelled excess inactive repeat evaluation",
                            ):
                                changed = True
                if learning_job is None:
                    continue
        if self._dedupe_repeat_eval_run_ids(state):
            changed = True
        return changed

    def _repeat_learning_descriptor(self, job: JobRecord) -> dict[str, Any] | None:
        action = str(job.spec.action or "").strip()
        family = REPEAT_LEARNING_ACTIONS.get(action)
        if family is None:
            return None
        metadata = dict(job.spec.metadata or {})
        request_payload = self._job_request_payload(job)
        required_envs = list(job.spec.constraints.required_environments or [])
        env_source = metadata.get("env_id")
        if not env_source and required_envs:
            env_source = required_envs[-1]
        env_id = str(env_source or "").strip().lower()
        marker = str(
            request_payload.get("run_timestamp")
            or Path(str(metadata.get("run_dir") or metadata.get("output_path") or request_payload.get("output_path") or "")).name
        ).strip()
        match = REPEAT_RUN_SUFFIX_RE.match(marker)
        if not env_id or match is None:
            return None
        run_dir = str(metadata.get("run_dir") or "").strip()
        run_id = Path(run_dir).name if run_dir else marker
        actual_run_index = int(match.group("index"))
        effective_run_index = self._repeat_run_slot_override(job, default=actual_run_index)
        return {
            "env_id": env_id,
            "family": family,
            "batch_prefix": str(match.group("prefix")),
            "run_index": effective_run_index,
            "actual_run_index": actual_run_index,
            "run_id": run_id or marker,
        }

    def _repeat_eval_descriptor(self, job: JobRecord) -> dict[str, Any] | None:
        if str(job.spec.action or "").strip() != "evaluate-deepphy":
            return None
        metadata = dict(job.spec.metadata or {})
        request_payload = self._job_request_payload(job)
        env_id = str(metadata.get("env_id") or "").strip().lower()
        pattern_run_dir = str(
            metadata.get("pattern_run_dir") or request_payload.get("pattern_run_dir") or ""
        ).strip()
        if not env_id or not pattern_run_dir:
            return None
        run_id = Path(pattern_run_dir).name
        run_match = REPEAT_RUN_SUFFIX_RE.match(run_id)
        if run_match is None:
            return None
        family = self._repeat_family_from_run_dir(pattern_run_dir)
        if family is None:
            return None
        eval_marker = str(
            metadata.get("evaluation_run_id")
            or request_payload.get("run_id")
            or Path(str(metadata.get("evaluation_run_dir") or request_payload.get("output_dir") or "")).name
        ).strip()
        eval_match = REPEAT_EVAL_SUFFIX_RE.match(eval_marker)
        if eval_match is None:
            return None
        actual_run_index = int(run_match.group("index"))
        effective_run_index = self._repeat_run_slot_override(job, default=actual_run_index)
        return {
            "env_id": env_id,
            "family": family,
            "batch_prefix": str(run_match.group("prefix")),
            "run_index": effective_run_index,
            "actual_run_index": actual_run_index,
            "run_id": run_id,
            "eval_index": int(eval_match.group("index")),
        }

    def _repeat_run_slot_override(self, job: JobRecord, *, default: int | None = None) -> int:
        override = self._explicit_repeat_run_slot_override(job)
        if override is not None:
            return override
        if default is not None:
            return int(default)
        return 0

    def _explicit_repeat_run_slot_override(self, job: JobRecord) -> int | None:
        metadata = dict(job.spec.metadata or {})
        request_payload = self._job_request_payload(job)
        raw_override = metadata.get("repeat_run_slot_override")
        if raw_override in (None, ""):
            raw_override = request_payload.get("repeat_run_slot_override")
        try:
            override = int(raw_override)
        except (TypeError, ValueError):
            override = None
        return override if override is not None and override > 0 else None

    def _repeat_family_from_run_dir(self, pattern_run_dir: str) -> str | None:
        normalized = str(pattern_run_dir).replace("\\", "/").lower()
        if "/supervised/" in normalized:
            return "supervised"
        if "/unsupervised/" in normalized:
            return "unsupervised"
        if "/automatic/" in normalized:
            return "automatic"
        run_id = Path(str(pattern_run_dir)).name.lower()
        if "-supervised-r" in run_id:
            return "supervised"
        if "-unsupervised-r" in run_id:
            return "unsupervised"
        if "-automatic-r" in run_id:
            return "automatic"
        return None

    def _select_canonical_repeat_job(self, jobs: list[JobRecord]) -> JobRecord | None:
        if not jobs:
            return None
        return sorted(
            jobs,
            key=lambda item: (
                0 if item.status in REPEAT_COUNTED_STATUSES else 1,
                0 if item.status != JobStatus.queued else 1,
                item.created_at,
                item.job_id,
            ),
        )[0]

    def _repeat_queued_duplicates(self, jobs: list[JobRecord], *, keep: JobRecord | None) -> list[JobRecord]:
        duplicates: list[JobRecord] = []
        for job in sorted(jobs, key=lambda item: (item.created_at, item.job_id)):
            if keep is not None and job.job_id == keep.job_id:
                continue
            if job.status == JobStatus.queued:
                duplicates.append(job)
        return duplicates

    def _cancel_inactive_repeat_job(self, state: dict[str, Any], job: JobRecord, *, message: str) -> bool:
        raw_job = state["jobs"].get(job.job_id)
        if raw_job is None:
            return False
        current = JobRecord.model_validate(raw_job)
        if current.status not in {
            JobStatus.queued,
            JobStatus.paused,
            JobStatus.waiting_for_dependencies,
        }:
            return False
        current.status = JobStatus.cancelled
        current.updated_at = _utc_now()
        current.finished_at = current.updated_at
        current.assigned_client_id = None
        current.started_at = None
        current.exit_code = None
        current.pending_commands = []
        current.message = message
        state["jobs"][current.job_id] = current.model_dump(mode="json")
        return True

    def _pause_repeat_learning_over_cap_job(
        self,
        state: dict[str, Any],
        job: JobRecord,
        *,
        message: str,
    ) -> bool:
        raw_job = state["jobs"].get(job.job_id)
        if raw_job is None:
            return False
        current = JobRecord.model_validate(raw_job)
        if current.status in {
            JobStatus.succeeded,
            JobStatus.cancelled,
            JobStatus.failed,
            JobStatus.stopped,
        }:
            return False
        now = _utc_now()
        if current.status in {
            JobStatus.queued,
            JobStatus.paused,
            JobStatus.waiting_for_dependencies,
        }:
            current.status = JobStatus.paused
            current.updated_at = now
            current.finished_at = now
            current.assigned_client_id = None
            current.started_at = None
            current.exit_code = None
            current.pending_commands = []
            current.message = message
            state["jobs"][current.job_id] = current.model_dump(mode="json")
            self._detach_job_from_client_record(
                state, str(raw_job.get("assigned_client_id") or ""), current.job_id
            )
            return True
        if current.status in ACTIVE_JOB_STATUSES:
            current.status = JobStatus.pausing
            current.updated_at = now
            current.message = message
            self._append_command(current, JobCommandType.pause)
            state["jobs"][current.job_id] = current.model_dump(mode="json")
            return True
        return False

    def _job_request_payload(self, job: JobRecord) -> dict[str, Any]:
        payload = (job.spec.metadata or {}).get("request")
        return dict(payload) if isinstance(payload, dict) else {}

    def _ensure_repeat_learning_jobs(
        self,
        state: dict[str, Any],
        *,
        group_key: tuple[str, str, str],
        canonical_jobs: dict[int, JobRecord],
        template_job: JobRecord | None,
    ) -> bool:
        if template_job is None:
            return False
        changed = False
        for run_index in range(1, REPEAT_LEARNING_TARGET_RUNS + 1):
            if run_index in canonical_jobs:
                continue
            request = self._build_missing_repeat_learning_request(
                template_job=template_job,
                batch_prefix=group_key[2],
                run_index=run_index,
            )
            if request is None:
                continue
            created = self._create_job_in_state(
                state,
                request,
                message="Queued from normalized repeat learning plan",
            )
            canonical_jobs[run_index] = created
            changed = True
        return changed

    def _retarget_repeat_automatic_job(
        self,
        state: dict[str, Any],
        *,
        job: JobRecord,
        run_index: int,
    ) -> bool:
        if str(job.spec.action or "").strip() != "automatic-launch":
            return False
        if job.status in {
            JobStatus.assigned,
            JobStatus.starting,
            JobStatus.running,
            JobStatus.resuming,
            JobStatus.stopping,
            JobStatus.succeeded,
            JobStatus.cancelled,
        }:
            return False
        raw_job = state["jobs"].get(job.job_id)
        if raw_job is None:
            return False
        current = JobRecord.model_validate(raw_job)
        metadata = dict(current.spec.metadata or {})
        request_payload = self._job_request_payload(current)
        raw_source_run_dir = str(
            metadata.get("source_supervised_run_dir")
            or request_payload.get("source_supervised_run_dir")
            or ""
        ).strip()
        if not raw_source_run_dir:
            return False
        desired_source_run_dir = self._resolve_repeat_automatic_source_run_dir(
            source_supervised_run_dir=raw_source_run_dir,
            automatic_run_dir=str(metadata.get("run_dir") or request_payload.get("run_dir") or "").strip(),
            run_index=run_index,
        )
        if desired_source_run_dir is None:
            return False
        desired_source_run_id = Path(desired_source_run_dir).name
        requested_eval_run_id = str(
            metadata.get("source_evaluation_run_id")
            or request_payload.get("source_evaluation_run_id")
            or ""
        ).strip()
        desired_eval_run_id = self._retarget_repeat_source_eval_id(
            requested_eval_run_id,
            current_source_run_id=Path(raw_source_run_dir).name,
            desired_source_run_id=desired_source_run_id,
        )
        changed = False
        if metadata.get("source_supervised_run_dir") != desired_source_run_dir:
            metadata["source_supervised_run_dir"] = desired_source_run_dir
            changed = True
        if metadata.get("source_supervised_run_id") != desired_source_run_id:
            metadata["source_supervised_run_id"] = desired_source_run_id
            changed = True
        if desired_eval_run_id and metadata.get("source_evaluation_run_id") != desired_eval_run_id:
            metadata["source_evaluation_run_id"] = desired_eval_run_id
            changed = True
        if not request_payload:
            if not changed:
                return False
            current.spec.metadata = metadata
            current.spec.argv = self._replace_or_add_flag(
                self._remove_flag(current.spec.argv, "--source-pattern-library"),
                "--source-supervised-run-dir",
                desired_source_run_dir,
            )
            if desired_eval_run_id:
                current.spec.argv = self._replace_or_add_flag(
                    current.spec.argv,
                    "--source-evaluation-run-id",
                    desired_eval_run_id,
                )
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                rewritten_resume = self._replace_or_add_flag(
                    self._remove_flag([str(part) for part in resume_argv], "--source-pattern-library"),
                    "--source-supervised-run-dir",
                    desired_source_run_dir,
                )
                if desired_eval_run_id:
                    rewritten_resume = self._replace_or_add_flag(
                        rewritten_resume,
                        "--source-evaluation-run-id",
                        desired_eval_run_id,
                    )
                metadata["resume_argv"] = rewritten_resume
                current.spec.metadata = metadata
            current.updated_at = _utc_now()
            state["jobs"][current.job_id] = current.model_dump(mode="json")
            return True
        if str(request_payload.get("source_supervised_run_dir") or "").strip() != desired_source_run_dir:
            request_payload["source_supervised_run_dir"] = desired_source_run_dir
            request_payload.pop("source_pattern_library", None)
            changed = True
        if desired_eval_run_id and str(request_payload.get("source_evaluation_run_id") or "").strip() != desired_eval_run_id:
            request_payload["source_evaluation_run_id"] = desired_eval_run_id
            changed = True
        if not changed:
            return False
        payload = dict(request_payload)
        payload["enqueue"] = True
        rebuilt = self._build_repeat_job_request("automatic-launch", payload)
        if rebuilt is None:
            current.spec.metadata = metadata
            current.spec.argv = self._replace_or_add_flag(
                self._remove_flag(current.spec.argv, "--source-pattern-library"),
                "--source-supervised-run-dir",
                desired_source_run_dir,
            )
            if desired_eval_run_id:
                current.spec.argv = self._replace_or_add_flag(
                    current.spec.argv,
                    "--source-evaluation-run-id",
                    desired_eval_run_id,
                )
            metadata["request"] = request_payload
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                rewritten_resume = self._replace_or_add_flag(
                    self._remove_flag([str(part) for part in resume_argv], "--source-pattern-library"),
                    "--source-supervised-run-dir",
                    desired_source_run_dir,
                )
                if desired_eval_run_id:
                    rewritten_resume = self._replace_or_add_flag(
                        rewritten_resume,
                        "--source-evaluation-run-id",
                        desired_eval_run_id,
                    )
                metadata["resume_argv"] = rewritten_resume
            current.spec.metadata = metadata
            current.updated_at = _utc_now()
            state["jobs"][current.job_id] = current.model_dump(mode="json")
            return True
        preserved_metadata = dict(current.spec.metadata or {})
        rebuilt_metadata = dict(rebuilt.spec.metadata or {})
        for key in (
            "queue_requested",
            "launch_mode",
            "queue_order",
            "preferred_client_id",
            "preferred_client_name",
            "workers",
            "requested_workers",
            "funsearch_template",
        ):
            if key in preserved_metadata and key not in rebuilt_metadata:
                rebuilt_metadata[key] = preserved_metadata[key]
        rebuilt.spec.metadata = rebuilt_metadata
        current.spec = rebuilt.spec
        current.updated_at = _utc_now()
        state["jobs"][current.job_id] = current.model_dump(mode="json")
        return True

    def _ensure_repeat_eval_jobs(
        self,
        state: dict[str, Any],
        *,
        family_key: tuple[str, str, str],
        learning_job: JobRecord,
        jobs_by_eval_index: dict[int, list[JobRecord]],
        template_job: JobRecord | None,
    ) -> bool:
        if template_job is None:
            return False
        changed = False
        learning_slot_override = self._explicit_repeat_run_slot_override(learning_job)
        active_eval_indices = {
            eval_index
            for eval_index, jobs in jobs_by_eval_index.items()
            if any(job.status in REPEAT_COUNTED_STATUSES for job in jobs)
        }
        for eval_index in range(1, REPEAT_EVALS_PER_RUN + 1):
            if eval_index in active_eval_indices:
                continue
            request = self._build_missing_repeat_eval_request(
                template_job=template_job,
                learning_job=learning_job,
                eval_index=eval_index,
            )
            if request is None:
                continue
            created = self._create_job_in_state(
                state,
                request,
                message="Queued from normalized repeat evaluation plan",
            )
            if learning_slot_override is not None and learning_slot_override > 0:
                created_metadata = dict(created.spec.metadata or {})
                created_metadata["repeat_run_slot_override"] = learning_slot_override
                created.spec.metadata = created_metadata
                state["jobs"][created.job_id] = created.model_dump(mode="json")
            changed = True
        return changed

    def _build_missing_repeat_learning_request(
        self,
        *,
        template_job: JobRecord,
        batch_prefix: str,
        run_index: int,
    ) -> CreateJobRequest | None:
        payload = self._job_request_payload(template_job)
        if not payload:
            return None
        payload["enqueue"] = True
        payload["run_timestamp"] = f"{batch_prefix}-r{run_index:02d}"
        payload.pop("output_path", None)
        if str(template_job.spec.action or "").strip() == "automatic-launch":
            raw_source_run_dir = str(payload.get("source_supervised_run_dir") or "").strip()
            if raw_source_run_dir:
                desired_source_run_dir = self._resolve_repeat_automatic_source_run_dir(
                    source_supervised_run_dir=raw_source_run_dir,
                    automatic_run_dir=str(payload.get("run_dir") or "").strip(),
                    run_index=run_index,
                )
                if desired_source_run_dir is not None:
                    payload["source_supervised_run_dir"] = desired_source_run_dir
                    payload.pop("source_pattern_library", None)
                    requested_eval_run_id = str(payload.get("source_evaluation_run_id") or "").strip()
                    desired_eval_run_id = self._retarget_repeat_source_eval_id(
                        requested_eval_run_id,
                        current_source_run_id=Path(raw_source_run_dir).name,
                        desired_source_run_id=Path(desired_source_run_dir).name,
                    )
                    if desired_eval_run_id:
                        payload["source_evaluation_run_id"] = desired_eval_run_id
        return self._build_repeat_job_request(str(template_job.spec.action or "").strip(), payload)

    def _retarget_repeat_run_path(self, raw_path: str, *, run_index: int) -> str | None:
        candidate = Path(str(raw_path).strip())
        match = REPEAT_RUN_SUFFIX_RE.match(candidate.name)
        if match is None:
            return None
        desired_name = f"{match.group('prefix')}-r{run_index:02d}"
        return str(candidate.with_name(desired_name))

    def _resolve_repeat_automatic_source_run_dir(
        self,
        *,
        source_supervised_run_dir: str,
        automatic_run_dir: str,
        run_index: int,
    ) -> str | None:
        desired_source_run_dir = self._retarget_repeat_run_path(
            source_supervised_run_dir,
            run_index=run_index,
        )
        if desired_source_run_dir is not None:
            return desired_source_run_dir

        automatic_run_name = Path(str(automatic_run_dir or "").strip()).name
        if not automatic_run_name:
            return None
        automatic_match = REPEAT_RUN_SUFFIX_RE.match(automatic_run_name)
        if automatic_match is None:
            return None
        automatic_prefix = automatic_match.group("prefix")
        if "automatic" not in automatic_prefix:
            return None
        desired_source_run_name = automatic_prefix.replace("automatic", "supervised") + f"-r{run_index:02d}"

        candidates: list[Path] = []
        seen: set[Path] = set()
        for source_candidate in self._candidate_pattern_run_dirs_for_prerequisite(source_supervised_run_dir):
            direct_sibling = source_candidate.with_name(desired_source_run_name)
            resolved = direct_sibling.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
        for automatic_candidate in self._candidate_pattern_run_dirs_for_prerequisite(automatic_run_dir):
            if "automatic" in automatic_candidate.parts:
                parts = list(automatic_candidate.parts)
                automatic_index = parts.index("automatic")
                parts[automatic_index] = "supervised"
                derived = Path(*parts).with_name(desired_source_run_name)
            else:
                derived = automatic_candidate.with_name(desired_source_run_name)
            resolved = derived.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)
        if not candidates:
            return None
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(candidates[0])

    def _retarget_repeat_source_eval_id(
        self,
        evaluation_run_id: str,
        *,
        current_source_run_id: str,
        desired_source_run_id: str,
    ) -> str:
        requested = str(evaluation_run_id or "").strip()
        if not requested or requested == "__auto_source_eval__":
            return requested
        prefix = f"{current_source_run_id}-repeat-"
        if requested.startswith(prefix):
            suffix = requested[len(prefix):]
            return f"{desired_source_run_id}-repeat-{suffix}"
        return requested

    def _build_missing_repeat_eval_request(
        self,
        *,
        template_job: JobRecord,
        learning_job: JobRecord,
        eval_index: int,
    ) -> CreateJobRequest | None:
        payload = self._job_request_payload(template_job)
        if not payload:
            return None
        learning_descriptor = self._repeat_learning_descriptor(learning_job)
        if learning_descriptor is None:
            return None
        learning_run_dir = str((learning_job.spec.metadata or {}).get("run_dir") or "").strip()
        learning_run_id = Path(learning_run_dir).name if learning_run_dir else str(learning_descriptor["run_id"])
        payload["enqueue"] = True
        payload.pop("pattern_library", None)
        payload.pop("output_dir", None)
        payload["pattern_run_dir"] = learning_run_dir or str(
            (learning_job.spec.metadata or {}).get("output_path") or ""
        )
        payload["run_id"] = f"{learning_run_id}-repeat-{eval_index:02d}"
        payload["source_run_id"] = learning_run_id
        template_metadata = dict(template_job.spec.metadata or {})
        enabled_labels = template_metadata.get("enabled_labels")
        disabled_labels = template_metadata.get("disabled_labels")
        if isinstance(enabled_labels, list):
            payload["enabled_labels"] = list(enabled_labels)
        if isinstance(disabled_labels, list):
            payload["disabled_labels"] = list(disabled_labels)
        return self._build_repeat_job_request("evaluate-deepphy", payload)

    def _build_repeat_job_request(self, action: str, payload: dict[str, Any]) -> CreateJobRequest | None:
        try:
            from .legacy_jobs import (
                create_automatic_job,
                create_deepphy_eval_job,
                create_supervised_job,
                create_unsupervised_job,
            )
        except Exception:
            return None
        builders = {
            "supervised-learn-and-evaluate": create_supervised_job,
            "unsupervised-launch": create_unsupervised_job,
            "automatic-launch": create_automatic_job,
            "evaluate-deepphy": create_deepphy_eval_job,
        }
        builder = builders.get(str(action or "").strip())
        if builder is None:
            return None
        try:
            return builder(payload)
        except Exception:
            return None

    def _create_job_in_state(
        self,
        state: dict[str, Any],
        request: CreateJobRequest,
        *,
        message: str | None = None,
    ) -> JobRecord:
        now = _utc_now()
        spec = request.spec.model_copy(deep=True)
        metadata = dict(spec.metadata or {})
        metadata.setdefault("queue_requested", bool(request.enqueue))
        metadata.setdefault("launch_mode", "queue" if request.enqueue else "run_now")
        if request.enqueue:
            metadata.setdefault("queue_order", self._next_queue_order(state))
        spec.metadata = metadata
        job = JobRecord(
            job_id=uuid.uuid4().hex,
            status=JobStatus.queued,
            created_at=now,
            updated_at=now,
            message=message,
            spec=spec,
        )
        state["jobs"][job.job_id] = job.model_dump(mode="json")
        self._job_root(job.job_id).mkdir(parents=True, exist_ok=True)
        return job

    def _dedupe_repeat_eval_run_ids(self, state: dict[str, Any]) -> bool:
        changed = False
        grouped: dict[tuple[str, str], list[JobRecord]] = defaultdict(list)
        for raw_job in state["jobs"].values():
            job = JobRecord.model_validate(raw_job)
            if str(job.spec.action or "").strip() != "evaluate-deepphy":
                continue
            metadata = dict(job.spec.metadata or {})
            pattern_run_dir = str(metadata.get("pattern_run_dir") or "").strip()
            eval_run_id = str(metadata.get("evaluation_run_id") or "").strip()
            if not pattern_run_dir or not eval_run_id:
                continue
            if REPEAT_EVAL_SUFFIX_RE.match(eval_run_id) is None:
                continue
            grouped[(pattern_run_dir, eval_run_id)].append(job)
        for jobs in grouped.values():
            keep = self._select_canonical_repeat_job(jobs)
            for extra in self._repeat_queued_duplicates(jobs, keep=keep):
                if self._cancel_inactive_repeat_job(
                    state,
                    extra,
                    message="Cancelled duplicate queued repeat evaluation",
                ):
                    changed = True
        return changed

    def _prune_stale_idle_clients(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        active_jobs_by_client: dict[str, int] = {}
        for raw_job in state["jobs"].values():
            job = JobRecord.model_validate(raw_job)
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            client_id = str(job.assigned_client_id or "").strip()
            if not client_id:
                continue
            active_jobs_by_client[client_id] = active_jobs_by_client.get(client_id, 0) + 1

        remove_client_ids: list[str] = []
        for client_id, raw_client in state["clients"].items():
            client = ClientRecord.model_validate(raw_client)
            if now - client.last_heartbeat <= CLIENT_PRUNE_TIMEOUT:
                continue
            if active_jobs_by_client.get(client_id, 0) > 0:
                continue
            remove_client_ids.append(client_id)

        if not remove_client_ids:
            return False

        for client_id in remove_client_ids:
            state["clients"].pop(client_id, None)
            state.setdefault("client_commands", {}).pop(client_id, None)
        return True

    def _recover_stale_clients(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        changed = False
        stale_client_ids: set[str] = set()

        for client_id, raw_client in state["clients"].items():
            client = ClientRecord.model_validate(raw_client)
            if now - client.last_heartbeat <= CLIENT_HEARTBEAT_TIMEOUT:
                continue
            stale_client_ids.add(client_id)
            if client.status != ClientStatus.offline:
                client.status = ClientStatus.offline
                state["clients"][client_id] = client.model_dump(mode="json")
                changed = True

        if not stale_client_ids:
            return changed

        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.assigned_client_id not in stale_client_ids:
                continue
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            self._mark_job_avoid_client(job, client_id=job.assigned_client_id, now=now)
            job.status = JobStatus.queued
            job.assigned_client_id = None
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.exit_code = None
            job.pending_commands = []
            job.message = (
                f"Recovered after client {raw_job.get('assigned_client_id') or ''} heartbeat timeout"
            ).strip()
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True

        return changed

    def _requeue_jobs_for_shutdown(self, state: dict[str, Any], client_id: str) -> bool:
        now = _utc_now()
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.assigned_client_id != client_id:
                continue
            if job.status == JobStatus.paused:
                job.assigned_client_id = None
                job.updated_at = now
                state["jobs"][job_id] = job.model_dump(mode="json")
                changed = True
                continue
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            self._mark_job_avoid_client(job, client_id=client_id, now=now)
            job.status = JobStatus.queued
            job.assigned_client_id = None
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.exit_code = None
            job.pending_commands = []
            job.message = f"Re-queued after client {client_id} shutdown"
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _recover_orphan_active_jobs(self, state: dict[str, Any]) -> bool:
        now = _utc_now()
        changed = False
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            if str(job.assigned_client_id or "").strip():
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            job.status = JobStatus.queued
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.exit_code = None
            job.pending_commands = []
            job.message = "Recovered after missing client assignment"
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _reconcile_client_current_job_views(self, state: dict[str, Any]) -> bool:
        valid_by_client: dict[str, set[str]] = {}
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            if not self._job_counts_toward_slot_limit(job):
                continue
            client_id = str(job.assigned_client_id or "").strip()
            if not client_id:
                continue
            valid_by_client.setdefault(client_id, set()).add(job_id)
        changed = False
        for client_id, raw_client in state["clients"].items():
            client = ClientRecord.model_validate(raw_client)
            valid_jobs = sorted(valid_by_client.get(client_id, set()))
            if list(client.current_jobs) == valid_jobs:
                continue
            client.current_jobs = valid_jobs
            state["clients"][client_id] = client.model_dump(mode="json")
            changed = True
        return changed

    def _is_stale_client_record(
        self,
        raw_client: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> bool:
        client = ClientRecord.model_validate(raw_client)
        reference = now or _utc_now()
        return reference - client.last_heartbeat > CLIENT_HEARTBEAT_TIMEOUT

    def _client_instance_key(self, raw_client: dict[str, Any]) -> str:
        client = ClientRecord.model_validate(raw_client)
        metadata = dict(client.capabilities.metadata or {})
        return str(metadata.get("client_instance_key") or "").strip()

    def _client_replacement_allowed(
        self,
        state: dict[str, Any],
        *,
        previous_client_id: str,
        replacement_client_id: str,
        now: datetime,
    ) -> bool:
        if previous_client_id == replacement_client_id:
            return True
        previous_raw = state["clients"].get(previous_client_id)
        replacement_raw = state["clients"].get(replacement_client_id)
        if previous_raw is None or replacement_raw is None:
            return True
        if self._is_stale_client_record(previous_raw, now=now):
            return True
        previous_key = self._client_instance_key(previous_raw)
        replacement_key = self._client_instance_key(replacement_raw)
        return bool(previous_key) and previous_key == replacement_key

    def _detach_job_from_client_record(
        self,
        state: dict[str, Any],
        client_id: str,
        job_id: str,
    ) -> bool:
        raw_client = state["clients"].get(client_id)
        if raw_client is None:
            return False
        client = ClientRecord.model_validate(raw_client)
        remaining_jobs = [item for item in client.current_jobs if item != job_id]
        if remaining_jobs == list(client.current_jobs):
            return False
        client.current_jobs = remaining_jobs
        state["clients"][client_id] = client.model_dump(mode="json")
        return True

    def _reconcile_client_reported_jobs(
        self,
        state: dict[str, Any],
        client_id: str,
        *,
        reported_current_jobs: set[str],
    ) -> bool:
        changed = False
        now = _utc_now()
        raw_client = state["clients"].get(client_id)
        if raw_client is None:
            return False
        client = ClientRecord.model_validate(raw_client)
        active_jobs = self._active_jobs_from_state(state)
        known_client_ids = set(state["clients"])
        for job_id in reported_current_jobs:
            raw = state["jobs"].get(job_id)
            if raw is None:
                continue
            job = JobRecord.model_validate(raw)
            if self._job_exceeds_image_only_eval_target(job):
                continue
            if self._job_avoids_client(job, client_id):
                if job.assigned_client_id == client_id and job.status in ACTIVE_JOB_STATUSES:
                    metadata = dict(job.spec.metadata or {})
                    resume_argv = metadata.get("resume_argv")
                    if isinstance(resume_argv, list) and resume_argv:
                        job.spec.argv = [str(part) for part in resume_argv]
                    job.status = JobStatus.queued
                    job.assigned_client_id = None
                    job.updated_at = now
                    job.started_at = None
                    job.finished_at = None
                    job.exit_code = None
                    job.pending_commands = []
                    job.message = (
                        f"Recovered after client {client_id} kept reporting an avoided job"
                    )
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                if self._detach_job_from_client_record(state, client_id, job_id):
                    changed = True
                continue
            if not self._client_can_run(client, job, active_jobs=active_jobs):
                if self._client_hard_compatible_with_job(client, job):
                    continue
                if job.assigned_client_id == client_id and job.status in ACTIVE_JOB_STATUSES:
                    metadata = dict(job.spec.metadata or {})
                    resume_argv = metadata.get("resume_argv")
                    if isinstance(resume_argv, list) and resume_argv:
                        job.spec.argv = [str(part) for part in resume_argv]
                    self._mark_job_avoid_client(job, client_id=client_id, now=now)
                    job.status = JobStatus.queued
                    job.assigned_client_id = None
                    job.updated_at = now
                    job.started_at = None
                    job.finished_at = None
                    job.exit_code = None
                    job.pending_commands = []
                    job.message = (
                        f"Recovered after incompatible client {client_id} reported the job"
                    )
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                if self._detach_job_from_client_record(state, client_id, job_id):
                    changed = True
                continue
            reclaiming_recovered_queued_job = self._queued_job_can_be_reclaimed_by_reporting_client(
                job,
                client_id=client_id,
            )
            if job.status not in ACTIVE_JOB_STATUSES and not reclaiming_recovered_queued_job:
                continue
            if job.assigned_client_id == client_id:
                if reclaiming_recovered_queued_job or job.status in {
                    JobStatus.assigned,
                    JobStatus.starting,
                }:
                    job.status = JobStatus.running
                    job.updated_at = now
                    job.started_at = job.started_at or now
                    if self._has_transient_job_message(job):
                        job.message = None
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                elif job.status == JobStatus.running:
                    job.updated_at = now
                    state["jobs"][job_id] = job.model_dump(mode="json")
                    changed = True
                continue
            assigned_client_id = str(job.assigned_client_id or "").strip()
            if (
                assigned_client_id
                and assigned_client_id in known_client_ids
                and not self._client_replacement_allowed(
                    state,
                    previous_client_id=assigned_client_id,
                    replacement_client_id=client_id,
                    now=now,
                )
            ):
                # Another client still owns this lease. Do not silently transfer it
                # just because this worker reported the job.
                continue
            if reclaiming_recovered_queued_job:
                job.status = JobStatus.running
                job.started_at = job.started_at or now
                if self._has_transient_job_message(job):
                    job.message = None
            job.assigned_client_id = client_id
            job.updated_at = now
            state["jobs"][job_id] = job.model_dump(mode="json")
            if assigned_client_id and assigned_client_id != client_id:
                self._detach_job_from_client_record(state, assigned_client_id, job_id)
            changed = True
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if job.assigned_client_id != client_id:
                continue
            if job.status == JobStatus.paused:
                job.assigned_client_id = None
                job.updated_at = now
                state["jobs"][job_id] = job.model_dump(mode="json")
                changed = True
                continue
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            if job_id in reported_current_jobs:
                continue
            # Newly assigned or just-starting work can legitimately miss one
            # heartbeat before the client updates current_jobs. Give it a short
            # grace window so restarts and burst dispatches do not immediately
            # reclaim healthy fresh assignments.
            if (now - job.updated_at).total_seconds() < JOB_REPORT_GRACE_SECONDS:
                continue
            client_last_activity = client.last_poll or client.last_heartbeat
            if (
                client.status == ClientStatus.online
                and client_last_activity is not None
                and (now - client_last_activity).total_seconds() < JOB_MISSING_REPORT_REQUEUE_SECONDS
            ):
                continue
            metadata = dict(job.spec.metadata or {})
            resume_argv = metadata.get("resume_argv")
            if isinstance(resume_argv, list) and resume_argv:
                job.spec.argv = [str(part) for part in resume_argv]
            self._mark_job_avoid_client(job, client_id=client_id, now=now)
            job.status = JobStatus.queued
            job.assigned_client_id = None
            job.updated_at = now
            job.started_at = None
            job.finished_at = None
            job.exit_code = None
            job.pending_commands = []
            job.message = (
                f"Recovered after client {client_id} stopped reporting the job"
            )
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _pending_assignments_for_client(
        self,
        state: dict[str, Any],
        client_id: str,
        *,
        reported_current_jobs: set[str],
    ) -> list[JobRecord]:
        pending: list[JobRecord] = []
        for raw_job in state["jobs"].values():
            job = JobRecord.model_validate(raw_job)
            if job.assigned_client_id != client_id:
                continue
            if job.status != JobStatus.assigned:
                continue
            if job.job_id in reported_current_jobs:
                continue
            pending.append(job)
        pending.sort(key=lambda job: self._dispatch_sort_key(job))
        return pending

    def _split_reported_current_jobs(
        self,
        state: dict[str, Any],
        client_id: str,
        *,
        reported_current_jobs: set[str],
    ) -> tuple[set[str], set[str]]:
        valid: set[str] = set()
        stale: set[str] = set()
        known_client_ids = set(state["clients"])
        now = _utc_now()
        for job_id in reported_current_jobs:
            raw = state["jobs"].get(job_id)
            if raw is None:
                stale.add(job_id)
                continue
            job = JobRecord.model_validate(raw)
            if self._job_exceeds_image_only_eval_target(job):
                stale.add(job_id)
                continue
            if job.status not in ACTIVE_JOB_STATUSES and not self._queued_job_can_be_reclaimed_by_reporting_client(
                job,
                client_id=client_id,
            ):
                stale.add(job_id)
                continue
            if not self._job_counts_toward_slot_limit(job):
                stale.add(job_id)
                continue
            assigned_client_id = str(job.assigned_client_id or "").strip()
            if (
                assigned_client_id
                and assigned_client_id != client_id
                and assigned_client_id in known_client_ids
                and not self._client_replacement_allowed(
                    state,
                    previous_client_id=assigned_client_id,
                    replacement_client_id=client_id,
                    now=now,
                )
            ):
                stale.add(job_id)
                continue
            valid.add(job_id)
        return valid, stale

    def _queue_stale_job_commands(
        self,
        state: dict[str, Any],
        client_id: str,
        *,
        stale_job_ids: set[str],
    ) -> bool:
        changed = False
        for job_id in stale_job_ids:
            raw = state["jobs"].get(job_id)
            if raw is None:
                continue
            job = JobRecord.model_validate(raw)
            self._append_command(job, JobCommandType.requeue)
            state["jobs"][job_id] = job.model_dump(mode="json")
            changed = True
        return changed

    def _prune_redundant_requeue_commands(self, state: dict[str, Any]) -> bool:
        changed = False
        client_current_jobs = {
            client_id: set(
                ClientRecord.model_validate(raw_client).current_jobs or []
            )
            for client_id, raw_client in state.get("clients", {}).items()
        }
        for job_id, raw_job in state["jobs"].items():
            job = JobRecord.model_validate(raw_job)
            if not job.pending_commands:
                continue
            keep: list[JobCommand] = []
            assigned_client_id = str(job.assigned_client_id or "").strip()
            assigned_current_jobs = (
                client_current_jobs.get(assigned_client_id, set())
                if assigned_client_id
                else set()
            )
            for command in job.pending_commands:
                if command.command != JobCommandType.requeue:
                    keep.append(command)
                    continue
                if job.status not in ACTIVE_JOB_STATUSES:
                    changed = True
                    continue
                if assigned_client_id and job.job_id in assigned_current_jobs:
                    if job.status == JobStatus.stopping:
                        keep.append(command)
                        continue
                    changed = True
                    continue
                keep.append(command)
            if len(keep) != len(job.pending_commands):
                job.pending_commands = keep
                job.updated_at = _utc_now()
                state["jobs"][job_id] = job.model_dump(mode="json")
        return changed

    def _commands_for_reported_job_mismatches(
        self,
        state: dict[str, Any],
        client_id: str,
        *,
        reported_current_jobs: set[str],
    ) -> list[JobCommand]:
        commands: list[JobCommand] = []
        now = _utc_now()
        for job_id in sorted(reported_current_jobs):
            raw = state["jobs"].get(job_id)
            if raw is None:
                commands.append(
                    JobCommand(
                        command_id=uuid.uuid4().hex,
                        job_id=job_id,
                        command=JobCommandType.requeue,
                        created_at=now,
                    )
                )
                continue
            job = JobRecord.model_validate(raw)
            if (
                job.status == JobStatus.stopping
                or any(
                    command.command == JobCommandType.requeue
                    for command in job.pending_commands
                )
            ):
                commands.append(
                    JobCommand(
                        command_id=uuid.uuid4().hex,
                        job_id=job_id,
                        command=JobCommandType.requeue,
                        created_at=now,
                    )
                )
                continue
            if (
                job.status in ACTIVE_JOB_STATUSES
                and job.assigned_client_id == client_id
            ):
                continue
            commands.append(
                JobCommand(
                    command_id=uuid.uuid4().hex,
                    job_id=job_id,
                    command=JobCommandType.requeue,
                    created_at=now,
                )
            )
        return commands

    def _next_queue_order(self, state: dict[str, Any]) -> int:
        max_seen = 0
        for raw_job in state["jobs"].values():
            metadata = (
                JobRecord.model_validate(raw_job).spec.metadata
                if raw_job is not None
                else {}
            )
            try:
                max_seen = max(max_seen, int(metadata.get("queue_order") or 0))
            except (TypeError, ValueError):
                continue
        return max_seen + 1

    def _trim_oversized_job_state(self, state: dict[str, Any]) -> bool:
        changed = False
        for job_id, raw_job in list(state.get("jobs", {}).items()):
            job_changed = False
            job = JobRecord.model_validate(raw_job)
            metadata = dict(job.spec.metadata or {})
            uploaded_artifacts = metadata.get("uploaded_artifacts")
            if isinstance(uploaded_artifacts, list):
                metadata["uploaded_artifact_count"] = len(uploaded_artifacts)
                if uploaded_artifacts:
                    metadata["uploaded_artifacts_preview"] = uploaded_artifacts[
                        :MAX_STORED_UPLOADED_ARTIFACT_PREVIEW
                    ]
                metadata.pop("uploaded_artifacts", None)
                job.spec.metadata = metadata
                changed = True
                job_changed = True
            compacted_relpaths = self._compact_artifact_relpaths(job.artifact_relpaths)
            if compacted_relpaths != job.artifact_relpaths:
                metadata = dict(job.spec.metadata or {})
                metadata["artifact_relpath_count"] = len(job.artifact_relpaths)
                job.spec.metadata = metadata
                job.artifact_relpaths = compacted_relpaths
                changed = True
                job_changed = True
            if job_changed:
                state["jobs"][job_id] = job.model_dump(mode="json")
        return changed

    def _compact_artifact_relpaths(self, relpaths: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in relpaths:
            rel = str(raw or "").strip().lstrip("/")
            if not rel:
                continue
            path = Path(rel)
            parts = path.parts
            compact = rel
            if "library_evaluations" in parts:
                index = parts.index("library_evaluations")
                compact = Path(*parts[: index + 2]).as_posix()
            elif "deepphy_eval" in parts:
                index = parts.index("deepphy_eval")
                compact = Path(*parts[: index + 2]).as_posix()
            elif "reward_program_experiments" in parts:
                index = parts.index("reward_program_experiments")
                compact = Path(*parts[: min(len(parts), index + 3)]).as_posix()
            elif len(parts) > 7:
                compact = Path(*parts[:7]).as_posix()
            if compact not in seen:
                cleaned.append(compact)
                seen.add(compact)
            if len(cleaned) >= MAX_STORED_ARTIFACT_RELPATHS:
                break
        return cleaned

    def _write(self, payload: dict[str, Any]) -> None:
        self._trim_oversized_job_state(payload)
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        tmp_path = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(encoded, encoding="utf-8")
            tmp_path.replace(self.path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        self._state_payload_cache = copy.deepcopy(payload)
        self._state_payload_mtime_ns = self._current_state_file_mtime_ns()
        self._refresh_snapshot(payload)
