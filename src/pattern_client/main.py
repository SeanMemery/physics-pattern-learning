from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

from public_connection_client_agent import ClientAgent, _default_execution_root
from public_connection_models import ClientKind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        default=os.environ.get("PATTERN_SERVER_URL", ""),
        help="Pattern server base URL. Defaults to PATTERN_SERVER_URL if set.",
    )
    parser.add_argument(
        "--server-api-key",
        default=os.environ.get("PATTERN_SERVER_API_KEY", ""),
        help="Pattern server API key. Defaults to PATTERN_SERVER_API_KEY if set.",
    )
    parser.add_argument(
        "--mode",
        choices=[item.value for item in ClientKind],
        default=os.environ.get("PATTERN_CLIENT_MODE", ClientKind.remote.value),
        help="Client mode. Defaults to remote; direct exposes a URL and accepts server-pushed jobs.",
    )
    parser.add_argument(
        "--display-name",
        default=os.environ.get("PATTERN_CLIENT_DISPLAY_NAME", ""),
        help="Optional client display name. Defaults to remote-client or direct-client by mode.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LLM_ROUTER_BASE_URL", ""),
        help="LLM base URL override for launched jobs. Defaults to LLM_ROUTER_BASE_URL if set.",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("LLM_ROUTER_API_KEY", ""),
        help="LLM router API key override for launched jobs. Defaults to LLM_ROUTER_API_KEY if set.",
    )
    parser.add_argument(
        "--client-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument("--execution-root", default=str(_default_execution_root()))
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("PATTERN_CLIENT_MAX_CONCURRENCY", "0") or "0"),
        help="Optional client slot override. Useful for local direct clients.",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--listen-host",
        default=os.environ.get("PATTERN_CLIENT_LISTEN_HOST", "127.0.0.1"),
        help="Direct-mode listen host.",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=int(os.environ.get("PATTERN_CLIENT_LISTEN_PORT", "8792")),
        help="Direct-mode listen port.",
    )
    parser.add_argument(
        "--direct-url",
        default=os.environ.get("PATTERN_CLIENT_DIRECT_URL", ""),
        help="Explicit direct-mode URL to register with the server. Defaults to http://<listen-host>:<listen-port>.",
    )
    parser.add_argument(
        "--reconnect-pause-seconds",
        type=float,
        default=float(os.environ.get("JOB_SERVER_RECONNECT_PAUSE_SECONDS", "15")),
        help=(
            "Pause after unexpected client control-loop failures. "
            "Defaults to JOB_SERVER_RECONNECT_PAUSE_SECONDS or 15."
        ),
    )
    parser.add_argument(
        "--transport-retry-seconds",
        type=float,
        default=float(os.environ.get("JOB_SERVER_TRANSPORT_RETRY_SECONDS", "15")),
        help=(
            "Pause after transient transport failures such as TLS EOFs. "
            "Defaults to JOB_SERVER_TRANSPORT_RETRY_SECONDS or 15."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.server_url:
        parser.error(
            "server URL is required via --server-url or PATTERN_SERVER_URL"
        )

    mode = ClientKind(args.mode)
    display_name = str(args.display_name or "").strip() or (
        "direct-client" if mode == ClientKind.direct else "remote-client"
    )

    agent = ClientAgent(
        args.server_url,
        server_api_key=args.server_api_key,
        client_root=Path(args.client_root),
        execution_root=Path(args.execution_root),
        display_name=display_name,
        client_kind=mode,
        direct_url=(
            args.direct_url.strip()
            or (
                f"http://{args.listen_host}:{int(args.listen_port)}"
                if mode == ClientKind.direct
                else ""
            )
        ),
        max_concurrency_override=(args.max_concurrency if int(args.max_concurrency or 0) > 0 else None),
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
    )
    if mode == ClientKind.direct:
        try:
            from public_connection_direct_app import create_direct_app
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("fastapi and uvicorn are required for direct mode") from exc
        agent.register()
        heartbeat_thread = threading.Thread(
            target=agent.run_heartbeat_forever,
            kwargs={
                "heartbeat_interval_seconds": args.poll_interval_seconds,
                "reconnect_pause_seconds": args.reconnect_pause_seconds,
                "transport_retry_seconds": args.transport_retry_seconds,
            },
            daemon=True,
        )
        heartbeat_thread.start()
        uvicorn.run(
            create_direct_app(agent),
            host=args.listen_host,
            port=int(args.listen_port),
            log_level="info",
        )
        return
    agent.run_forever(
        poll_interval_seconds=args.poll_interval_seconds,
        reconnect_pause_seconds=args.reconnect_pause_seconds,
        transport_retry_seconds=args.transport_retry_seconds,
    )


if __name__ == "__main__":
    main()
