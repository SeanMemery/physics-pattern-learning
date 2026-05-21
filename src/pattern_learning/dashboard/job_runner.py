from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--cwd")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("dashboard.job_runner requires a command after --")
    return args


def _drain_stream(
    stream: TextIO | None,
    *,
    target_path: Path,
    combined_path: Path,
    label: str,
) -> None:
    if stream is None:
        return
    with (
        target_path.open("a", encoding="utf-8") as target,
        combined_path.open("a", encoding="utf-8") as combined,
    ):
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            target.write(raw_line)
            combined.write(f"[{label}] {line}\n")
            target.flush()
            combined.flush()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_proc_status_value(pid: int, key: str) -> int | None:
    status_path = Path(f"/proc/{pid}/status")
    try:
        text = status_path.read_text(encoding="utf-8")
    except OSError:
        return None
    prefix = f"{key}:"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _process_metrics(pid: int) -> dict[str, int | str | None]:
    return {
        "pid": pid,
        "rss_kb": _read_proc_status_value(pid, "VmRSS"),
        "hwm_kb": _read_proc_status_value(pid, "VmHWM"),
        "threads": _read_proc_status_value(pid, "Threads"),
    }


def _system_memory_metrics() -> dict[str, int] | None:
    meminfo = Path("/proc/meminfo")
    try:
        text = meminfo.read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    payload = {
        "mem_total_kb": values["MemTotal"],
        "mem_available_kb": values["MemAvailable"],
    }
    if "SwapTotal" in values:
        payload["swap_total_kb"] = values["SwapTotal"]
    if "SwapFree" in values:
        payload["swap_free_kb"] = values["SwapFree"]
    return payload


def _temperature_metrics() -> list[dict[str, int | str]] | None:
    thermal_root = Path("/sys/class/thermal")
    if not thermal_root.is_dir():
        return None
    readings: list[dict[str, int | str]] = []
    for zone in sorted(thermal_root.glob("thermal_zone*")):
        temp_path = zone / "temp"
        if not temp_path.is_file():
            continue
        try:
            raw_temp = temp_path.read_text(encoding="utf-8").strip()
            temp_millic = int(raw_temp)
        except (OSError, ValueError):
            continue
        zone_type = "unknown"
        type_path = zone / "type"
        try:
            zone_type = type_path.read_text(encoding="utf-8").strip() or zone.name
        except OSError:
            zone_type = zone.name
        readings.append(
            {
                "source": zone.name,
                "label": zone_type,
                "temp_c": round(temp_millic / 1000.0, 1),
            }
        )
    return readings or None


def _gpu_metrics() -> list[dict[str, int | str | float]] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    readings: list[dict[str, int | str | float]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            index = int(parts[0])
            memory_used_mb = int(parts[2])
            memory_total_mb = int(parts[3])
            temperature_c = int(parts[4])
            utilization_gpu = int(parts[5])
        except ValueError:
            continue
        readings.append(
            {
                "index": index,
                "name": parts[1],
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "temperature_c": temperature_c,
                "utilization_gpu": utilization_gpu,
            }
        )
    return readings or None


def _sample_resource_metrics(pid: int) -> dict[str, object]:
    return {
        "timestamp": _utc_now(),
        "process": _process_metrics(pid),
        "system_memory": _system_memory_metrics(),
        "temperatures": _temperature_metrics(),
        "gpus": _gpu_metrics(),
    }


def _capture_failure_resource_snapshot(pid: int) -> dict[str, object]:
    return {
        "captured_at": _utc_now(),
        "process": _process_metrics(pid),
        "system_memory": _system_memory_metrics(),
        "temperatures": _temperature_metrics(),
        "gpus": _gpu_metrics(),
    }


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _build_metrics_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "sample_count": len(samples),
    }
    process_rss_values: list[int] = []
    process_hwm_values: list[int] = []
    mem_available_values: list[int] = []
    gpu_memory_by_index: dict[int, int] = {}
    gpu_temp_by_index: dict[int, int] = {}
    temperature_by_label: dict[str, float] = {}
    for sample in samples:
        process = sample.get("process")
        if isinstance(process, dict):
            rss_kb = process.get("rss_kb")
            hwm_kb = process.get("hwm_kb")
            if isinstance(rss_kb, int):
                process_rss_values.append(rss_kb)
            if isinstance(hwm_kb, int):
                process_hwm_values.append(hwm_kb)
        system_memory = sample.get("system_memory")
        if isinstance(system_memory, dict):
            mem_available_kb = system_memory.get("mem_available_kb")
            if isinstance(mem_available_kb, int):
                mem_available_values.append(mem_available_kb)
        gpus = sample.get("gpus")
        if isinstance(gpus, list):
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                index = gpu.get("index")
                memory_used_mb = gpu.get("memory_used_mb")
                temperature_c = gpu.get("temperature_c")
                if isinstance(index, int) and isinstance(memory_used_mb, int):
                    gpu_memory_by_index[index] = max(
                        gpu_memory_by_index.get(index, 0),
                        memory_used_mb,
                    )
                if isinstance(index, int) and isinstance(temperature_c, int):
                    gpu_temp_by_index[index] = max(
                        gpu_temp_by_index.get(index, 0),
                        temperature_c,
                    )
        temperatures = sample.get("temperatures")
        if isinstance(temperatures, list):
            for item in temperatures:
                if not isinstance(item, dict):
                    continue
                label = item.get("label")
                temp_c = item.get("temp_c")
                if isinstance(label, str) and isinstance(temp_c, (int, float)):
                    temperature_by_label[label] = max(
                        temperature_by_label.get(label, float("-inf")),
                        float(temp_c),
                    )
    if process_rss_values:
        summary["max_process_rss_kb"] = max(process_rss_values)
    if process_hwm_values:
        summary["max_process_hwm_kb"] = max(process_hwm_values)
    if mem_available_values:
        summary["min_system_mem_available_kb"] = min(mem_available_values)
    if gpu_memory_by_index:
        summary["max_gpu_memory_used_mb"] = gpu_memory_by_index
    if gpu_temp_by_index:
        summary["max_gpu_temperature_c"] = gpu_temp_by_index
    if temperature_by_label:
        summary["max_temperature_c_by_label"] = temperature_by_label
    return summary


def _resource_monitor(
    *,
    child: subprocess.Popen[str],
    metrics_path: Path,
    summary_path: Path,
    combined_path: Path,
    sample_interval_seconds: float,
    stop_event: threading.Event,
) -> None:
    samples: list[dict[str, object]] = []
    while not stop_event.is_set():
        if child.poll() is not None:
            break
        sample = _sample_resource_metrics(child.pid)
        samples.append(sample)
        _append_jsonl(metrics_path, sample)
        stop_event.wait(sample_interval_seconds)
    final_sample = _sample_resource_metrics(child.pid)
    samples.append(final_sample)
    _append_jsonl(metrics_path, final_sample)
    summary = _build_metrics_summary(samples)
    summary["completed_at"] = _utc_now()
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with combined_path.open("a", encoding="utf-8") as combined:
        combined.write(f"[dashboard] resource_summary {json.dumps(summary, sort_keys=True)}\n")
        combined.flush()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    job_root = Path(args.job_root)
    job_root.mkdir(parents=True, exist_ok=True)
    stdout_path = job_root / "stdout.log"
    stderr_path = job_root / "stderr.log"
    log_path = job_root / "job.log"
    exit_code_path = job_root / "exit_code.txt"
    metrics_path = job_root / "system_metrics.jsonl"
    metrics_summary_path = job_root / "system_metrics_summary.json"
    failure_snapshot_path = job_root / "failed_resource_snapshot.json"

    child = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        cwd=args.cwd or None,
    )

    def forward_termination(signum: int, _frame: object) -> None:
        try:
            child.terminate()
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward_termination)
    signal.signal(signal.SIGINT, forward_termination)

    stdout_thread = threading.Thread(
        target=_drain_stream,
        kwargs={
            "stream": child.stdout,
            "target_path": stdout_path,
            "combined_path": log_path,
            "label": "stdout",
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_stream,
        kwargs={
            "stream": child.stderr,
            "target_path": stderr_path,
            "combined_path": log_path,
            "label": "stderr",
        },
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    metrics_stop_event = threading.Event()
    metrics_thread = threading.Thread(
        target=_resource_monitor,
        kwargs={
            "child": child,
            "metrics_path": metrics_path,
            "summary_path": metrics_summary_path,
            "combined_path": log_path,
            "sample_interval_seconds": 5.0,
            "stop_event": metrics_stop_event,
        },
        daemon=True,
    )
    metrics_thread.start()
    exit_code = child.wait()
    failure_snapshot: dict[str, object] | None = None
    if exit_code != 0:
        failure_snapshot = _capture_failure_resource_snapshot(child.pid)
    metrics_stop_event.set()
    stdout_thread.join(timeout=5.0)
    stderr_thread.join(timeout=5.0)
    metrics_thread.join(timeout=6.0)
    exit_code_path.write_text(f"{exit_code}\n", encoding="utf-8")
    if failure_snapshot is not None:
        failure_snapshot_path.write_text(
            json.dumps(failure_snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8") as combined:
            combined.write(
                "[dashboard] failure_resource_snapshot "
                f"{json.dumps(failure_snapshot, sort_keys=True)}\n"
            )
            combined.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
