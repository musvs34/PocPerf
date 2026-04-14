from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psutil


DEFAULT_CONFIG = {
    "artifact_root": "benchmark_outputs",
    "sample_interval_s": 0.50,
    "random_seed": 42,
    "cleanup_temp_files": True,
    "cpu_matrix_size": 1200,
    "cpu_repeats": 4,
    "cpu_hash_buffer_mb": 128,
    "cpu_hash_cycles": 12,
    "memory_array_gb": 0.25,
    "memory_cycles": 6,
    "memory_copy_array_gb": 0.35,
    "memory_copy_cycles": 8,
    "disk_file_size_mb": 1024,
    "disk_chunk_mb": 16,
    "small_file_count": 128,
    "small_file_size_kb": 512,
    "combined_cpu_matrix_size": 900,
    "combined_cpu_repeats": 3,
    "combined_memory_gb": 0.15,
    "combined_memory_cycles": 4,
}

def sanitize_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value.strip())
    return cleaned.strip("-_") or "server"


def normalize_for_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def persist_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalize_for_json(payload), handle, indent=2)


def safe_unlink(path: Path, retries: int = 10, delay_s: float = 0.2) -> bool:
    path = Path(path)
    if not path.exists():
        return True

    last_error = None
    for _ in range(retries):
        try:
            path.unlink()
            return True
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_s)

    if path.exists() and last_error is not None:
        print(f"Warning: temporary file still locked: {path} ({last_error})")
    return not path.exists()


def safe_rmtree(path: Path, retries: int = 10, delay_s: float = 0.2) -> bool:
    path = Path(path)
    if not path.exists():
        return True

    last_error = None
    for _ in range(retries):
        try:
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()
            path.rmdir()
            return True
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_s)

    if path.exists() and last_error is not None:
        print(f"Warning: temporary folder still locked: {path} ({last_error})")
    return not path.exists()


def build_env_info(artifact_dir: Path, run_id: str, server_name: str) -> dict:
    disk_usage = None
    try:
        anchor = artifact_dir.resolve().anchor or str(Path.cwd())
        disk_usage = psutil.disk_usage(anchor)
    except Exception:
        disk_usage = None

    return {
        "run_id": run_id,
        "server_name": server_name,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "artifact_dir": str(artifact_dir.resolve()),
        "disk_free_gb": round(disk_usage.free / (1024 ** 3), 2) if disk_usage else None,
    }


class MetricSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._t0 = None

    def capture(self) -> None:
        memory = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        self.samples.append(
            {
                "t_s": time.perf_counter() - self._t0,
                "system_cpu_pct": psutil.cpu_percent(interval=None),
                "process_cpu_pct": self.process.cpu_percent(interval=None),
                "process_rss_gb": self.process.memory_info().rss / (1024 ** 3),
                "available_memory_gb": memory.available / (1024 ** 3),
                "used_memory_pct": memory.percent,
                "system_disk_read_mb": (disk.read_bytes / (1024 ** 2)) if disk else np.nan,
                "system_disk_write_mb": (disk.write_bytes / (1024 ** 2)) if disk else np.nan,
            }
        )

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.capture()

    def start(self) -> None:
        self._t0 = time.perf_counter()
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        self.capture()
        self._thread.start()

    def stop(self) -> pd.DataFrame:
        self.capture()
        self._stop_event.set()
        self._thread.join(timeout=self.interval_s * 2)
        return pd.DataFrame(self.samples)


def summarize_samples(samples_df: pd.DataFrame) -> dict:
    if samples_df.empty:
        return {}

    summary = {
        "sample_count": int(len(samples_df)),
        "avg_system_cpu_pct": samples_df["system_cpu_pct"].mean(),
        "peak_system_cpu_pct": samples_df["system_cpu_pct"].max(),
        "avg_process_cpu_pct": samples_df["process_cpu_pct"].mean(),
        "peak_process_cpu_pct": samples_df["process_cpu_pct"].max(),
        "avg_process_rss_gb": samples_df["process_rss_gb"].mean(),
        "peak_process_rss_gb": samples_df["process_rss_gb"].max(),
        "min_available_memory_gb": samples_df["available_memory_gb"].min(),
        "max_used_memory_pct": samples_df["used_memory_pct"].max(),
    }
    if samples_df["system_disk_read_mb"].notna().all():
        summary["system_disk_read_delta_mb"] = (
            samples_df["system_disk_read_mb"].iloc[-1] - samples_df["system_disk_read_mb"].iloc[0]
        )
    if samples_df["system_disk_write_mb"].notna().all():
        summary["system_disk_write_delta_mb"] = (
            samples_df["system_disk_write_mb"].iloc[-1] - samples_df["system_disk_write_mb"].iloc[0]
        )
    return summary


def run_with_monitor(test_name: str, workload, config: dict) -> dict:
    sampler = MetricSampler(interval_s=config["sample_interval_s"])
    start_ts = datetime.now().isoformat(timespec="seconds")
    sampler.start()
    wall_start = time.perf_counter()
    try:
        workload_metrics = workload(config)
    finally:
        wall_elapsed = time.perf_counter() - wall_start
        samples_df = sampler.stop()

    summary = {
        "test_name": test_name,
        "started_at": start_ts,
        "elapsed_s": wall_elapsed,
    }
    summary.update(summarize_samples(samples_df))
    summary.update(workload_metrics)
    return {
        "summary": summary,
        "samples": samples_df,
    }


def write_file_benchmark(target_path: Path, file_size_mb: int, chunk_mb: int) -> dict:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = int(file_size_mb * 1024 ** 2)
    chunk_bytes = int(chunk_mb * 1024 ** 2)
    block = os.urandom(chunk_bytes)
    remaining = total_bytes
    start = time.perf_counter()
    with target_path.open("wb", buffering=0) as handle:
        while remaining > 0:
            payload = block if remaining >= chunk_bytes else block[:remaining]
            written = handle.write(payload)
            remaining -= written
        handle.flush()
        os.fsync(handle.fileno())
    elapsed = time.perf_counter() - start
    return {
        "disk_write_elapsed_s": elapsed,
        "disk_write_mb_s": total_bytes / elapsed / (1024 ** 2),
        "disk_write_size_mb": file_size_mb,
        "disk_write_path": str(target_path.resolve()),
    }


def ensure_seed_file(seed_path: Path, file_size_mb: int, chunk_mb: int) -> Path:
    seed_path = Path(seed_path)
    expected_size = int(file_size_mb * 1024 ** 2)
    if seed_path.exists() and seed_path.stat().st_size == expected_size:
        return seed_path
    _ = write_file_benchmark(seed_path, file_size_mb=file_size_mb, chunk_mb=chunk_mb)
    return seed_path


def read_file_benchmark(source_path: Path, chunk_mb: int) -> dict:
    source_path = Path(source_path)
    chunk_bytes = int(chunk_mb * 1024 ** 2)
    total_bytes = source_path.stat().st_size
    consumed_bytes = 0
    checksum = 0
    start = time.perf_counter()
    with source_path.open("rb", buffering=0) as handle:
        while True:
            payload = handle.read(chunk_bytes)
            if not payload:
                break
            consumed_bytes += len(payload)
            checksum ^= payload[0]
    elapsed = time.perf_counter() - start
    return {
        "disk_read_elapsed_s": elapsed,
        "disk_read_mb_s": consumed_bytes / elapsed / (1024 ** 2),
        "disk_read_size_mb": total_bytes / (1024 ** 2),
        "disk_read_checksum": checksum,
        "disk_read_path": str(source_path.resolve()),
    }


def write_small_files_benchmark(target_dir: Path, file_count: int, file_size_kb: int) -> dict:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = os.urandom(int(file_size_kb * 1024))
    start = time.perf_counter()
    for index in range(file_count):
        current_file = target_dir / f"chunk_{index:04d}.bin"
        with current_file.open("wb", buffering=0) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    elapsed = time.perf_counter() - start
    total_mb = file_count * file_size_kb / 1024
    return {
        "small_files_elapsed_s": elapsed,
        "small_file_count": file_count,
        "small_file_size_kb": file_size_kb,
        "small_files_total_mb": total_mb,
        "small_files_per_s": file_count / elapsed,
        "small_files_mb_s": total_mb / elapsed,
        "small_files_path": str(target_dir.resolve()),
    }


def cpu_matrix_kernel(matrix_size: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.random((matrix_size, matrix_size), dtype=np.float32)
    b = rng.random((matrix_size, matrix_size), dtype=np.float32)
    checksum = 0.0
    start = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
        checksum += float(c[0, 0])
        a, b = b, c
    elapsed = time.perf_counter() - start
    estimated_flops = repeats * 2 * (matrix_size ** 3)
    return {
        "cpu_matrix_size": matrix_size,
        "cpu_repeats": repeats,
        "cpu_elapsed_s": elapsed,
        "cpu_gflops": estimated_flops / elapsed / 1e9,
        "cpu_checksum": checksum,
    }


def cpu_hash_kernel(buffer_mb: int, cycles: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, size=int(buffer_mb * 1024 ** 2), dtype=np.uint8).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    start = time.perf_counter()
    for _ in range(cycles):
        digest = hashlib.sha256(payload + digest.encode("ascii")).hexdigest()
    elapsed = time.perf_counter() - start
    total_mb = buffer_mb * cycles
    return {
        "cpu_hash_buffer_mb": buffer_mb,
        "cpu_hash_cycles": cycles,
        "cpu_hash_elapsed_s": elapsed,
        "cpu_hash_mb_s": total_mb / elapsed,
        "cpu_hash_digest_prefix": digest[:16],
    }


def memory_transform_kernel(array_gb: float, cycles: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    target_bytes = int(array_gb * (1024 ** 3))
    element_size = np.dtype(np.float64).itemsize
    element_count = max(1, target_bytes // element_size)
    work = rng.random(element_count)
    checksum = 0.0
    start = time.perf_counter()
    for _ in range(cycles):
        np.multiply(work, 1.000001, out=work)
        np.add(work, 3.14159, out=work)
        np.sqrt(work, out=work)
        checksum += float(work[:: max(1, element_count // 10000)].sum())
    elapsed = time.perf_counter() - start
    estimated_bytes = work.nbytes * 3 * cycles
    return {
        "memory_array_gb": array_gb,
        "memory_cycles": cycles,
        "memory_elapsed_s": elapsed,
        "memory_est_bandwidth_gb_s": estimated_bytes / elapsed / (1024 ** 3),
        "memory_checksum": checksum,
    }


def memory_copy_kernel(array_gb: float, cycles: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    target_bytes = int(array_gb * (1024 ** 3))
    element_size = np.dtype(np.float64).itemsize
    element_count = max(1, target_bytes // element_size)
    source = rng.random(element_count)
    target = np.empty_like(source)
    checksum = 0.0
    start = time.perf_counter()
    for _ in range(cycles):
        np.copyto(target, source)
        np.copyto(source, target)
        checksum += float(target[:: max(1, element_count // 10000)].sum())
    elapsed = time.perf_counter() - start
    moved_bytes = source.nbytes * 2 * cycles
    return {
        "memory_copy_array_gb": array_gb,
        "memory_copy_cycles": cycles,
        "memory_copy_elapsed_s": elapsed,
        "memory_copy_gb_s": moved_bytes / elapsed / (1024 ** 3),
        "memory_copy_checksum": checksum,
    }


def build_workloads(run_id: str, artifact_dir: Path):
    def cpu_matrix_workload(config: dict) -> dict:
        return cpu_matrix_kernel(
            matrix_size=config["cpu_matrix_size"],
            repeats=config["cpu_repeats"],
            seed=config["random_seed"],
        )

    def cpu_hash_workload(config: dict) -> dict:
        return cpu_hash_kernel(
            buffer_mb=config["cpu_hash_buffer_mb"],
            cycles=config["cpu_hash_cycles"],
            seed=config["random_seed"] + 1,
        )

    def memory_transform_workload(config: dict) -> dict:
        return memory_transform_kernel(
            array_gb=config["memory_array_gb"],
            cycles=config["memory_cycles"],
            seed=config["random_seed"] + 2,
        )

    def memory_copy_workload(config: dict) -> dict:
        return memory_copy_kernel(
            array_gb=config["memory_copy_array_gb"],
            cycles=config["memory_copy_cycles"],
            seed=config["random_seed"] + 3,
        )

    def disk_write_workload(config: dict) -> dict:
        target_path = artifact_dir / f"disk_write_{run_id}.bin"
        metrics = write_file_benchmark(
            target_path=target_path,
            file_size_mb=config["disk_file_size_mb"],
            chunk_mb=config["disk_chunk_mb"],
        )
        if config["cleanup_temp_files"]:
            safe_unlink(target_path)
        return metrics

    def disk_small_files_workload(config: dict) -> dict:
        target_dir = artifact_dir / f"small_files_{run_id}"
        metrics = write_small_files_benchmark(
            target_dir=target_dir,
            file_count=config["small_file_count"],
            file_size_kb=config["small_file_size_kb"],
        )
        if config["cleanup_temp_files"]:
            safe_rmtree(target_dir)
        return metrics

    def combined_workload(config: dict) -> dict:
        read_seed_path = artifact_dir / f"combined_read_seed_{config['disk_file_size_mb']}mb.bin"
        ensure_seed_file(
            seed_path=read_seed_path,
            file_size_mb=config["disk_file_size_mb"],
            chunk_mb=config["disk_chunk_mb"],
        )
        write_target_path = artifact_dir / f"combined_write_{run_id}.bin"

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {
                "cpu": executor.submit(
                    cpu_matrix_kernel,
                    config["combined_cpu_matrix_size"],
                    config["combined_cpu_repeats"],
                    config["random_seed"] + 10,
                ),
                "memory": executor.submit(
                    memory_transform_kernel,
                    config["combined_memory_gb"],
                    config["combined_memory_cycles"],
                    config["random_seed"] + 11,
                ),
                "write": executor.submit(
                    write_file_benchmark,
                    write_target_path,
                    config["disk_file_size_mb"],
                    config["disk_chunk_mb"],
                ),
                "small_files": executor.submit(
                    write_small_files_benchmark,
                    artifact_dir / f"combined_small_files_{run_id}",
                    max(16, config["small_file_count"] // 4),
                    config["small_file_size_kb"],
                ),
            }
            results = {name: future.result() for name, future in future_map.items()}
        wall_elapsed = time.perf_counter() - wall_start

        if config["cleanup_temp_files"]:
            safe_unlink(write_target_path)
            safe_rmtree(artifact_dir / f"combined_small_files_{run_id}")

        overlap_factor = (
            results["cpu"]["cpu_elapsed_s"]
            + results["memory"]["memory_elapsed_s"]
            + results["write"]["disk_write_elapsed_s"]
            + results["small_files"]["small_files_elapsed_s"]
        ) / wall_elapsed

        return {
            "combined_wall_s": wall_elapsed,
            "combined_overlap_factor": overlap_factor,
            "combined_cpu_gflops": results["cpu"]["cpu_gflops"],
            "combined_memory_bandwidth_gb_s": results["memory"]["memory_est_bandwidth_gb_s"],
            "combined_write_mb_s": results["write"]["disk_write_mb_s"],
            "combined_small_files_per_s": results["small_files"]["small_files_per_s"],
            "combined_cpu_elapsed_s": results["cpu"]["cpu_elapsed_s"],
            "combined_memory_elapsed_s": results["memory"]["memory_elapsed_s"],
            "combined_write_elapsed_s": results["write"]["disk_write_elapsed_s"],
            "combined_small_files_elapsed_s": results["small_files"]["small_files_elapsed_s"],
        }

    return [
        ("cpu_matrix_only", cpu_matrix_workload),
        ("cpu_hash_only", cpu_hash_workload),
        ("memory_transform_only", memory_transform_workload),
        ("memory_copy_only", memory_copy_workload),
        ("disk_write_only", disk_write_workload),
        ("disk_small_files_only", disk_small_files_workload),
        ("combined_parallel", combined_workload),
    ]


def merge_config(config_overrides: Optional[dict] = None, **kwargs) -> dict:
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    override_values = {key: value for key, value in kwargs.items() if value is not None}
    config.update(override_values)
    return config


def run_benchmark_suite(server_name: Optional[str] = None, config_overrides: Optional[dict] = None, **kwargs) -> dict:
    config = merge_config(config_overrides, **kwargs)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_server_name = sanitize_name(server_name or socket.gethostname())
    artifact_dir = Path(config["artifact_root"]) / f"{resolved_server_name}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env_info = build_env_info(artifact_dir=artifact_dir, run_id=run_id, server_name=resolved_server_name)
    persist_json(artifact_dir / "config.json", config)
    persist_json(artifact_dir / "environment.json", env_info)

    runs = {}
    for test_name, workload in build_workloads(run_id=run_id, artifact_dir=artifact_dir):
        print(f"Running {test_name}...")
        runs[test_name] = run_with_monitor(test_name, workload, config)
        runs[test_name]["samples"].to_csv(artifact_dir / f"{test_name}_timeseries.csv", index=False)
        persist_json(artifact_dir / f"{test_name}_summary.json", runs[test_name]["summary"])

    summary_df = pd.DataFrame([entry["summary"] for entry in runs.values()])
    summary_order = [name for name, _ in build_workloads(run_id=run_id, artifact_dir=artifact_dir)]
    summary_df["test_name"] = pd.Categorical(summary_df["test_name"], categories=summary_order, ordered=True)
    summary_df = summary_df.sort_values("test_name").reset_index(drop=True)

    for key, value in env_info.items():
        summary_df[key] = value

    summary_df.to_csv(artifact_dir / "summary.csv", index=False)
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "environment": env_info,
        "summary": summary_df,
    }


def print_run_overview(summary_df: pd.DataFrame) -> None:
    display_columns = [
        "test_name",
        "elapsed_s",
        "cpu_gflops",
        "cpu_hash_mb_s",
        "memory_est_bandwidth_gb_s",
        "memory_copy_gb_s",
        "disk_write_mb_s",
        "small_files_per_s",
        "combined_overlap_factor",
    ]
    display_columns = [column for column in display_columns if column in summary_df.columns]
    print(summary_df[display_columns].round(2).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark suite for Jupyter environments.")
    parser.add_argument("--server-name", default=None, help="Logical server label used in output folders.")
    parser.add_argument("--artifact-root", default="benchmark_outputs", help="Where artifacts are written.")
    parser.add_argument("--config-json", default=None, help="Optional JSON file with config overrides.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_overrides = None
    if args.config_json:
        config_overrides = json.loads(Path(args.config_json).read_text(encoding="utf-8"))

    result = run_benchmark_suite(
        server_name=args.server_name,
        config_overrides=config_overrides,
        artifact_root=args.artifact_root,
    )
    print(f"Artifacts written to: {result['artifact_dir']}")
    print_run_overview(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
