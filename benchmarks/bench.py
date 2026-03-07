"""Benchmark tool for glassbox SVD attention backend vs vanilla vLLM.

Usage:
    python benchmarks/bench.py run                                # all configs
    python benchmarks/bench.py run -c vanilla -c svd_interval64   # specific configs
    python benchmarks/bench.py run --list                         # show configs
    python benchmarks/bench.py run --max-seconds 30 --sweep-size 3

    python benchmarks/bench.py compare benchmarks/results/<timestamp>
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import click

# ── Constants ──────────────────────────────────────────────────────────────

VLLM = "/opt/pytorch/bin/vllm"
GUIDELLM = "/opt/pytorch/bin/guidellm"
DEFAULT_MODEL = "facebook/opt-125m"
DEFAULT_PORT = 8000


@dataclass
class BenchConfig:
    name: str
    backend: str
    enforce_eager: bool
    svd_interval: int | None = None
    svd_rank: int | None = None
    svd_method: str | None = None  # "randomized" or "lanczos"

    @property
    def description(self) -> str:
        parts = [f"backend={self.backend}"]
        if self.enforce_eager:
            parts.append("enforce_eager")
        if self.svd_interval is not None:
            parts.append(f"interval={self.svd_interval}")
            parts.append(f"rank={self.svd_rank}")
        if self.svd_method is not None:
            parts.append(f"method={self.svd_method}")
        return ", ".join(parts)


CONFIGS: dict[str, BenchConfig] = {
    c.name: c
    for c in [
        BenchConfig("vanilla", "TRITON_ATTN", enforce_eager=False),
        BenchConfig("vanilla_eager", "TRITON_ATTN", enforce_eager=True),
        BenchConfig(
            "svd_interval16", "CUSTOM", enforce_eager=True, svd_interval=16, svd_rank=4
        ),
        BenchConfig(
            "svd_interval64", "CUSTOM", enforce_eager=True, svd_interval=64, svd_rank=4
        ),
        BenchConfig(
            "svd_interval256",
            "CUSTOM",
            enforce_eager=True,
            svd_interval=256,
            svd_rank=4,
        ),
        BenchConfig(
            "svd_rank8_interval64",
            "CUSTOM",
            enforce_eager=True,
            svd_interval=64,
            svd_rank=8,
        ),
        BenchConfig(
            "svd_lanczos_interval64",
            "CUSTOM",
            enforce_eager=True,
            svd_interval=64,
            svd_rank=4,
            svd_method="lanczos",
        ),
    ]
}
CONFIG_ORDER = list(CONFIGS.keys())

METRIC_COLUMNS = [
    ("Tput tok/s", "throughput_tok_s", "{:.1f}"),
    ("TTFT p50 ms", "ttft_p50_ms", "{:.1f}"),
    ("TTFT p99 ms", "ttft_p99_ms", "{:.1f}"),
    ("ITL p50 ms", "itl_p50_ms", "{:.2f}"),
    ("ITL p99 ms", "itl_p99_ms", "{:.2f}"),
    ("GPU MiB", "gpu_peak_mib", "{:.0f}"),
]


# ── Shared helpers ─────────────────────────────────────────────────────────


def log(msg: str) -> None:
    click.echo(f"[bench] {msg}")


# ── Run helpers ────────────────────────────────────────────────────────────


def kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL, text=True
        )
    except subprocess.CalledProcessError:
        return  # nothing on the port
    for pid in out.strip().split("\n"):
        if pid:
            os.kill(int(pid), signal.SIGKILL)
            log(f"Killed pid {pid} on port {port}")


def wait_for_server(port: int, timeout: int = 120) -> bool:
    log(f"Waiting for server on port {port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            urlopen(f"http://localhost:{port}/health", timeout=2)
            elapsed = int(time.time() - start)
            log(f"Server ready after {elapsed}s")
            return True
        except Exception:
            time.sleep(2)
    log(f"ERROR: Server failed to start within {timeout}s")
    return False


def _run_one(
    config: BenchConfig,
    outdir: Path,
    model: str,
    request_type: str,
    max_seconds: int,
    sweep_size: int,
    port: int,
    server_timeout: int,
    config_idx: int,
    total_configs: int,
) -> bool:
    click.echo("")
    click.echo("=" * 64)
    log(f"[{config_idx}/{total_configs}] {config.name} ({config.description})")
    click.echo("=" * 64)

    outdir.mkdir(parents=True, exist_ok=True)

    # Write config metadata for `compare` subcommand
    metadata = {"model": model, **asdict(config)}
    (outdir / "config.json").write_text(json.dumps(metadata, indent=2))

    # Set up environment for SVD configs
    env = os.environ.copy()
    env.pop("GLASSBOX_SVD_INTERVAL", None)
    env.pop("GLASSBOX_SVD_RANK", None)
    env.pop("GLASSBOX_SVD_METHOD", None)
    if config.svd_interval is not None:
        env["GLASSBOX_SVD_INTERVAL"] = str(config.svd_interval)
        env["GLASSBOX_SVD_RANK"] = str(config.svd_rank)
    if config.svd_method is not None:
        env["GLASSBOX_SVD_METHOD"] = config.svd_method

    # Build server command
    server_cmd = [
        VLLM,
        "serve",
        model,
        "--attention-backend",
        config.backend,
        "--port",
        str(port),
    ]
    if config.enforce_eager:
        server_cmd.append("--enforce-eager")

    # Start server
    log(f"Starting server: {' '.join(server_cmd)}")
    server_log = open(outdir / "server.log", "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    try:
        if not wait_for_server(port, timeout=server_timeout):
            log("Server log tail:")
            server_log.close()
            click.echo((outdir / "server.log").read_text()[-2000:])
            os.killpg(server_proc.pid, signal.SIGKILL)
            return False

        # Start GPU memory logging
        gpu_log = open(outdir / "gpu_memory.csv", "w")
        gpu_proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,memory.used,memory.total",
                "--format=csv",
                "-l",
                "1",
            ],
            stdout=gpu_log,
            stderr=subprocess.DEVNULL,
        )

        # Run guidellm
        log(f"Running guidellm ({sweep_size} rate levels x {max_seconds}s each)...")

        guidellm_cmd = [
            GUIDELLM,
            "benchmark",
            "run",
            "--target",
            f"http://localhost:{port}",
            "--request-type",
            request_type,
            "--profile",
            "sweep",
            "--rate",
            str(sweep_size),
            "--data",
            "prompt_tokens=128,output_tokens=64",
            "--max-seconds",
            str(max_seconds),
            "--output-dir",
            str(outdir),
            "--outputs",
            "json,csv",
            "--disable-console-interactive",
        ]

        guidellm_log = open(outdir / "guidellm.log", "w")
        guidellm_proc = subprocess.Popen(
            guidellm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        start_time = time.time()
        for line in guidellm_proc.stdout:
            guidellm_log.write(line)
            guidellm_log.flush()

            elapsed = int(time.time() - start_time)
            if "✔" in line and "complete" in line.lower():
                log(line.strip())
            elif line.strip().startswith("✔"):
                log(f"  {line.strip()} [{elapsed}s elapsed]")

        guidellm_proc.wait()
        guidellm_log.close()
        elapsed = int(time.time() - start_time)
        log(f"guidellm finished in {elapsed}s (exit code {guidellm_proc.returncode})")

        # Stop GPU logging
        gpu_proc.kill()
        gpu_log.close()

    finally:
        os.killpg(server_proc.pid, signal.SIGKILL)
        server_proc.wait()
        server_log.close()
        time.sleep(2)

    log(f"Results saved to {outdir}")
    return guidellm_proc.returncode == 0


# ── Compare helpers ────────────────────────────────────────────────────────


def load_guidellm_metrics(result_dir: Path) -> dict | None:
    """Load metrics from guidellm JSON output (v0.5 schema).

    Picks the highest-throughput benchmark from the sweep.
    """
    json_files = list(result_dir.glob("benchmark*.json"))
    if not json_files:
        return None

    with open(json_files[0]) as f:
        data = json.load(f)

    metrics = {}
    try:
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            return None

        best_idx = 0
        best_tp = 0.0
        for i, bench in enumerate(benchmarks):
            m = bench["metrics"]
            tp = m["output_tokens_per_second"]["successful"]["mean"]
            if tp and tp > best_tp:
                best_tp = tp
                best_idx = i

        metrics["throughput_tok_s"] = best_tp

        best = benchmarks[best_idx]["metrics"]
        ttft = best["time_to_first_token_ms"]["successful"]
        itl = best["inter_token_latency_ms"]["successful"]

        metrics["ttft_p50_ms"] = ttft["percentiles"]["p50"]
        metrics["ttft_p99_ms"] = ttft["percentiles"]["p99"]
        metrics["itl_p50_ms"] = itl["percentiles"]["p50"]
        metrics["itl_p99_ms"] = itl["percentiles"]["p99"]

    except (KeyError, IndexError, TypeError) as e:
        click.echo(f"  Warning: Could not parse metrics from {json_files[0]}: {e}")
        return None

    return metrics


def load_gpu_memory(result_dir: Path) -> float | None:
    """Parse peak GPU memory from nvidia-smi CSV log."""
    csv_path = result_dir / "gpu_memory.csv"
    if not csv_path.exists():
        return None

    peak_mib = 0.0
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            mem_str = row[1].strip()
            if "MiB" in mem_str:
                try:
                    val = float(mem_str.replace("MiB", "").strip())
                    peak_mib = max(peak_mib, val)
                except ValueError:
                    continue

    return peak_mib if peak_mib > 0 else None


def load_config_metadata(config_dir: Path) -> dict | None:
    """Load config.json written by the `run` subcommand."""
    p = config_dir / "config.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_result(config_dir: Path) -> dict:
    row = {"name": config_dir.name}
    metrics = load_guidellm_metrics(config_dir)
    if metrics:
        row.update(metrics)
    row["gpu_peak_mib"] = load_gpu_memory(config_dir)
    row["_meta"] = load_config_metadata(config_dir)
    return row


def format_delta(baseline_val: float, test_val: float) -> str:
    if baseline_val == 0:
        return "n/a"
    pct = (test_val - baseline_val) / baseline_val * 100
    return f"{pct:+.1f}%"


def format_config_header(meta: dict | None, name: str) -> str:
    """Format a one-line description from config metadata."""
    if not meta:
        return name
    parts = [name]
    if "model" in meta:
        parts.append(f"model={meta['model']}")
    if "backend" in meta:
        parts.append(f"backend={meta['backend']}")
    if meta.get("enforce_eager"):
        parts.append("enforce_eager")
    if meta.get("svd_interval") is not None:
        parts.append(f"interval={meta['svd_interval']}")
        parts.append(f"rank={meta.get('svd_rank')}")
    if meta.get("svd_method") is not None:
        parts.append(f"method={meta['svd_method']}")
    return "  ".join(parts)


def print_table(rows: list[dict], label_col: str = "label") -> None:
    columns = [(label_col, label_col, "{}")] + METRIC_COLUMNS
    widths = []
    for header, key, fmt in columns:
        w = len(header)
        for row in rows:
            val = row.get(key)
            if val is None:
                w = max(w, 3)
            elif isinstance(val, str):
                w = max(w, len(val))
            else:
                w = max(w, len(fmt.format(val)))
        widths.append(w)

    click.echo(
        " | ".join(h.ljust(w) for (h, _, _), w in zip(columns, widths))
    )
    click.echo("-+-".join("-" * w for w in widths))

    for row in rows:
        cells = []
        for (_, key, fmt), w in zip(columns, widths):
            val = row.get(key)
            if val is None:
                cells.append("n/a".ljust(w))
            elif isinstance(val, str):
                cells.append(val.ljust(w))
            else:
                cells.append(fmt.format(val).ljust(w))
        click.echo(" | ".join(cells))


# ── CLI ────────────────────────────────────────────────────────────────────


@click.group()
def cli():
    """Benchmark tool for glassbox SVD attention backend vs vanilla vLLM."""


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_names",
    multiple=True,
    type=click.Choice(CONFIG_ORDER, case_sensitive=False),
    help="Config(s) to run. Omit for all.",
)
@click.option(
    "--list", "list_configs", is_flag=True, help="List available configs and exit."
)
@click.option(
    "--max-seconds", default=60, show_default=True, help="Max seconds per rate level."
)
@click.option(
    "--sweep-size", default=5, show_default=True, help="Number of rate levels in sweep."
)
@click.option("--port", default=DEFAULT_PORT, show_default=True, help="Server port.")
@click.option(
    "--model", default=DEFAULT_MODEL, show_default=True, help="HuggingFace model name."
)
@click.option(
    "--request-type",
    "request_type",
    default="text_completions",
    show_default=True,
    type=click.Choice(["text_completions", "chat_completions"]),
    help="Request type (use chat_completions for instruct models).",
)
@click.option(
    "--server-timeout",
    default=300,
    show_default=True,
    help="Max seconds to wait for vLLM server startup.",
)
def run(
    config_names: tuple[str, ...],
    list_configs: bool,
    max_seconds: int,
    sweep_size: int,
    port: int,
    model: str,
    request_type: str,
    server_timeout: int,
) -> None:
    """Run guidellm benchmarks for selected configurations."""
    if list_configs:
        for name, cfg in CONFIGS.items():
            click.echo(f"  {name:25s} {cfg.description}")
        return

    selected = (
        [CONFIGS[n] for n in config_names]
        if config_names
        else [CONFIGS[n] for n in CONFIG_ORDER]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = Path("benchmarks/results") / timestamp
    results_base.mkdir(parents=True, exist_ok=True)

    log(f"Results directory: {results_base}")
    log(f"Model: {model}")
    log(f"Configs: {', '.join(c.name for c in selected)}")
    log(f"Sweep: {sweep_size} rate levels x {max_seconds}s each")
    est_minutes = len(selected) * (1 + sweep_size * max_seconds / 60)
    log(f"Estimated total time: ~{est_minutes:.0f} min")
    click.echo("")

    kill_port(port)
    time.sleep(1)

    results: list[tuple[str, bool]] = []
    for i, cfg in enumerate(selected, 1):
        ok = _run_one(
            config=cfg,
            outdir=results_base / cfg.name,
            model=model,
            request_type=request_type,
            max_seconds=max_seconds,
            sweep_size=sweep_size,
            port=port,
            server_timeout=server_timeout,
            config_idx=i,
            total_configs=len(selected),
        )
        results.append((cfg.name, ok))
        kill_port(port)
        time.sleep(1)

    click.echo("")
    click.echo("=" * 64)
    log("All benchmarks complete!")
    log(f"Results: {results_base}")
    for name, ok in results:
        status = click.style("OK", fg="green") if ok else click.style("FAIL", fg="red")
        log(f"  {name:25s} {status}")
    click.echo("=" * 64)


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
def compare(results_dir: str) -> None:
    """Compare benchmark results across configurations."""
    base = Path(results_dir)
    config_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    if not config_dirs:
        click.echo(f"No config directories found in {base}")
        sys.exit(1)

    all_rows = {d.name: load_result(d) for d in config_dirs}

    # Find vanilla baseline
    baseline = all_rows.get("vanilla") or all_rows.get("vanilla_eager")
    if not baseline:
        click.echo("Warning: no vanilla baseline found, showing raw results only")
        print_table([{"label": r["name"], **r} for r in all_rows.values()])
        return

    # Baseline header + table
    click.echo(f"Baseline: {format_config_header(baseline.get('_meta'), baseline['name'])}")
    print_table([{"label": baseline["name"], **baseline}])

    # Per-config comparison
    others = [r for r in all_rows.values() if r["name"] != baseline["name"]]
    for row in others:
        click.echo("")
        click.echo(f"vs {format_config_header(row.get('_meta'), row['name'])}:")
        delta_row = {"label": "delta"}
        for _, key, _ in METRIC_COLUMNS:
            bv = baseline.get(key)
            tv = row.get(key)
            if bv is not None and tv is not None:
                delta_row[key] = format_delta(bv, tv)
            else:
                delta_row[key] = "n/a"
        print_table([{"label": row["name"], **row}, delta_row])


if __name__ == "__main__":
    cli()
