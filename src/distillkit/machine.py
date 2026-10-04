"""What machine produced a result: stored in every benchmark report, and as research/laptop/.

    python -m distillkit.machine [--llama-cpp ~/tools/llama.cpp] [--out research/laptop/<name>]

No hostname or user name is recorded: reports are committed to a public repository.
Every probe is optional; a missing tool (nvidia-smi, ollama, ...) leaves its field empty.
"""

import argparse
import json
import os
import platform
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ["torch", "transformers", "trl", "peft", "bitsandbytes", "openai"]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def parse_lscpu(text: str) -> dict:
    f = {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in text.splitlines())}
    sockets, per_socket = int(f.get("Socket(s)", 1) or 1), int(f.get("Core(s) per socket", 0) or 0)
    return {
        "model": f.get("Model name", ""),
        "physical_cores": sockets * per_socket or None,
        "logical_cpus": int(f["CPU(s)"]) if f.get("CPU(s)", "").isdigit() else os.cpu_count(),
        "max_mhz": float(f["CPU max MHz"]) if f.get("CPU max MHz") else None,
        "l3_cache": f.get("L3 cache", ""),
        "avx2": "avx2" in f.get("Flags", "").split(),
        "avx512": any(x.startswith("avx512") for x in f.get("Flags", "").split()),
    }


def parse_meminfo(text: str) -> dict:
    kb = {m[1]: int(m[2]) for m in re.finditer(r"^(\w+):\s+(\d+) kB", text, re.MULTILINE)}
    return {"ram_gb": round(kb.get("MemTotal", 0) / 1024**2, 1), "swap_gb": round(kb.get("SwapTotal", 0) / 1024**2, 1)}


def parse_os_release(text: str) -> str:
    m = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', text, re.MULTILINE)
    return m.group(1) if m else platform.system()


def collect(llama_cpp: Path | None = None) -> dict:
    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = version(p)
        except PackageNotFoundError:
            pass
    llama_build = ""
    if llama_cpp and (llama_cpp / ".git").exists():
        llama_build = _run(["git", "-C", str(llama_cpp), "describe", "--tags", "--always"])
    return {
        "vendor": _read("/sys/class/dmi/id/sys_vendor"),
        "product": _read("/sys/class/dmi/id/product_name"),
        "cpu": parse_lscpu(_run(["lscpu"])),
        "memory": parse_meminfo(_read("/proc/meminfo")),
        "gpu": gpu.splitlines()[0] if gpu else "",
        "os": parse_os_release(_read("/etc/os-release")),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "packages": pkgs,
        "llama_cpp": llama_build,
        "ollama": _run(["ollama", "--version"]).rsplit(" ", 1)[-1],
    }


def markdown(m: dict) -> str:
    c, mem = m["cpu"], m["memory"]
    rows = [
        ("Machine", f"{m['vendor']} {m['product']}".strip() or "unknown"),
        ("CPU", f"{c['model']}: {c['physical_cores']} cores / {c['logical_cpus']} threads, max {c['max_mhz']:.0f} MHz, L3 {c['l3_cache']}" if c["max_mhz"] else c["model"]),
        ("SIMD", ", ".join(x for x, on in (("AVX2", c["avx2"]), ("AVX-512", c["avx512"])) if on) or "none of AVX2/AVX-512"),
        ("RAM / swap", f"{mem['ram_gb']} GB / {mem['swap_gb']} GB"),
        ("GPU", m["gpu"] or "none (benchmarks are CPU-only anyway)"),
        ("OS", f"{m['os']}, kernel {m['kernel']}"),
        ("Python", m["python"]),
        ("Packages", ", ".join(f"{k} {v}" for k, v in m["packages"].items())),
        ("llama.cpp", m["llama_cpp"] or "unknown"),
        ("Ollama", m["ollama"] or "not installed"),
    ]
    return "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llama-cpp", type=Path, default=Path("~/tools/llama.cpp").expanduser())
    p.add_argument("--out", type=Path, help="write <out>.json and <out>.md")
    a = p.parse_args()
    m = collect(a.llama_cpp)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.with_suffix(".json").write_text(json.dumps(m, indent=2) + "\n")
        a.out.with_suffix(".md").write_text(f"# Benchmark machine\n\n{markdown(m)}")
    print(markdown(m))


if __name__ == "__main__":
    main()
