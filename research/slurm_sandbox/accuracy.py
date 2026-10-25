#!/usr/bin/env python3
"""Accuracy for the Slurm sandbox evals: strict, lenient, Wilson 95%."""
import json
import math
from pathlib import Path

Z = 1.959964


def wilson(correct, total):
    p = correct / total
    d = 1 + Z * Z / total
    c = (p + Z * Z / (2 * total)) / d
    h = Z / d * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total))
    return p, c - h, c + h


def row(name, correct, partly, total, fmt=None):
    strict, lo, hi = wilson(correct, total)
    lenient = (correct + 0.5 * partly) / total
    fmt_cell = f"{fmt[0]}/{fmt[1]}" if fmt else "n/a"
    print(f"| {name} | {total} | {100 * strict:.1f}% | {100 * lenient:.1f}% | "
          f"[{100 * lo:.1f}%, {100 * hi:.1f}%] | {fmt_cell} |")


def main():
    data = json.loads(
        (Path(__file__).parent / "results/2026-10-25/results.json").read_text())
    results = data["results"]
    correct = sum(r["rating"] == "correct" for r in results)
    partly = sum(r["rating"] == "partly correct" for r in results)
    print("| model / round | n | strict | lenient | Wilson 95% (strict) | format |")
    print("|---|---|---|---|---|---|")
    row("round 2, qwen3-4b", correct, partly, len(results))
    row("round 1, qwen3-4b", 3, 0, 4, (4, 4))
    row("round 1, hpc-assistant-x02", 1, 0, 4, (0, 4))
    print()
    print("Round 2 per item (each item is one task type):")
    print("| id | task | rating |")
    print("|---|---|---|")
    for r in results:
        print(f"| {r['id']} | {r['task']} | {r['rating']} |")


if __name__ == "__main__":
    main()
