"""Our scores next to MemT-1.0 Table 2.  usage: python report.py <scores.json from judge_bench.py> [--md out.md]

MemT columns are copied from the report's Table 2 (judge gpt-4.1-mini; ours: local Qwen3.6-35B-A3B judge, so the
absolute numbers are not directly comparable; compare our modes with each other).
"""
import argparse
import json

# Table 2 of arXiv 2610.26825: MemT-1.0-Nano, MemT-1.0-Flash, Full Context with the Qwen3.6-35B-A3B backbone (QW)
MEMT = {"locomo": (57.5, 65.1, 83.8), "longmemeval_s": (32.6, 37.2, 76.8), "personamem_32k": (52.3, 61.6, 65.2),
        "mab": (15.5, 17.4, 35.9), "beam_1m": (35.8, 37.0, 43.7)}
NAMES = {"locomo": "LoCoMo", "longmemeval_s": "LongMemEval-S", "personamem_32k": "PersonaMem-32K",
         "mab": "MemoryAgentBench", "beam_1m": "BEAM-1M"}
COLS = [("plain", "memory", "4B memory (plain)"), ("plain", "question", "4B question only"),
        ("plain", "wrong", "4B wrong history"), ("plain", "full", "Qwen3-4B full ctx 32K"),
        ("chat", "memory", "4B memory (chat, MemT format)"), ("chat", "question", "4B question only (chat)"),
        ("nano", "memory", "MemT-Nano memory (ours, HF)"), ("nano", "question", "MemT-Nano empty memory (ours)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    s = json.load(open(a.scores))
    head = ["Benchmark", "MemT-Nano", "MemT-Flash", "QW full ctx"] + [c[2] for c in COLS]
    rows = []
    for b in MEMT:
        cells = [NAMES[b]] + [f"{x:.1f}" for x in MEMT[b]]
        for fmt, mode, _ in COLS:
            r = s.get(f"{b}/{fmt}/{mode}")
            cells.append("-" if r is None or r["score"] is None else f"{r['score']:.1f} (n={r['n']})")
        rows.append(cells)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    print("\n".join(lines))
    if a.md:
        open(a.md, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
