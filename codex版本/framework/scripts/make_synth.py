"""Generate the synthetic retention / correction / recent-write task as jsonl.

Example (three difficulty levels used in the plan):
    python scripts/make_synth.py --out data/generated/synth_8f.jsonl  --n 200 --num-facts 8   --num-turns 16  --words-per-turn 200
    python scripts/make_synth.py --out data/generated/synth_32f.jsonl --n 200 --num-facts 32  --num-turns 48  --words-per-turn 250
    python scripts/make_synth.py --out data/generated/synth_128f.jsonl --n 200 --num-facts 128 --num-turns 160 --words-per-turn 150
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.synth import SynthConfig, generate, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200, help="number of dialogues")
    p.add_argument("--num-turns", type=int, default=24)
    p.add_argument("--num-facts", type=int, default=12)
    p.add_argument("--correction-frac", type=float, default=0.3)
    p.add_argument("--recent-turns", type=int, default=2)
    p.add_argument("--words-per-turn", type=int, default=300)
    p.add_argument("--queries-per-category", type=int, default=4)
    p.add_argument("--filler", choices=["repeat", "file"], default="repeat")
    p.add_argument("--filler-file", default=None, help="plain text used as distractor when --filler file")
    p.add_argument("--value-kind", choices=["digits", "word"], default="digits")
    p.add_argument("--seed", type=int, default=0, help="use different seeds for train and test")
    args = p.parse_args()

    cfg = SynthConfig(
        num_turns=args.num_turns,
        num_facts=args.num_facts,
        correction_frac=args.correction_frac,
        recent_turns=args.recent_turns,
        words_per_turn=args.words_per_turn,
        queries_per_category=args.queries_per_category,
        filler=args.filler,
        filler_file=args.filler_file,
        value_kind=args.value_kind,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = write_jsonl(args.out, generate(cfg, args.n))
    print(f"wrote {n} dialogues to {args.out}")


if __name__ == "__main__":
    main()
