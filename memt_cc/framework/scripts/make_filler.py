"""Dump plain text from long documents to use as distractor text in the synthetic dialogues.

    python scripts/make_filler.py --data /path/to/data-00000-of-01026.arrow --out data/generated/filler.txt --mb 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lact_qwen.text_data import iter_documents


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="same spec as eval_position_loss.py --data")
    p.add_argument("--out", required=True)
    p.add_argument("--mb", type=float, default=50.0, help="stop after this many megabytes of text")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    limit, written = int(args.mb * 1e6), 0
    with open(args.out, "w") as f:
        for text in iter_documents(args.data):
            text = " ".join(text.split())
            f.write(text + "\n")
            written += len(text) + 1
            if written >= limit:
                break
    print(f"wrote {written / 1e6:.1f} MB to {args.out}")


if __name__ == "__main__":
    main()
