"""Print the newest resumable step checkpoint, or nothing if none exists."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lact_qwen.checkpointing import latest_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    result = latest_checkpoint(args.root)
    if result:
        print(result)
