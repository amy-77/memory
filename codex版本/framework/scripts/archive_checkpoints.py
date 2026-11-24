"""Copy completed checkpoints to a POSIX-mounted store, with checksum verification."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lact_qwen.checkpointing import archive_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--move", action="store_true", help="remove verified local weights/optimizer; keep a pointer")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        entries = [p for p in Path(args.source).glob("step_*") if p.name[5:].isdigit()]
        entries.sort(key=lambda p: int(p.name[5:]))
        entries.append(Path(args.source) / "final")
        for entry in entries:
            if (entry / "COMPLETE").is_file() and not (entry / "MOVED.txt").exists():
                archive_checkpoint(entry, Path(args.destination) / entry.name, move=args.move)
        if not args.watch or (entries[-1] / "COMPLETE").exists():
            break
        time.sleep(60)
