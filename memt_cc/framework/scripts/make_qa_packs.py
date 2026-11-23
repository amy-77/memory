"""Build long-context QA inputs for stage 2 from public multi-hop QA sets (MuSiQue, HotpotQA distractor).

One example of these sets is only 1-3K tokens, which needs no memory.  As in LiveMem's 64K packs, the
paragraphs of several examples are packed into one long context and the question of ONE of them is asked,
so the answer has to be found among many unrelated passages read earlier.

Output: .jsonl rows {"id", "context", "question", "answer"} for scripts/make_post_data.py
("answer" is the gold answer, kept for reference only; stage 2 trains on the teacher's roll-out).

    python scripts/make_qa_packs.py --musique musique_ans_v1.0_train.jsonl --hotpot "distractor/train-*.parquet" \
        --out qa_packs.jsonl --num 4000 --min-chars 24000 --max-chars 100000
"""

import argparse
import glob
import json
import random


def musique_examples(path):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("answerable", True):
                continue
            passages = [f"{p['title']}\n{p['paragraph_text']}" for p in row["paragraphs"]]
            yield {"id": "musique-" + str(row["id"]), "passages": passages, "question": row["question"], "answer": row["answer"]}


def hotpot_examples(pattern):
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(pattern)):
        for row in pq.read_table(path, columns=["id", "question", "answer", "context"]).to_pylist():
            ctx = row["context"]
            passages = [f"{t}\n{''.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"])]
            yield {"id": "hotpot-" + str(row["id"]), "passages": passages, "question": row["question"], "answer": row["answer"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--musique", default="")
    p.add_argument("--hotpot", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=4000)
    p.add_argument("--min-chars", type=int, default=24000)
    p.add_argument("--max-chars", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    pool = []
    if args.musique:
        pool += list(musique_examples(args.musique))
    if args.hotpot:
        pool += list(hotpot_examples(args.hotpot))
    rng.shuffle(pool)
    written, cursor = 0, 0
    with open(args.out, "w") as out:
        while written < args.num and cursor < len(pool):
            target = rng.randint(args.min_chars, args.max_chars)
            group, size = [], 0
            while cursor < len(pool) and size < target:
                ex = pool[cursor]
                cursor += 1
                group.append(ex)
                size += sum(len(x) + 2 for x in ex["passages"])
            if size < args.min_chars:
                break
            asked = rng.choice(group)  # position of the relevant passages is uniform over the context
            passages = [x for ex in group for x in ex["passages"]]
            seen, unique = set(), []
            for x in passages:  # the same Wikipedia paragraph can appear in several examples
                if x not in seen:
                    seen.add(x)
                    unique.append(x)
            rng.shuffle(unique)
            context = "\n\n".join(f"Document {i + 1}: {x}" for i, x in enumerate(unique))
            question = "Answer the question using the documents above. Give a short answer.\nQuestion: " + asked["question"]
            out.write(json.dumps({"id": asked["id"], "context": context, "question": question, "answer": asked["answer"]}, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} packs from {cursor} examples to {args.out}")


if __name__ == "__main__":
    main()
