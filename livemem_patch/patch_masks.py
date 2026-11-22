p = "/data0/qyl/livemem_repro_20261121/LiveMem/tools/build_select_masks.py"
s = open(p).read()
if "attn:K:A" not in s:
    def rep(a, b):
        global s
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    rep("  last:K:A             the K most recent sessions 1, the others A\n",
        "  last:K:A             the K most recent sessions 1, the others A\n"
        "  attn:K:A             the K sessions with the highest question-conditioned attention mass of the frozen\n"
        "                       base model (tools/score_attn.py, --scores file) 1, the others A\n")
    rep('    ap.add_argument("--seed", type=int, default=0)\n',
        '    ap.add_argument("--seed", type=int, default=0)\n    ap.add_argument("--scores", default=None, help="attn policy: json written by tools/score_attn.py")\n')
    rep("    rng = random.Random(args.seed)\n",
        "    rng = random.Random(args.seed)\n    attn_scores = json.load(open(args.scores)) if args.scores else None\n")
    rep("        else:\n            raise SystemExit(f\"unknown policy {args.policy}\")\n",
        "        elif name == \"attn\":\n"
        "            k, alpha = int(pa[0]), float(pa[1])\n"
        "            sc = attn_scores[req.id][\"doc_scores\"]\n"
        "            assert len(sc) == n, f\"score/document count mismatch for {req.id}\"\n"
        "            keep = set(sorted(range(n), key=lambda i: -sc[i])[:k])\n"
        "        else:\n            raise SystemExit(f\"unknown policy {args.policy}\")\n")
    open(p, "w").write(s)
import ast; ast.parse(s); print("patched")
