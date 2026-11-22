import asyncio, json, sys
from openai import AsyncOpenAI
out, prompts, *servers = sys.argv[1:]   # servers: policy=url
rows = [json.loads(l) for l in open(prompts)]
async def one(c, model, sem, r):
    async with sem:
        try:
            x = await c.chat.completions.create(model=model, messages=r["messages"], max_tokens=48, temperature=0.0)
            return x.choices[0].message.content
        except Exception as e:
            return "ERROR " + repr(e)[:200]
async def main():
    res = {}
    for s in servers:
        pol, url = s.split("=", 1)
        c = AsyncOpenAI(base_url=url, api_key="x", timeout=1800)
        model = (await c.models.list()).data[0].id
        sem = asyncio.Semaphore(4)
        outs = await asyncio.gather(*[one(c, model, sem, r) for r in rows])
        res[pol] = outs
    json.dump({"rows": [{k: v for k, v in r.items() if k != "messages"} for r in rows], "outputs": res}, open(out, "w"), indent=1)
    targets = sorted({r["target"] for r in rows})
    print("%-10s" % "policy" + "".join("%16s" % f"len {t}" for t in targets) + "   errors")
    for pol, outs in res.items():
        line = "%-10s" % pol
        for t in targets:
            idx = [i for i, r in enumerate(rows) if r["target"] == t]
            ok = sum(rows[i]["code"] in outs[i] for i in idx)
            line += "%16s" % f"{ok}/{len(idx)}"
        print(line + "   %d" % sum(o.startswith("ERROR") for o in outs))
    print("example:", {p: res[p][0][:60] for p in res}, "gold", rows[0]["code"])
asyncio.run(main())
