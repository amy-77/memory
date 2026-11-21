import os, yaml, pyarrow as pa
W = os.environ["W"]
for ds in ["musique", "2wikimultihopqa", "longmit", "narrativeqa", "qasper", "ldc", "longalign", "longalpaca", "agnews", "dbpedia"]:
    try:
        t = pa.ipc.open_file(f"dataset/train/{ds}/single_sft.arrow").read_all()
    except Exception as e:
        print(ds, "MISSING", repr(e)[:80]); continue
    ev = t.column("evidence_chunk_idx").to_pylist() if "evidence_chunk_idx" in t.column_names else None
    print(ds, "rows", t.num_rows, "with evidence", None if ev is None else sum(1 for e in ev if e))
base = yaml.safe_load(open("train/sft/config/livemem_sft_stage2.yaml"))
base["data"]["datasets"] = [d for d in base["data"]["datasets"] if os.path.exists(f"dataset/train/{d[0]}/single_sft.arrow")]
base["model"]["init_from_merged"] = f"{W}/models/LiveMem-SFT-vllm"
base["model"]["qwen3_path"] = f"{W}/models/Qwen3-4B-Instruct-2507"
base["train"].pop("resume_trainable_from", None)
base["train"]["loop"].update(max_steps=120, grad_accum=4, save_every=40, constant_steps=20, use_wandb=False, log_every=1)
for name, sel in (("sel", 0.8), ("ctrl", 0.0)):
    c = yaml.safe_load(yaml.safe_dump(base))
    c["data"]["sel_prob"] = sel; c["data"]["sel_distract"] = 0.3
    c["train"]["loop"]["output_dir"] = f"{W}/outputs/sft/ft_{name}"; c["train"]["loop"]["run_name"] = f"ft_{name}"
    yaml.safe_dump(c, open(f"{W}/sel/ft_{name}.yaml", "w"), sort_keys=False)
    print(name, [d[0] for d in c["data"]["datasets"]], c["train"]["loop"]["max_steps"])
