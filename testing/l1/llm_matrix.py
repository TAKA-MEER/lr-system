"""L1 議事録生成(LLM)の限界試験(TEST_PLAN_v2.md C群)。

台本から模擬文字起こしを作り、本番と同じ Stage1/Stage2 処理(app/llm/pipeline.py)に
使い捨てコンテナから直接投入して採点する。音声を介さないので条件を大量に振れる。

使い方:
    .venv/Scripts/python.exe l1/llm_matrix.py --sets domain,status --repeats 3 --tag base
    .venv/Scripts/python.exe l1/llm_matrix.py --sets all --repeats 3 --tag base
    (--score-only で採点だけやり直す)

結果: results/v2/l1_llm/<tag>/{jobs.json, out.jsonl, scores.jsonl, summary.md}
"""
import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluate"))
import docker_util  # noqa: E402
from transcript_sim import simulate  # noqa: E402
from v2_metrics import minutes_from_json, score_minutes, summary_row  # noqa: E402

TESTING_DIR = Path(__file__).resolve().parent.parent
SCEN_DIR = TESTING_DIR / "scenario"
PY = sys.executable


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_scenario(spec: dict) -> Path:
    """仕様dictから台本を生成し、出力ディレクトリを返す(動的に作る条件用)"""
    out = SCEN_DIR / "generated" / spec["name"]
    spec_path = SCEN_DIR / "specs" / "dynamic" / f"{spec['name']}.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    subprocess.run([PY, str(SCEN_DIR / "generate_scenario.py"), str(spec_path)], check=True, capture_output=True)
    return out


def scen(name: str) -> Path:
    out = SCEN_DIR / "generated" / name
    if not out.exists():
        subprocess.run([PY, str(SCEN_DIR / "generate_scenario.py"), str(SCEN_DIR / "specs" / f"{name}.yaml")], check=True)
    return out


ALL_DOMAINS = ["switchgear", "transformer", "motor", "plc"]


def condition_list(sets: list[str]) -> list[dict]:
    """各条件: {set, cond, scen_dir, labels, cer, flip, llm_overrides}"""
    conds = []

    def add(set_name, cond, scen_dir, labels="system", cer=0.0, flip=0.0, llm_overrides=None):
        conds.append({"set": set_name, "cond": cond, "scen_dir": str(scen_dir), "labels": labels,
                      "cer": cer, "flip": flip, "llm_overrides": llm_overrides or {}})

    if "domain" in sets:  # C-8 題材の違い、C-10 再現性
        for n in ["s_sg", "s_tr", "s_mt", "s_plc"]:
            add("domain", n, scen(n))
    if "status" in sets:  # C-5
        add("status", "s_status", scen("s_status"))
    if "reldate" in sets:  # C-6
        add("reldate", "s_reldate", scen("s_reldate"))
    if "chitchat" in sets:  # C-7
        add("chitchat", "s_chitchat", scen("s_chitchat"))
    if "cer" in sets:  # C-3 文字起こしの誤り
        for c in [0.0, 0.1, 0.2, 0.3, 0.4]:
            add("cer", f"cer{c:.1f}", scen("s_e2e_mix"), labels="oracle", cer=c)
    if "flip" in sets:  # C-4 話者ラベルの誤り
        for fl in [0.0, 0.1, 0.3, 0.5]:
            add("flip", f"flip{fl:.1f}", scen("s_e2e_mix"), labels="oracle", flip=fl)
    if "topics" in sets:  # C-1 協議事項の数
        for n in [2, 3, 5, 8, 10]:
            d = ensure_scenario({"name": f"dyn_topics_{n}", "seed": 100 + n, "domains": ALL_DOMAINS,
                                 "blocks_per_domain": n})
            add("topics", f"topics_x{n}", d)
    if "long" in sets:  # C-2 試験時間(文字数)
        for chars in [10000, 30000, 60000, 100000]:
            d = ensure_scenario({"name": f"dyn_long_{chars}", "seed": 200, "domains": ALL_DOMAINS,
                                 "filler_chars": chars})
            add("long", f"filler{chars // 1000}k", d)
    return conds


def build_jobs(conds: list[dict], repeats: int) -> list[dict]:
    jobs = []
    for c in conds:
        script = load_yaml(Path(c["scen_dir"]) / "script.yaml")
        m = script["meta"]
        meta = {"trial_name": m["trial_name"], "location": m["location"], "date": m["date"],
                "attendees": {"client": m["client_attendees"], "our_side": m["our_attendees"]}}
        reps = 1 if c["set"] == "long" else repeats
        for r in range(reps):
            text = simulate(script, labels=c["labels"], cer=c["cer"], flip=c["flip"], seed=r)
            jobs.append({"job_id": f"{c['set']}/{c['cond']}/r{r}", "set": c["set"], "cond": c["cond"],
                         "scen_dir": c["scen_dir"], "full_text": text, "meta": meta,
                         "llm_overrides": c["llm_overrides"], "text_chars": len(text)})
    return jobs


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def score(work: Path):
    jobs = {j["job_id"]: j for j in json.load(open(work / "jobs.json", encoding="utf-8"))}
    rows = []
    with open(work / "scores.jsonl", "w", encoding="utf-8") as fo:
        for line in open(work / "out.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            job = jobs.get(rec["job_id"])
            if not job:
                continue
            expected = load_yaml(Path(job["scen_dir"]) / "expected.yaml")
            if rec["error"]:
                row = {"error": rec["error"].splitlines()[0]}
                detail = None
            else:
                detail = score_minutes(minutes_from_json(rec["minutes"]), expected)
                row = summary_row(detail)
                row["parse_ok"] = rec["info"]["stage2_parse_ok"]
                row["chunks"] = rec["info"]["stage1_chunks"]
                row["stage2_out_chars"] = rec["info"].get("stage2_output_chars")
            row |= {"job_id": rec["job_id"], "set": job["set"], "cond": job["cond"], "seconds": rec["seconds"],
                    "text_chars": job["text_chars"]}
            rows.append(row)
            fo.write(json.dumps({"row": row, "detail": detail}, ensure_ascii=False) + "\n")

    # 条件ごとに平均(と最小)をまとめる
    cols = ["d_recall", "d_precision", "d_status", "d_status_changed", "d_values", "a_recall", "a_owner", "a_deadline"]
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["set"], r["cond"])].append(r)
    lines = ["| set | 条件 | n | 文字数 | 秒 | " + " | ".join(cols) + " | 解析失敗 | 余分(雑談/重複/他) | 年の捏造 | エラー |",
             "|" + " --- |" * (len(cols) + 10)]
    for (s, c), rs in grouped.items():
        ok = [r for r in rs if "error" not in r]
        vals = []
        for col in cols:
            xs = [r[col] for r in ok if r.get(col) is not None]
            vals.append(f"{statistics.mean(xs):.2f} (min {min(xs):.2f})" if len(xs) > 1 else fmt(xs[0] if xs else None))
        fails = sum(1 for r in ok if not r["parse_ok"])
        extra = "/".join(str(sum(r[k] for r in ok)) for k in ("d_extra_chitchat", "d_extra_dup", "d_extra_other"))
        years = ",".join(sorted({r["a_bad_years"] for r in ok if r["a_bad_years"]}))
        secs = statistics.mean(r["seconds"] for r in rs)
        lines.append(f"| {s} | {c} | {len(rs)} | {rs[0]['text_chars']} | {secs:.0f} | " + " | ".join(vals)
                     + f" | {fails} | {extra} | {years or '-'} | {len(rs) - len(ok)} |")
    md = "\n".join(lines)
    (work / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--keep-app", action="store_true", help="minutes-appを止めない(デバッグ用)")
    ap.add_argument("--num-gpu", type=int, default=None,
                    help="全ジョブのOllama num_gpu。試験時間短縮用(品質は同じモデルなので変わらない想定)")
    args = ap.parse_args()

    work = TESTING_DIR / "results" / "v2" / "l1_llm" / args.tag
    work.mkdir(parents=True, exist_ok=True)
    if not args.score_only:
        sets = ["domain", "status", "reldate", "chitchat", "cer", "flip", "topics", "long"] \
            if args.sets == "all" else args.sets.split(",")
        jobs = build_jobs(condition_list(sets), args.repeats)
        if args.num_gpu is not None:
            for j in jobs:
                j["llm_overrides"] = {"num_gpu": args.num_gpu} | j["llm_overrides"]
        jobs_path = work / "jobs.json"
        if jobs_path.exists():  # 既存ジョブに追記(同じjob_idは上書き)
            old = {j["job_id"]: j for j in json.load(open(jobs_path, encoding="utf-8"))}
            old |= {j["job_id"]: j for j in jobs}
            jobs = list(old.values())
        jobs_path.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"jobs={len(jobs)}")
        if not args.keep_app:
            docker_util.stop_app()
        rc = docker_util.run_in_container("llm_runner.py", ["/work/jobs.json", "/work/out.jsonl"], work, work / "run.log")
        print(f"container exit={rc}")
    score(work)


if __name__ == "__main__":
    main()
