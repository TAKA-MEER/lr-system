"""重複対策の比較実験(exp_queue.py)の結果を1つの表にまとめる。

    .venv/Scripts/python.exe evaluate/compare_exp.py base prog-merge topic-cluster stage2-thinking
出力: results/v2/exp_compare.md
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TESTING_DIR = Path(__file__).resolve().parent.parent
SETS = ["long", "long_single", "topics", "domain", "status"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def f1(p, r):
    return 2 * p * r / (p + r) if p and r else 0.0


def load(name: str):
    work = TESTING_DIR / "results" / "v2" / "l1_llm" / f"exp_{name}"
    rows = [json.loads(l)["row"] for l in open(work / "scores.jsonl", encoding="utf-8")]
    info = {}
    for l in open(work / "out.jsonl", encoding="utf-8"):
        r = json.loads(l)
        info[r["job_id"]] = r.get("info") or {}
    return rows, info


def fmt(v, p=2):
    return "-" if v is None else f"{v:.{p}f}"


def main():
    names = sys.argv[1:]
    lines = ["| 案 | セット | 件数 | 再現率 | 適合率 | F値 | 重複件数(平均) | 状態正解率 | 状態変化 | 測定値 | アクション再現 | 解析失敗 | 生成時間 平均[秒] |",
             "|" + " --- |" * 13]
    long_time = []
    for name in names:
        rows, info = load(name)
        by = defaultdict(list)
        for r in rows:
            if "error" not in r:
                by[r["set"]].append(r)
        for s in SETS:
            rs = by.get(s, [])
            if not rs:
                continue
            rec, pre = mean(r["d_recall"] for r in rs), mean(r["d_precision"] for r in rs)
            fs = mean(f1(r["d_precision"] or 0, r["d_recall"] or 0) for r in rs)
            lines.append(
                f"| {name} | {s} | {len(rs)} | {fmt(rec)} | {fmt(pre)} | {fmt(fs)} | {fmt(mean(r['d_extra_dup'] for r in rs), 1)} | "
                f"{fmt(mean(r['d_status'] for r in rs))} | {fmt(mean(r['d_status_changed'] for r in rs))} | "
                f"{fmt(mean(r['d_values'] for r in rs))} | {fmt(mean(r['a_recall'] for r in rs))} | "
                f"{sum(1 for r in rs if not r.get('parse_ok', True))} | {fmt(mean(r['seconds'] for r in rs), 0)} |")
        for r in rows:
            if r.get("cond") in ("filler30k",):
                long_time.append((name, r["job_id"], r["seconds"]))
        think = [i.get("stage2_think_ok") for i in info.values() if "stage2_think_ok" in i]
        if think:
            lines.append(f"| {name} | (thinking で回答できた割合) | {len(think)} | {sum(think) / len(think):.2f} | | | | | | | | | |")
    lines += ["", "3時間相当(3万文字)の生成時間:", ""]
    for name, job, sec in long_time:
        lines.append(f"- {name} {job}: {sec / 60:.1f}分")
    md = "\n".join(lines)
    (TESTING_DIR / "results" / "v2" / "exp_compare.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
