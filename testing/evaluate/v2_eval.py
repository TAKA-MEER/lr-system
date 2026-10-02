"""L2 E2E試験(run_injection.py の結果)を第2期の指標で採点する。

    .venv/Scripts/python.exe evaluate/v2_eval.py <run_id> --scen scenario/generated/s_e2e_mix --audio-dir results/v2/audio_l2/<cond>

出力: results/<run_id>/eval_v2.json と results/v2/l2_summary.md への1行追記
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minutes_metrics import parse_minutes_docx  # noqa: E402
from v2_metrics import cer, score_minutes, summary_row, term_accuracy  # noqa: E402

TESTING_DIR = Path(__file__).resolve().parent.parent
WIN = 5.0


def speaker_accuracy(segments: list[dict], turns: list[dict]) -> dict:
    """発話時間の重み付き正解率。チャンク番号 j(1始まり)は内蔵マイク音声の [5(j-1), 5j] 秒に当たる"""
    correct = total = 0.0
    nonbt = nonbt_as_client = 0.0
    for s in segments:
        j = s.get("chunk_index")
        if not j:
            continue
        a, b = (j - 1) * WIN, j * WIN
        rs = {"client": 0.0, "our_side": 0.0}
        for t in turns:
            o = min(b, t["end_sec"]) - max(a, t["start_sec"])
            if o > 0:
                rs[t["role"]] += o
                if t["role"] == "our_side" and not t["is_bt_wearer"]:
                    nonbt += o
                    nonbt_as_client += o if s["speaker"] == "client" else 0
        correct += rs.get(s["speaker"], 0.0)
        total += rs["client"] + rs["our_side"]
    return {"acc": correct / total if total else None, "nonbt_share": nonbt / total if total else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--scen", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    run_dir = TESTING_DIR / "results" / args.run_id
    scen = TESTING_DIR / args.scen
    script = yaml.safe_load(open(scen / "script.yaml", encoding="utf-8"))
    expected = yaml.safe_load(open(scen / "expected.yaml", encoding="utf-8"))
    tl = json.loads((TESTING_DIR / args.audio_dir / "timeline.json").read_text(encoding="utf-8"))
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    segs = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))["segments"]
    segs = sorted(segs, key=lambda s: (s.get("chunk_index") or 0, s["timestamp"]))
    hyp = "".join(s["text"] for s in segs)
    ref = "".join(t["text"] for t in script["turns"])

    rep = {"run_id": args.run_id, "label": args.label, "run_info": info,
           "cer": cer(ref, hyp), "terms": term_accuracy(expected["terms"], hyp),
           "speaker": speaker_accuracy(segs, tl["turns"])}
    lat_path = run_dir / "latency.json"
    if lat_path.exists():
        lat = json.loads(lat_path.read_text(encoding="utf-8"))
        ds = [lat["received"][k] - lat["sent"][k] for k in lat["received"] if k in lat["sent"]]
        if ds:
            ds.sort()
            rep["latency"] = {"median": statistics.median(ds), "p95": ds[int(0.95 * (len(ds) - 1))], "max": ds[-1],
                              "n": len(ds), "chunks_sent": len(lat["sent"])}
    if (run_dir / "minutes.docx").exists():
        m = parse_minutes_docx(str(run_dir / "minutes.docx"))
        rep["minutes"] = score_minutes(m, expected)
        since = str(int(info.get("restart_epoch", 0))) if info.get("restart_epoch") else "3h"
        r = subprocess.run(["docker", "logs", "--since", since, "minutes-app"], capture_output=True, timeout=60)
        rep["stage2_parse_fail"] = (r.stdout + r.stderr).decode(errors="replace").count("JSON解析失敗")
    (run_dir / "eval_v2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    def f(v, p=3):
        return "-" if v is None else f"{v:.{p}f}"

    row = summary_row(rep["minutes"]) if "minutes" in rep else {}
    lt = rep.get("latency", {})
    line = (f"| {args.run_id} | {args.label} | {f(rep['cer']['cer'])} | {f(rep['terms']['accuracy'], 2)} | "
            f"{f(rep['speaker']['acc'])} | {f(lt.get('median'), 1)} / {f(lt.get('p95'), 1)} / {f(lt.get('max'), 1)} | "
            f"{f(row.get('d_recall'), 2)} | {f(row.get('d_precision'), 2)} | {f(row.get('d_status'), 2)} | "
            f"{f(row.get('d_values'), 2)} | {f(row.get('a_recall'), 2)} | {f(row.get('a_deadline'), 2)} | "
            f"{rep.get('stage2_parse_fail', '-')} | {f(info.get('generate_duration_sec'), 0)} |")
    summ = TESTING_DIR / "results" / "v2" / "l2_summary.md"
    if not summ.exists():
        summ.write_text("| run | 条件 | CER | 数値・用語 | 話者 | 遅延 中央/p95/最大[s] | 協議 再現 | 協議 適合 | 状態 | 数値転記 | "
                        "アクション再現 | 期限 | 解析失敗 | 生成[s] |\n" + "|" + " --- |" * 14 + "\n", encoding="utf-8")
    with open(summ, "a", encoding="utf-8") as fo:
        fo.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
