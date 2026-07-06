"""
results/<run_id>/ の成果物(transcript.json, minutes.docx, run_info.json)を
scenario/script_v1.yaml, scenario/expected_minutes_v1.yaml, results/audio/timeline.json
と突き合わせて評価し、eval_report.md を生成する。results/comparison.md にも追記する。

使い方:
    .venv/Scripts/python.exe evaluate/run_eval.py <run_id>
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minutes_metrics import evaluate_action_items, evaluate_discussions, evaluate_status_conflict, parse_minutes_docx
from speaker_metrics import evaluate_speaker_accuracy
from stt_metrics import build_hypothesis_text, build_reference_text, compute_cer

TESTING_DIR = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_stage2_fallback_in_logs(since_iso: str, container_name: str = "minutes-app") -> int:
    """docker logs --since <restart時刻> から'JSON解析失敗'の出現回数を数える。

    コンテナはrunごとにrestart(再作成ではない)されるため、--sinceを付けないと
    過去runのログが混入してしまう点に注意(docker logsはプロセス再起動をまたいで蓄積される)。
    """
    try:
        result = subprocess.run(
            ["docker", "logs", "--since", since_iso, container_name], capture_output=True, timeout=30
        )
        text = result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
        return text.count("JSON解析失敗")
    except Exception as e:
        print(f"docker logs取得失敗: {e}")
        return -1


def main():
    if len(sys.argv) < 2:
        print("使い方: run_eval.py <run_id>")
        sys.exit(1)
    run_id = sys.argv[1]

    cfg = load_yaml(TESTING_DIR / "config.yaml")
    run_dir = TESTING_DIR / cfg["paths"]["results_dir"] / run_id
    script = load_yaml(TESTING_DIR / cfg["paths"]["script"])
    expected = load_yaml(TESTING_DIR / cfg["paths"]["expected"])
    timeline = json.loads((TESTING_DIR / cfg["paths"]["audio_dir"] / "timeline.json").read_text(encoding="utf-8"))

    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    segments = transcript["segments"]

    # ---- STT精度 ----
    ref = build_reference_text(script)
    hyp = build_hypothesis_text(segments)
    cer_result = compute_cer(ref, hyp)

    # ---- 話者判定精度 ----
    speaker_result = None
    if "internal_start_wallclock" in run_info:
        speaker_result = evaluate_speaker_accuracy(
            segments, timeline, run_info["internal_start_wallclock"],
            chunk_window_sec=cfg["injection"]["chunk_interval_seconds"],
        )

    # ---- 議事録メトリクス ----
    minutes = parse_minutes_docx(str(run_dir / "minutes.docx"))
    discussion_result = evaluate_discussions(expected["discussions"], minutes["discussions"])
    action_result = evaluate_action_items(expected["action_items"], minutes["action_items"])
    conflict_result = evaluate_status_conflict(expected["status_conflict_check"], minutes["discussions"])
    since = str(int(run_info["restart_epoch"])) if "restart_epoch" in run_info else "1h"
    stage2_fallback_count = check_stage2_fallback_in_logs(since)

    report = {
        "run_id": run_id,
        "run_info": run_info,
        "stt_cer": cer_result,
        "speaker_accuracy": speaker_result,
        "discussions": discussion_result,
        "action_items": action_result,
        "status_conflict_check": conflict_result,
        "stage2_fallback_count_in_logs": stage2_fallback_count,
    }

    report_json_path = run_dir / "eval_report.json"
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render_markdown(report)
    (run_dir / "eval_report.md").write_text(md, encoding="utf-8")

    comparison_path = TESTING_DIR / cfg["paths"]["results_dir"] / "comparison.md"
    with open(comparison_path, "a", encoding="utf-8") as f:
        f.write(render_comparison_row(report) + "\n")

    print(md)


def render_markdown(r: dict) -> str:
    lines = [f"# 評価レポート: {r['run_id']}", ""]
    lines.append(f"- 音声注入時間: {r['run_info'].get('injection_duration_sec')}秒")
    lines.append(f"- 議事録生成時間: {r['run_info'].get('generate_duration_sec')}秒")
    lines.append(f"- 文字起こしセグメント数: {r['run_info'].get('num_segments')}")
    lines.append("")

    lines.append("## STT文字誤り率(CER)")
    cer = r["stt_cer"]
    if cer["cer"] is not None:
        lines.append(f"- CER: {cer['cer']:.3f} (編集距離={cer['edit_distance']}, 正解長={cer['ref_len']}, 認識長={cer['hyp_len']})")
    else:
        lines.append("- 計算不可(正解データが空)")
    lines.append("")

    lines.append("## 話者判定精度")
    sp = r["speaker_accuracy"]
    if sp:
        lines.append(f"- 精度: {sp['accuracy']:.3f} ({sp['correct']}/{sp['total']})")
    else:
        lines.append("- 計算不可(internal_start_wallclockが記録されていないrun)")
    lines.append("")

    lines.append("## 協議事項の再現率・正確性")
    d = r["discussions"]
    lines.append(f"- 再現率: {d['recall']:.3f} ({d['matched_count']}/{d['total']})" if d["recall"] is not None else "- N/A")
    if d["status_accuracy_among_matched"] is not None:
        lines.append(f"- 状態正確性(マッチ分のみ): {d['status_accuracy_among_matched']:.3f}")
    lines.append(f"- ハルシネーション候補(正解と対応がつかない生成項目数): {len(d['hallucination_candidates'])}")
    for item in d["details"]:
        mark = "OK" if item["matched"] else "MISS"
        status_mark = "OK" if item["status_correct"] else ("NG" if item["matched"] else "-")
        lines.append(f"  - [{item['id']}] {mark} status:{status_mark} expected={item['expected_status']} generated={item['generated_status']}")
    lines.append("")

    lines.append("## アクションアイテムの再現率・正確性")
    a = r["action_items"]
    lines.append(f"- 再現率: {a['recall']:.3f} ({a['matched_count']}/{a['total']})" if a["recall"] is not None else "- N/A")
    if a["owner_accuracy_among_matched"] is not None:
        lines.append(f"- owner正確性(マッチ分のみ): {a['owner_accuracy_among_matched']:.3f}")
    if a["deadline_accuracy_among_matched"] is not None:
        lines.append(f"- deadline正確性(マッチ分のみ): {a['deadline_accuracy_among_matched']:.3f}")
    for item in a["details"]:
        mark = "OK" if item["matched"] else "MISS"
        lines.append(f"  - [{item['id']}] {mark} owner:{item['generated_owner']} deadline:{item['generated_deadline']}")
    lines.append("")

    lines.append("## 状態矛盾解消の正しさ(絶縁抵抗: 継続協議→合意済み)")
    c = r["status_conflict_check"]
    lines.append(f"- 検出: {c['found']}, 正解一致: {c['correct']}, 生成された状態: {c['generated_status']}")
    lines.append("")

    lines.append("## Stage2 JSON解析失敗(フォールバック発生)回数")
    lines.append(f"- {r['stage2_fallback_count_in_logs']} 回 (docker logs 走査)")

    return "\n".join(lines)


def render_comparison_row(r: dict) -> str:
    cer = r["stt_cer"]["cer"]
    sp = r["speaker_accuracy"]["accuracy"] if r["speaker_accuracy"] else None
    d_recall = r["discussions"]["recall"]
    a_recall = r["action_items"]["recall"]
    conflict_ok = r["status_conflict_check"]["correct"]

    cer_s = f"{cer:.3f}" if cer is not None else "N/A"
    sp_s = f"{sp:.3f}" if sp is not None else "N/A"
    d_s = f"{d_recall:.3f}" if d_recall is not None else "N/A"
    a_s = f"{a_recall:.3f}" if a_recall is not None else "N/A"

    return f"| {r['run_id']} | {cer_s} | {sp_s} | {d_s} | {a_s} | {conflict_ok} | {r['stage2_fallback_count_in_logs']} |"


if __name__ == "__main__":
    main()
