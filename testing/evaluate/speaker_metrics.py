"""話者判定精度を評価する。

現行システムのSpeakerDetectorは「BTマイクのRMS履歴内で閾値を超えた記録があるか」で
our_side/clientを判定するため、bt_wearer以外のour_side話者(鈴木舞)の発話は
clientと誤判定されることが期待される(expected_minutes_v1.yaml の system_expected を正解とする)。
"""


def build_expected_timeline(timeline: dict) -> list[dict]:
    """timeline.json の各ターンに [start_sec, end_sec] と is_bt_wearer を保持したまま返す。"""
    return timeline["turns"]


def expected_speaker_for_window(turns: list[dict], window_start: float, window_end: float) -> str:
    """[window_start, window_end] 内にbt_wearerの発話が(部分的にでも)含まれていれば our_side、
    含まれていなければ client を返す(SpeakerDetector.get_speaker のロジックを模倣)。"""
    for t in turns:
        if t["is_bt_wearer"] and t["start_sec"] < window_end and t["end_sec"] > window_start:
            return "our_side"
    return "client"


def evaluate_speaker_accuracy(
    transcript_segments: list[dict],
    timeline: dict,
    internal_start_wallclock: float,
    chunk_window_sec: float = 5.0,
) -> dict:
    turns = build_expected_timeline(timeline)
    results = []
    correct = 0
    for seg in transcript_segments:
        elapsed_end = seg["timestamp"] - internal_start_wallclock
        elapsed_start = elapsed_end - chunk_window_sec
        expected = expected_speaker_for_window(turns, elapsed_start, elapsed_end)
        actual = seg["speaker"]
        is_correct = expected == actual
        correct += int(is_correct)
        results.append(
            {
                "elapsed_start": round(elapsed_start, 2),
                "elapsed_end": round(elapsed_end, 2),
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "text": seg["text"][:40],
            }
        )
    total = len(results)
    accuracy = correct / total if total else None
    return {"accuracy": accuracy, "correct": correct, "total": total, "details": results}
