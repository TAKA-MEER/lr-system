"""文字起こし精度(文字誤り率 CER)を計算する。"""
from rapidfuzz.distance import Levenshtein


def normalize(text: str) -> str:
    return "".join(text.split())  # 空白除去のみ(日本語は分かち書きしない)


def build_reference_text(script: dict) -> str:
    return normalize("".join(turn["text"] for turn in script["turns"]))


def build_hypothesis_text(transcript_segments: list[dict]) -> str:
    ordered = sorted(transcript_segments, key=lambda s: s["timestamp"])
    return normalize("".join(s["text"] for s in ordered))


def compute_cer(reference: str, hypothesis: str) -> dict:
    if not reference:
        return {"cer": None, "edit_distance": None, "ref_len": 0}
    dist = Levenshtein.distance(reference, hypothesis)
    return {
        "cer": dist / len(reference),
        "edit_distance": dist,
        "ref_len": len(reference),
        "hyp_len": len(hypothesis),
    }
