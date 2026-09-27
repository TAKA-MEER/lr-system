"""話者判定の採点(文字列の突き合わせ版)。文字起こしの各行を台本の文字列に対応付け、
対応した文字ごとに「台本上の話者の立場」と「行に付いた話者ラベル」が一致するかを数える。
チャンクの時刻に頼らないので、1チャンクを話者ごとに分けて記録する改善版も公平に採点できる。

    .venv/Scripts/python.exe evaluate/speaker_text_align.py <run_id> --scen scenario/generated/s_e2e_mix
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_metrics import norm  # noqa: E402

TESTING_DIR = Path(__file__).resolve().parent.parent


def score(segments: list[dict], script: dict) -> dict:
    ref, ref_role, ref_nonbt = [], [], []
    wearers = set(script["meta"].get("bt_wearers") or [script["meta"].get("bt_wearer")])
    for t in script["turns"]:
        s = norm(t["text"])
        role = script["speakers"][t["speaker"]]
        ref.append(s)
        ref_role += [role] * len(s)
        ref_nonbt += [role == "our_side" and t["speaker"] not in wearers] * len(s)
    ref_s = "".join(ref)
    hyp_s, hyp_label = "", []
    for seg in sorted(segments, key=lambda x: (x.get("chunk_index") or 0, x["timestamp"])):
        s = norm(seg["text"])
        hyp_s += s
        hyp_label += [seg["speaker"]] * len(s)
    sm = difflib.SequenceMatcher(None, ref_s, hyp_s, autojunk=False)
    matched = correct = nonbt = nonbt_correct = 0
    for a, b, n in sm.get_matching_blocks():
        for i in range(n):
            matched += 1
            ok = ref_role[a + i] == hyp_label[b + i]
            correct += ok
            if ref_nonbt[a + i]:
                nonbt += 1
                nonbt_correct += ok
    bt_only = (correct - nonbt_correct) / (matched - nonbt) if matched - nonbt else None
    return {"acc": correct / matched if matched else None, "acc_excluding_nonbt": bt_only,
            "matched_chars": matched, "nonbt_share": nonbt / matched if matched else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--scen", required=True)
    a = ap.parse_args()
    script = yaml.safe_load(open(TESTING_DIR / a.scen / "script.yaml", encoding="utf-8")) if not a.scen.endswith(".yaml") \
        else yaml.safe_load(open(TESTING_DIR / a.scen, encoding="utf-8"))
    segs = json.loads((TESTING_DIR / "results" / a.run_id / "transcript.json").read_text(encoding="utf-8"))["segments"]
    print(json.dumps(score(segs, script), ensure_ascii=False))


if __name__ == "__main__":
    main()
