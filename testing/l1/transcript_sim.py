"""台本から「本番システムが出すはずの文字起こし」を模擬する(L1のLLM試験用、音声は使わない)。

本番は5秒ごとの音声チャンクを1行として "[話者] 本文" を並べたものを Stage1 に渡す。
ここでは台本の各ターンの長さを文字数から見積もり、5秒の窓に切り分けて同じ形の行を作る。

話者ラベルの付け方:
- "system": 本番の判定ロジックの理想動作(窓内にBT装着者の発話があれば our_side、なければ client)。
            BT非装着の自社側発話は client になる。
- "oracle": 窓内で最も長く話している人の本当の立場。話者判定が完璧な場合。
誤りの注入:
- cer: 文字単位で置換・脱落・挿入を入れる(実際のSTT誤りより無作為なので厳しめ)
- flip: 行ごとに話者ラベルを反転させる確率
"""
import random

CHARS_PER_SEC = 6.5
TURN_GAP_SEC = 0.6
WINDOW_SEC = 5.0


def build_timeline(script: dict) -> list[dict]:
    t = 0.0
    out = []
    for turn in script["turns"]:
        dur = max(0.6, len(turn["text"]) / CHARS_PER_SEC)
        out.append({"start": t, "end": t + dur, "text": turn["text"], "speaker": turn["speaker"],
                    "role": script["speakers"][turn["speaker"]],
                    "is_bt": turn["speaker"] in script["meta"].get("bt_wearers", [script["meta"].get("bt_wearer")])})
        t += dur + TURN_GAP_SEC + float(turn.get("wait_seconds", 0) or 0)
    return out


def windows(timeline: list[dict]) -> list[dict]:
    end = timeline[-1]["end"] if timeline else 0
    out = []
    w0 = 0.0
    while w0 < end:
        w1 = w0 + WINDOW_SEC
        text, role_sec, bt = [], {"client": 0.0, "our_side": 0.0}, False
        for tu in timeline:
            s, e = max(w0, tu["start"]), min(w1, tu["end"])
            if e <= s:
                continue
            n = len(tu["text"])
            dur = tu["end"] - tu["start"]
            a = round((s - tu["start"]) / dur * n)
            b = round((e - tu["start"]) / dur * n)
            text.append(tu["text"][a:b])
            role_sec[tu["role"]] += e - s
            bt = bt or tu["is_bt"]
        if "".join(text).strip():
            out.append({"text": "".join(text), "system": "our_side" if bt else "client",
                        "oracle": max(role_sec, key=role_sec.get)})
        w0 = w1
    return out


def corrupt(text: str, rate: float, rng: random.Random, charset: str) -> str:
    if rate <= 0:
        return text
    out = []
    for ch in text:
        r = rng.random()
        if r < rate * 0.5:
            out.append(rng.choice(charset))
        elif r < rate * 0.75:
            continue
        elif r < rate:
            out.append(ch)
            out.append(rng.choice(charset))
        else:
            out.append(ch)
    return "".join(out)


def simulate(script: dict, labels: str = "system", cer: float = 0.0, flip: float = 0.0, seed: int = 0) -> str:
    rng = random.Random(seed)
    ws = windows(build_timeline(script))
    charset = "".join(sorted(set("".join(t["text"] for t in script["turns"])) - set(" 、。")))
    lines = []
    for w in ws:
        sp = w[labels]
        if flip and rng.random() < flip:
            sp = "client" if sp == "our_side" else "our_side"
        lines.append(f"[{sp}] {corrupt(w['text'], cer, rng, charset)}")
    return "\n".join(lines)
