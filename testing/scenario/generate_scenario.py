"""
協議ブロック集(scenario/bank/*.yaml)と仕様(scenario/specs/*.yaml)から、
台本(script.yaml)と正解データ(expected.yaml)を同時に生成する。

台本は tts/generate_audio_v2.py(音声合成)と l1/llm_matrix.py(テキスト直接投入)の両方が読む。
正解データは evaluate/v2_eval.py が読む。

使い方:
    .venv/Scripts/python.exe scenario/generate_scenario.py scenario/specs/s_tr.yaml
    → scenario/generated/<spec.name>/script.yaml, expected.yaml

仕様ファイルの項目(省略時は既定値):
    name: 出力名
    seed: 乱数シード
    domains: [switchgear, transformer, motor, plc]   ブロックを取り出す題材(この順に並べる)
    blocks_per_domain: 各題材から使うブロック数(省略時は全部)
    trial_name / location: 省略時は先頭題材のmeta
    speakers: {client: [...], our_side: [...], bt_wearers: [...]}
    resolve: true            継続協議のブロックに resolve_turns があれば、数ブロック後に「合意済み」確定を挿入
    retract: 0               合意済みブロックのうち何件を後で撤回して継続協議に戻すか
    chitchat: 0.0            協議ブロック数に対する雑談の挿入割合
    deadline_mode: abs | rel | mixed
    filler_chars: 0          協議に関係しない測定値の読み上げ(長時間試験の模擬)の総文字数
    wait_seconds_after_block: [0, 0]   各ブロック後の無音(計測待ち)秒数の範囲
"""
import argparse
import random
import sys
from pathlib import Path

import yaml

SCENARIO_DIR = Path(__file__).resolve().parent
BANK_DIR = SCENARIO_DIR / "bank"
META_DATE = "2026年10月1日"

DEFAULT_SPEAKERS = {
    "client": ["山田 隆", "佐藤 健二"],
    "our_side": ["田中 誠", "鈴木 舞"],
    "bt_wearers": ["田中 誠"],
}

FILLER_POINTS = ["A相", "B相", "C相", "1番端子", "2番端子", "3番端子", "上部", "中央", "下部", "入口側", "出口側"]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def filler_turns(rng: random.Random, target_chars: int) -> list[dict]:
    """協議に関係しない測定値の読み上げと記録の確認。長時間試験の文字量を増やすためのもの。"""
    turns, total = [], 0
    while total < target_chars:
        pts = rng.sample(FILLER_POINTS, 3)
        vals = [f"{rng.uniform(10, 99):.1f}" for _ in pts]
        t1 = "測定値を読み上げます。" + "、".join(f"{p}が{v}" for p, v in zip(pts, vals)) + "です。"
        t2 = rng.choice(["記録しました。", "はい、記録表に記入しました。", "控えました。次お願いします。"])
        turns += [{"r": "our", "t": t1, "filler": True}, {"r": "our", "t": t2, "filler": True}]
        total += len(t1) + len(t2)
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    spec = load_yaml(Path(args.spec))
    rng = random.Random(spec.get("seed", 0))
    common = load_yaml(BANK_DIR / "common.yaml")
    banks = [load_yaml(BANK_DIR / f"{d}.yaml") for d in spec["domains"]]

    speakers = spec.get("speakers", DEFAULT_SPEAKERS)
    roles = {n: "client" for n in speakers["client"]} | {n: "our_side" for n in speakers["our_side"]}
    trial_name = spec.get("trial_name", banks[0]["meta"]["trial_name"])
    location = spec.get("location", banks[0]["meta"]["location"])

    # ---- ブロックの選択 ----
    blocks = []
    for bank in banks:
        bs = list(bank["blocks"])
        n = spec.get("blocks_per_domain")
        if n is not None and n < len(bs):
            keep = set(b["id"] for b in rng.sample(bs, n))
            bs = [b for b in bs if b["id"] in keep]
        blocks += bs

    # ---- 期限表現の割り当て ----
    deadline_mode = spec.get("deadline_mode", "abs")
    rel_list = common["relative_deadlines"]
    rel_idx = 0
    deadline_of = {}
    for b in blocks:
        if not b.get("action"):
            continue
        use_rel = deadline_mode == "rel" or (deadline_mode == "mixed" and rng.random() < 0.5)
        if use_rel:
            rel = rel_list[rel_idx % len(rel_list)]
            rel_idx += 1
            deadline_of[b["id"]] = (rel["text"], rel["accept"])
        else:
            abs_d = b["action"]["deadline_abs"]
            deadline_of[b["id"]] = (abs_d, [abs_d])

    # ---- 撤回するブロック ----
    agreed = [b for b in blocks if b["status"] == "合意済み"]
    retract_ids = set(b["id"] for b in rng.sample(agreed, min(spec.get("retract", 0), len(agreed))))

    # ---- 並べる: 開会 → ブロック(+雑談/再訪/撤回/測定読み上げ) → 閉会 ----
    # 各要素は (種類, ブロックID or None, ターン列)
    seq = [("opening", None, common["opening"])]
    pending = []  # (挿入位置=何ブロック後, 種類, ブロック)
    n_chitchat = round(spec.get("chitchat", 0.0) * len(blocks))
    chitchat_pool = list(common["chitchat"])
    rng.shuffle(chitchat_pool)
    chitchat_slots = set(rng.sample(range(len(blocks)), min(n_chitchat, len(blocks))))
    filler_total = spec.get("filler_chars", 0)
    filler_per_block = filler_total // max(1, len(blocks))
    used_chitchat = []

    for i, b in enumerate(blocks):
        seq.append(("block", b["id"], b["turns"]))
        if spec.get("resolve", True) and b["status"] == "継続協議" and b.get("resolve_turns"):
            pending.append((i + rng.randint(2, 4), "resolve", b))
        if b["id"] in retract_ids:
            pending.append((i + rng.randint(2, 4), "retract", b))
        if filler_per_block:
            seq.append(("filler", None, filler_turns(rng, filler_per_block)))
        for p in [p for p in pending if p[0] == i]:
            turns = p[2]["resolve_turns"] if p[1] == "resolve" else common["retract_turns"]
            seq.append((p[1], p[2]["id"], turns))
        pending = [p for p in pending if p[0] != i]
        if i in chitchat_slots:
            cc = chitchat_pool[len(used_chitchat) % len(chitchat_pool)]
            used_chitchat.append(cc)
            seq.append(("chitchat", cc["id"], cc["turns"]))
    for p in pending:  # 末尾まで来ても未挿入の再訪・撤回
        turns = p[2]["resolve_turns"] if p[1] == "resolve" else common["retract_turns"]
        seq.append((p[1], p[2]["id"], turns))
    seq.append(("closing", None, common["closing"]))

    # ---- 話者の割り当てと本文の置換 ----
    block_by_id = {b["id"]: b for b in blocks}
    wait_lo, wait_hi = spec.get("wait_seconds_after_block", [0, 0])
    turns_out = []
    for kind, bid, turns in seq:
        # 1つの話題は同じ担当者が話すのが自然なので、要素ごとに双方1名ずつ選ぶ
        pick = {"client": rng.choice(speakers["client"]), "our": rng.choice(speakers["our_side"])}
        for t in turns:
            text = t["t"]
            if bid in block_by_id:
                text = text.replace("{name}", block_by_id[bid]["name"])
                if bid in deadline_of:
                    text = text.replace("{deadline}", deadline_of[bid][0])
            text = text.replace("{trial_name}", trial_name)
            turns_out.append({
                "id": len(turns_out) + 1,
                "speaker": pick[t["r"]],
                "text": text,
                "kind": kind,
                "block": bid,
            })
        if kind == "block" and wait_hi > 0:
            turns_out[-1]["wait_seconds"] = round(rng.uniform(wait_lo, wait_hi), 1)

    script = {
        "meta": {
            "trial_name": trial_name,
            "location": location,
            "date": META_DATE,
            "client_attendees": speakers["client"],
            "our_attendees": speakers["our_side"],
            "bt_wearers": speakers["bt_wearers"],
            "bt_wearer": speakers["bt_wearers"][0],  # 旧ツール互換
        },
        "speakers": roles,
        "turns": turns_out,
    }

    # ---- 正解データ ----
    discussions, actions = [], []
    for b in blocks:
        status = b["status"]
        if status == "継続協議" and spec.get("resolve", True) and b.get("resolve_turns"):
            status = "合意済み"
        if b["id"] in retract_ids:
            status = "継続協議"
        discussions.append({
            "id": b["id"],
            "topic_keywords": b["topic_keywords"],
            "request_keywords": b.get("request_keywords", []),
            "response_keywords": b.get("response_keywords", []),
            "expected_status": status,
            "status_changed": status != b["status"],
            "values": [str(v) for v in b.get("values", [])],
        })
        if b.get("action"):
            actions.append({
                "id": f"{b['id']}_action",
                "content_keywords": b["action"]["content_keywords"],
                "owner": b["action"]["owner"],
                "deadline_text": deadline_of[b["id"]][0],
                "deadline_accept": deadline_of[b["id"]][1],
            })
    terms = []
    for t in turns_out:
        if t["block"] in block_by_id and t["kind"] == "block":
            for term in block_by_id[t["block"]].get("terms", []):
                alts = term.split("|")
                if any(a in t["text"] for a in alts):
                    terms.append({"turn_id": t["id"], "alts": alts})

    expected = {
        "meta": {"trial_name": trial_name, "location": location, "date": META_DATE,
                 "client_attendees": speakers["client"], "our_attendees": speakers["our_side"]},
        "discussions": discussions,
        "action_items": actions,
        "chitchat_keywords": sorted({k for cc in used_chitchat for k in cc["keywords"]}),
        "terms": terms,
    }

    out_dir = Path(args.out_dir) if args.out_dir else SCENARIO_DIR / "generated" / spec["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "script.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(script, f, allow_unicode=True, sort_keys=False, width=200)
    with open(out_dir / "expected.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(expected, f, allow_unicode=True, sort_keys=False, width=200)

    chars = sum(len(t["text"]) for t in turns_out)
    print(f"{spec['name']}: blocks={len(blocks)} turns={len(turns_out)} chars={chars} "
          f"actions={len(actions)} retract={len(retract_ids)} chitchat={len(used_chitchat)} -> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
