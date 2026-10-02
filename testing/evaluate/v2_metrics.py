"""第2期評価試験(TEST_PLAN_v2.md)の採点部品。

旧 minutes_metrics.py との違い:
- 生成項目と正解項目を1対1で対応付ける(1つの生成項目が複数の正解に当たるのを防ぐ)
- 適合率(正解に対応しない生成項目の割合)と、その内訳(雑談の混入・重複・その他)を出す
- 数値の転記正確性、期限の年のでっち上げ、雑談の混入を採点する
- STTは記号・全角半角を正規化したCERと、数値・型式・専門用語の認識率を出す
"""
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

PUNCT_RE = re.compile(r"[\s、。，．,！？!?「」『』（）()・…ー―\-〜~:：;；\"'“”‘’\[\]]")


def norm(text: str) -> str:
    """NFKC(全角→半角)、記号・空白除去。「ー」も除く(長音の有無の揺れを吸収)"""
    return PUNCT_RE.sub("", unicodedata.normalize("NFKC", text or ""))


def norm_loose(text: str) -> str:
    """キーワード照合用。NFKCと空白除去のみ(小数点やハイフンは残す)"""
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", text or ""))


# ------------------------------------------------------------------
# STT
# ------------------------------------------------------------------

def cer(ref: str, hyp: str) -> dict:
    r, h = norm(ref), norm(hyp)
    if not r:
        return {"cer": None, "ref_len": 0, "hyp_len": len(h), "edit_distance": None}
    d = Levenshtein.distance(r, h)
    return {"cer": d / len(r), "ref_len": len(r), "hyp_len": len(h), "edit_distance": d}


def term_accuracy(terms: list[dict], hyp_text: str) -> dict:
    """正解データの terms(各 {turn_id, alts})が文字起こし全体に出現するか。
    同じ語が複数回出る場合を考慮し、出現回数を上限として数える。"""
    h = norm_loose(hyp_text)
    hit = 0
    budget: dict[str, int] = {}
    details = []
    for t in terms:
        alts = [norm_loose(a) for a in t["alts"]]
        found = None
        for a in alts:
            used = budget.get(a, 0)
            if h.count(a) > used:
                budget[a] = used + 1
                found = a
                break
        hit += int(found is not None)
        details.append({"turn_id": t["turn_id"], "alts": t["alts"], "found": found})
    return {"accuracy": hit / len(terms) if terms else None, "hit": hit, "total": len(terms), "details": details}


# ------------------------------------------------------------------
# 議事録
# ------------------------------------------------------------------

def minutes_from_json(m: dict) -> dict:
    """議事録JSON(LLM出力)を parse_minutes_docx と同じ形にそろえる"""
    ds = []
    for d in m.get("discussions", []) or []:
        if not isinstance(d, dict):
            continue
        ds.append({k: str(d.get(k) or "") for k in ("topic", "client_request", "our_response", "status")})
    acts = []
    for a in m.get("action_items", []) or []:
        if not isinstance(a, dict):
            continue
        owner = a.get("owner")
        label = "自社" if owner == "our_side" else "相手方" if owner == "client" else str(owner or "")
        acts.append({"content": str(a.get("content") or ""), "owner_label": label, "deadline": str(a.get("deadline") or "")})
    return {"discussions": ds, "action_items": acts}


def _hits(keywords: list[str], text: str) -> int:
    t = norm_loose(text)
    return sum(1 for k in keywords if norm_loose(k) in t)


def _greedy_assign(scores: list[tuple[int, int, float]]) -> dict[int, int]:
    """(expected_idx, generated_idx, score) の一覧から、スコアの高い順に1対1で割り当てる"""
    assigned_e, assigned_g, pairs = set(), set(), {}
    for e, g, s in sorted(scores, key=lambda x: -x[2]):
        if e in assigned_e or g in assigned_g:
            continue
        assigned_e.add(e)
        assigned_g.add(g)
        pairs[e] = g
    return pairs


def score_discussions(expected: list[dict], generated: list[dict], chitchat_keywords: list[str]) -> dict:
    scores = []
    for ei, e in enumerate(expected):
        for gi, g in enumerate(generated):
            th = _hits(e["topic_keywords"], g["topic"])
            body = g["topic"] + " " + g["client_request"] + " " + g["our_response"]
            bh = _hits(e.get("request_keywords", []) + e.get("response_keywords", []), body)
            if th >= 1 or bh >= 2:
                scores.append((ei, gi, th * 2 + bh))
    pairs = _greedy_assign(scores)

    details = []
    status_ok = values_hit = values_total = 0
    changed_total = changed_ok = 0
    for ei, e in enumerate(expected):
        gi = pairs.get(ei)
        g = generated[gi] if gi is not None else None
        ok = g is not None and g["status"].strip() == e["expected_status"]
        status_ok += int(ok)
        vh = 0
        if g is not None:
            body = norm_loose(g["topic"] + g["client_request"] + g["our_response"])
            vh = sum(1 for v in e.get("values", []) if norm_loose(v) in body)
            values_hit += vh
            values_total += len(e.get("values", []))
        if e.get("status_changed"):
            changed_total += 1
            changed_ok += int(ok)
        details.append({
            "id": e["id"], "matched": g is not None, "expected_status": e["expected_status"],
            "generated_status": g["status"] if g else None, "status_correct": ok,
            "values": f"{vh}/{len(e.get('values', []))}" if g else None,
            "generated_topic": g["topic"] if g else None,
        })

    matched = len(pairs)
    extras = []
    matched_g = set(pairs.values())
    for gi, g in enumerate(generated):
        if gi in matched_g:
            continue
        body = g["topic"] + g["client_request"] + g["our_response"]
        if chitchat_keywords and _hits(chitchat_keywords, body):
            kind = "chitchat"
        elif any(_hits(e["topic_keywords"], g["topic"]) for e in expected):
            kind = "duplicate"
        else:
            kind = "other"
        extras.append({"kind": kind, "topic": g["topic"], "status": g["status"]})

    n = len(expected)
    return {
        "recall": matched / n if n else None,
        "precision": matched / len(generated) if generated else None,
        "status_accuracy": status_ok / matched if matched else None,
        "status_accuracy_all": status_ok / n if n else None,
        "changed_status_accuracy": changed_ok / changed_total if changed_total else None,
        "values_accuracy": values_hit / values_total if values_total else None,
        "matched": matched, "expected": n, "generated": len(generated),
        "extras": extras,
        "extras_by_kind": {k: sum(1 for x in extras if x["kind"] == k) for k in ("chitchat", "duplicate", "other")},
        "details": details,
    }


YEAR_RE = re.compile(r"(20\d\d)")


def score_actions(expected: list[dict], generated: list[dict], valid_year: str = "2026") -> dict:
    scores = []
    for ei, e in enumerate(expected):
        for gi, g in enumerate(generated):
            h = _hits(e["content_keywords"], g["content"])
            if h:
                scores.append((ei, gi, h))
    pairs = _greedy_assign(scores)
    owner_ok = deadline_ok = 0
    details = []
    for ei, e in enumerate(expected):
        gi = pairs.get(ei)
        g = generated[gi] if gi is not None else None
        o = d = False
        if g:
            o = g["owner_label"] == ("自社" if e["owner"] == "our_side" else "相手方")
            d = _hits(e["deadline_accept"], g["deadline"]) > 0
            owner_ok += int(o)
            deadline_ok += int(d)
        details.append({"id": e["id"], "matched": g is not None, "owner_correct": o, "deadline_correct": d,
                        "expected_deadline": e["deadline_text"], "generated_deadline": g["deadline"] if g else None})
    bad_years = sorted({y for g in generated for y in YEAR_RE.findall(norm_loose(g["deadline"])) if y != valid_year})
    matched = len(pairs)
    n = len(expected)
    return {
        "recall": matched / n if n else None,
        "owner_accuracy": owner_ok / matched if matched else None,
        "deadline_accuracy": deadline_ok / matched if matched else None,
        "matched": matched, "expected": n, "generated": len(generated),
        "fabricated_years": bad_years,
        "details": details,
    }


def score_minutes(minutes: dict, expected: dict) -> dict:
    d = score_discussions(expected["discussions"], minutes["discussions"], expected.get("chitchat_keywords", []))
    a = score_actions(expected["action_items"], minutes["action_items"])
    return {"discussions": d, "actions": a}


def summary_row(s: dict) -> dict:
    """集計表用に主要指標だけを取り出す"""
    d, a = s["discussions"], s["actions"]
    return {
        "d_recall": d["recall"], "d_precision": d["precision"], "d_status": d["status_accuracy"],
        "d_status_changed": d["changed_status_accuracy"], "d_values": d["values_accuracy"],
        "d_extra_chitchat": d["extras_by_kind"]["chitchat"], "d_extra_dup": d["extras_by_kind"]["duplicate"],
        "d_extra_other": d["extras_by_kind"]["other"], "d_generated": d["generated"], "d_expected": d["expected"],
        "a_recall": a["recall"], "a_owner": a["owner_accuracy"], "a_deadline": a["deadline_accuracy"],
        "a_bad_years": ",".join(a["fabricated_years"]),
    }
