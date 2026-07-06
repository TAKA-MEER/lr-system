"""生成された議事録docxを解析し、正解データ(expected_minutes_v1.yaml)と突き合わせて
協議事項・アクションアイテムの再現率/正確性、状態矛盾解消の正しさを評価する。"""
from docx import Document


def parse_minutes_docx(path: str) -> dict:
    doc = Document(path)
    result = {"info": {}, "discussions": [], "action_items": []}

    for table in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        header = rows[0]

        if header and header[0] in ("試験名",) or (len(rows) >= 1 and rows[0][0] == "試験名"):
            for r in rows:
                if len(r) >= 2:
                    result["info"][r[0]] = r[1]
            continue

        if header[:2] == ["No", "議題"]:
            for r in rows[1:]:
                if len(r) >= 5:
                    result["discussions"].append(
                        {"topic": r[1], "client_request": r[2], "our_response": r[3], "status": r[4]}
                    )
            continue

        if header[:2] == ["No", "対応内容"]:
            for r in rows[1:]:
                if len(r) >= 4:
                    result["action_items"].append(
                        {"content": r[1], "owner_label": r[2], "deadline": r[3]}
                    )
            continue

    return result


def _keyword_hits(keywords: list[str], *texts: str) -> int:
    joined = " ".join(t or "" for t in texts)
    return sum(1 for kw in keywords if kw in joined)


def _best_match(expected_keywords_sets: list[list[str]], candidates: list[dict], text_fields: list[str]) -> tuple[int, int]:
    """expected 1件に対し、最もキーワードヒット数が多い candidate のインデックスとヒット数を返す。"""
    best_idx, best_hits = -1, 0
    for idx, cand in enumerate(candidates):
        texts = [cand.get(f, "") for f in text_fields]
        hits = sum(_keyword_hits(kws, *texts) for kws in expected_keywords_sets)
        if hits > best_hits:
            best_idx, best_hits = idx, hits
    return best_idx, best_hits


def evaluate_discussions(expected: list[dict], generated: list[dict]) -> dict:
    results = []
    matched_count = 0
    status_correct_count = 0
    for exp in expected:
        kw_sets = [exp["topic_keywords"], exp["client_request_keywords"], exp["our_response_keywords"]]
        kw_sets = [k for k in kw_sets if k]
        idx, hits = _best_match(kw_sets, generated, ["topic", "client_request", "our_response"])
        matched = idx >= 0 and hits >= 1
        status_ok = False
        gen_status = None
        if matched:
            matched_count += 1
            gen_status = generated[idx]["status"]
            status_ok = gen_status == exp["expected_status"]
            status_correct_count += int(status_ok)
        results.append(
            {
                "id": exp["id"],
                "matched": matched,
                "matched_generated_idx": idx if matched else None,
                "expected_status": exp["expected_status"],
                "generated_status": gen_status,
                "status_correct": status_ok,
            }
        )
    total = len(expected)
    return {
        "recall": matched_count / total if total else None,
        "status_accuracy_among_matched": (status_correct_count / matched_count) if matched_count else None,
        "matched_count": matched_count,
        "total": total,
        "details": results,
        "hallucination_candidates": [
            g for i, g in enumerate(generated) if i not in {r["matched_generated_idx"] for r in results if r["matched"]}
        ],
    }


def evaluate_action_items(expected: list[dict], generated: list[dict]) -> dict:
    results = []
    matched_count = 0
    owner_correct = 0
    deadline_correct = 0
    for exp in expected:
        idx, hits = _best_match([exp["content_keywords"]], generated, ["content"])
        matched = idx >= 0 and hits >= 1
        owner_ok = deadline_ok = False
        gen_owner_label = gen_deadline = None
        if matched:
            matched_count += 1
            cand = generated[idx]
            gen_owner_label = cand["owner_label"]
            gen_deadline = cand["deadline"]
            expected_owner_label = "自社" if exp["owner"] == "our_side" else "相手方"
            owner_ok = gen_owner_label == expected_owner_label
            deadline_ok = any(kw in (gen_deadline or "") for kw in exp["deadline_keywords"])
            owner_correct += int(owner_ok)
            deadline_correct += int(deadline_ok)
        results.append(
            {
                "id": exp["id"],
                "matched": matched,
                "owner_correct": owner_ok,
                "deadline_correct": deadline_ok,
                "generated_owner": gen_owner_label,
                "generated_deadline": gen_deadline,
            }
        )
    total = len(expected)
    return {
        "recall": matched_count / total if total else None,
        "owner_accuracy_among_matched": (owner_correct / matched_count) if matched_count else None,
        "deadline_accuracy_among_matched": (deadline_correct / matched_count) if matched_count else None,
        "matched_count": matched_count,
        "total": total,
        "details": results,
    }


def evaluate_status_conflict(check: dict, generated_discussions: list[dict]) -> dict:
    kw = check["topic_keywords"]
    candidates = [d for d in generated_discussions if _keyword_hits(kw, d.get("topic", "")) >= 1]
    if not candidates:
        return {"found": False, "correct": False, "generated_status": None}
    status = candidates[0]["status"]
    return {"found": True, "correct": status == check["expected_final_status"], "generated_status": status}
