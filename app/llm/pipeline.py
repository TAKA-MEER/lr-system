"""文字起こし全文から議事録JSONを作る2段階LLM処理。

routes.generate_minutes と、testing/ の試験スクリプト(コンテナ内で直接呼ぶ)の両方から使う。
"""
import json
import logging
import re
import time

from llm.client import OllamaClient
from llm.prompts import build_cluster_prompt, build_stage1_prompt, build_stage2_prompt
from postprocess.chunker import split_transcript
from postprocess.merge import merge_actions, merge_discussions

logger = logging.getLogger(__name__)

STATUSES = ["合意済み", "継続協議", "保留"]
_DISCUSSION = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "client_request": {"type": "string"},
        "our_response": {"type": "string"},
        "status": {"type": "string", "enum": STATUSES},
    },
    "required": ["topic", "client_request", "our_response", "status"],
}
_ACTION = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "owner": {"type": "string", "enum": ["our_side", "client"]},
        "deadline": {"type": "string"},
    },
    "required": ["content", "owner", "deadline"],
}
STAGE1_SCHEMA = {
    "type": "object",
    "properties": {"discussions": {"type": "array", "items": _DISCUSSION},
                   "action_items": {"type": "array", "items": _ACTION}},
    "required": ["discussions", "action_items"],
}
CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {"groups": {"type": "array", "items": {
        "type": "object",
        "properties": {"topic": {"type": "string"}, "members": {"type": "array", "items": {"type": "integer"}}},
        "required": ["topic", "members"]}}},
    "required": ["groups"],
}
STAGE2_SCHEMA = {
    "type": "object",
    "properties": {
        "trial_name": {"type": "string"},
        "date": {"type": "string"},
        "location": {"type": "string"},
        "attendees": {"type": "object",
                      "properties": {"client": {"type": "array", "items": {"type": "string"}},
                                     "our_side": {"type": "array", "items": {"type": "string"}}},
                      "required": ["client", "our_side"]},
        "discussions": {"type": "array", "items": _DISCUSSION},
        "action_items": {"type": "array", "items": _ACTION},
    },
    "required": ["trial_name", "date", "location", "attendees", "discussions", "action_items"],
}


def extract_json(raw: str) -> str:
    # ```json ... ``` のようなコードフェンスで囲われる場合があるため除去する
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        return m.group(1).strip()
    # 前後に説明文が付与された場合に備え、最初の{から最後の}までを抜き出す
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


def _try_parse(raw: str) -> dict | None:
    try:
        d = json.loads(extract_json(raw))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def fallback_minutes(meta: dict, raw: str, summaries: list[dict] | None = None) -> dict:
    """Stage2が使えなかった場合の議事録。Stage1の結果があれば、話題名で単純に統合して使う
    (以前は協議事項・対応事項が空の議事録になっていた)。同じ話題は後に出てきた状態で上書きする。"""
    discussions: dict[str, dict] = {}
    actions: list[dict] = []
    for s in summaries or []:
        for d in s.get("discussions", []) or []:
            if isinstance(d, dict) and d.get("topic"):
                key = re.sub(r"\s", "", d["topic"])
                prev = discussions.get(key, {})
                discussions[key] = {k: (d.get(k) or prev.get(k) or "") for k in ("topic", "client_request", "our_response", "status")}
        for a in s.get("action_items", []) or []:
            if isinstance(a, dict) and a.get("content") and all(a["content"] != x["content"] for x in actions):
                actions.append(a)
    return {
        "trial_name": meta.get("trial_name", ""),
        "date": meta.get("date", ""),
        "location": meta.get("location", ""),
        "attendees": meta.get("attendees", {"client": [], "our_side": []}),
        "discussions": list(discussions.values()),
        "action_items": actions,
        "raw": raw,
    }


async def build_minutes(full_text: str, meta: dict, llm_cfg: dict, ollama: OllamaClient) -> tuple[dict, dict]:
    """(議事録JSON, 処理情報) を返す。処理情報は試験での計測用。"""
    model = llm_cfg["model"]
    chunk_chars = llm_cfg["chunk_chars"]
    temperature = llm_cfg.get("temperature", 0.2)
    num_ctx = llm_cfg.get("num_ctx", 32768)
    max_tokens = llm_cfg.get("max_tokens", 4096)
    stage2_max_tokens = llm_cfg.get("stage2_max_tokens", max_tokens)
    num_gpu = llm_cfg.get("num_gpu")
    structured = llm_cfg.get("structured_output", True)
    info = {"stage1_chunks": 0, "stage1_seconds": 0.0, "stage2_seconds": 0.0, "stage2_parse_ok": True,
            "stage1_parse_fail": 0, "stage2_retries": 0}

    async def call(prompt: str, schema: dict, tokens: int) -> str:
        return await ollama.chat(model=model, prompt=prompt, max_tokens=tokens, temperature=temperature,
                                 num_ctx=num_ctx, num_gpu=num_gpu, json_schema=schema if structured else None)

    # Stage 1: チャンクごとに要点JSON抽出
    chunks = split_transcript(full_text, max_chars=chunk_chars)
    info["stage1_chunks"] = len(chunks)
    logger.info(f"Stage1: {len(chunks)} チャンクを処理中...")
    summaries_raw, summaries = [], []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        raw = await call(build_stage1_prompt(chunk), STAGE1_SCHEMA, max_tokens)
        parsed = _try_parse(raw)
        if parsed is None:
            info["stage1_parse_fail"] += 1
            logger.warning(f"Stage1 [{i+1}] JSON解析失敗。raw={raw[:500]!r}")
        summaries_raw.append(json.dumps(parsed, ensure_ascii=False) if parsed else raw)
        if parsed:
            summaries.append(parsed)
        logger.info(f"  Stage1 [{i+1}/{len(chunks)}] 完了")
    info["stage1_seconds"] = round(time.time() - t0, 1)

    # 話題名だけをLLMに渡してグループ分けさせ、グループごとにプログラムで統合してからStage2に渡す
    stage2_inputs = summaries_raw
    if llm_cfg.get("topic_cluster", False) and summaries:
        items = [d for sm in summaries for d in (sm.get("discussions") or []) if isinstance(d, dict) and d.get("topic")]
        t1 = time.time()
        raw_c = await call(build_cluster_prompt([d["topic"] for d in items]), CLUSTER_SCHEMA, max_tokens)
        parsed_c = _try_parse(raw_c)
        info["cluster_seconds"] = round(time.time() - t1, 1)
        info["cluster_ok"] = parsed_c is not None
        merged_ds = None
        if parsed_c:
            seen: set[int] = set()
            merged_ds = []
            for g in parsed_c.get("groups", []):
                members = [m for m in g.get("members", []) if isinstance(m, int) and 0 <= m < len(items) and m not in seen]
                seen.update(members)
                if not members:
                    continue
                members.sort()  # 時系列順に統合する(状態は後の記述を優先)
                d = merge_discussions([dict(items[m], topic=g.get("topic") or items[m]["topic"]) for m in members])
                merged_ds += d[:1]
            merged_ds += [dict(items[i]) for i in range(len(items)) if i not in seen]  # グループ漏れはそのまま
        else:
            merged_ds = merge_discussions(items)  # グループ分けに失敗したらプログラムの名寄せで代用
        acts = merge_actions([a for sm in summaries for a in (sm.get("action_items") or [])])
        info["stage1_topics_total"] = len(items)
        info["merged_topics"] = len(merged_ds)
        logger.info(f"話題のグループ分け: {len(items)}件 → {len(merged_ds)}件")
        stage2_inputs = [json.dumps({"discussions": merged_ds, "action_items": acts}, ensure_ascii=False)]

    # Stage 2: 要点JSONを統合して最終議事録JSON生成(解析できなければ1回だけやり直す)
    logger.info("Stage2: 最終議事録を生成中...")
    stage2_prompt = build_stage2_prompt(stage2_inputs, meta)
    info["stage2_prompt_chars"] = len(stage2_prompt)
    t0 = time.time()
    minutes_json = None
    raw = ""
    for attempt in range(2):
        raw = await call(stage2_prompt, STAGE2_SCHEMA, stage2_max_tokens)
        minutes_json = _try_parse(raw)
        if minutes_json is not None:
            break
        info["stage2_retries"] += 1
        logger.warning(f"JSON解析失敗(Stage2 {attempt + 1}回目)。raw末尾={raw[-300:]!r}")
    info["stage2_seconds"] = round(time.time() - t0, 1)
    info["stage2_output_chars"] = len(raw)

    if minutes_json is None:
        logger.warning(f"JSON解析失敗。Stage1の結果を統合したフォールバックを使用。raw={raw[:2000]!r}")
        info["stage2_parse_ok"] = False
        minutes_json = fallback_minutes(meta, raw, summaries)

    return minutes_json, info
