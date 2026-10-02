"""Ollama で think:true と format(JSONスキーマ)を併用できるかの確認"""
import json
import httpx
schema = {"type": "object", "properties": {"topics": {"type": "array", "items": {"type": "string"}}}, "required": ["topics"]}
prompt = '次の話題名のうち同じ話題をまとめ、まとめた話題名の一覧を {"topics": [...]} の形のJSONのみで出力: 絶縁抵抗試験, 絶縁抵抗の再測定, 耐電圧試験, A盤の絶縁抵抗, B盤の絶縁抵抗'
for think in (True,):
    r = httpx.post("http://minutes-ollama:11434/api/generate", timeout=600, json={
        "model": "qwen3.5:9b", "prompt": prompt, "stream": False, "think": think, **({} if think else {"format": schema}),
        "options": {"num_predict": 3000, "temperature": 0.2, "num_ctx": 8192}}).json()
    print(json.dumps({"think": think, "response": r.get("response", "")[:300], "thinking_chars": len(r.get("thinking") or ""),
                      "eval_count": r.get("eval_count"), "done_reason": r.get("done_reason"),
                      "sec": round(r.get("total_duration", 0) / 1e9, 1), "error": r.get("error")}, ensure_ascii=False), flush=True)
