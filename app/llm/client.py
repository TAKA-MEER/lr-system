import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def chat(self, model: str, prompt: str, max_tokens: int = 4096, temperature: float = 0.2, num_ctx: int = 32768) -> str:
        """Ollama /api/generate を呼んでテキストを返す"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # thinking対応モデル(qwen3系等)でも思考ブロックを出力させず、
                             # 素直にJSONだけを出力させる(トークン上限到達によるJSON途中切れを防ぐ)
            "options": {"num_predict": max_tokens, "temperature": temperature, "num_ctx": num_ctx},
            # num_ctx未指定時はOllamaの既定コンテキスト長が使われ、入力プロンプトだけで
            # それを超過すると出力がほぼ生成されない(JSON解析失敗)不具合が3時間耐久試験で
            # 判明したため、明示的に大きめの値を指定する(testing/CHANGELOG.md参照)。
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
            except httpx.HTTPError as e:
                logger.error(f"Ollama APIエラー: {e}")
                raise

    async def unload_model(self, model: str):
        """モデルをVRAMからアンロードする"""
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "keep_alive": 0}
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(url, json=payload)
        logger.info(f"Ollamaモデルをアンロード: {model}")

    async def is_ready(self) -> bool:
        """Ollamaが起動しているか確認"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
