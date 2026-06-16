import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def chat(self, model: str, prompt: str, max_tokens: int = 4096) -> str:
        """Ollama /api/generate を呼んでテキストを返す"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
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
