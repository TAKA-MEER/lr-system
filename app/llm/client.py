import logging
import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def chat(self, model: str, prompt: str, max_tokens: int = 4096, temperature: float = 0.2, num_ctx: int = 32768,
                   num_gpu: int | None = None, json_schema: dict | None = None, timeout: float = 600.0,
                   think: bool = False) -> str:
        """Ollama /api/generate を呼んでテキストを返す"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,  # 既定False: thinking対応モデル(qwen3系等)でも思考ブロックを出力させず、
                             # 素直にJSONだけを出力させる(トークン上限到達によるJSON途中切れを防ぐ)
            "options": {"num_predict": max_tokens, "temperature": temperature, "num_ctx": num_ctx},
            # num_ctx未指定時はOllamaの既定コンテキスト長が使われ、入力プロンプトだけで
            # それを超過すると出力がほぼ生成されない(JSON解析失敗)不具合が3時間耐久試験で
            # 判明したため、明示的に大きめの値を指定する(testing/CHANGELOG.md参照)。
        }
        if json_schema is not None:
            # 構造化出力: 出力をJSONスキーマに沿う形に制約する。引用符の抜けなどで
            # JSONが壊れて議事録全体が空になる不具合への対策(第2期試験)
            payload["format"] = json_schema
        if num_gpu is not None:
            # GPUに載せる層数。未指定だとOllamaが空きVRAMに余裕を残そうとして一部の層をCPUに
            # 逃がすことがあり、生成が数倍遅くなる(testing/CHANGELOG.md 第2期試験参照)
            payload["options"]["num_gpu"] = num_gpu
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
            except httpx.HTTPError as e:
                if num_gpu is None:
                    logger.error(f"Ollama APIエラー: {e}")
                    raise
                # VRAMが足りずに全層をGPUに載せられなかった場合など。Ollamaの自動配置でやり直す
                logger.warning(f"Ollama APIエラー(num_gpu={num_gpu})。GPU層数の指定なしで再試行します: {e}")
                payload["options"].pop("num_gpu")
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")

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
