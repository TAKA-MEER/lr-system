"""VOICEVOX engine (http://localhost:50021) への薄いクライアント。"""
import httpx


class VoicevoxClient:
    def __init__(self, base_url: str = "http://localhost:50021"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    def synthesize(self, text: str, speaker_id: int, speed_scale: float = 1.0) -> bytes:
        """テキストを音声合成し、WAVバイト列を返す。"""
        query_resp = self._client.post(
            f"{self.base_url}/audio_query",
            params={"text": text, "speaker": speaker_id},
        )
        query_resp.raise_for_status()
        query = query_resp.json()
        query["speedScale"] = speed_scale

        synth_resp = self._client.post(
            f"{self.base_url}/synthesis",
            params={"speaker": speaker_id},
            json=query,
        )
        synth_resp.raise_for_status()
        return synth_resp.content

    def close(self):
        self._client.close()
