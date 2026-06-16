import logging
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router
from api.websocket import router as ws_router
from stt.transcriber import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時: STTモデルをロード
    cfg = load_config()
    logger.info("STTモデルを初期化中...")
    transcriber = Transcriber(
        model_size=cfg["stt"]["model_size"],
        device=cfg["stt"]["device"],
        compute_type=cfg["stt"]["compute_type"],
    )
    app.state.transcriber = transcriber
    app.state.config = cfg
    logger.info("STTモデルの初期化完了")
    yield
    # 終了時: VRAMを解放
    logger.info("STTモデルを解放中...")
    transcriber.unload()


app = FastAPI(title="立会試験議事録システム", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(api_router, prefix="/api")

# 静的ファイル（index.html等）を最後にマウント
app.mount("/", StaticFiles(directory="static", html=True), name="static")
