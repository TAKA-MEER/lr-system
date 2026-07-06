"""
scenario/script_v1.yaml の台本からVOICEVOXで音声を合成し、
internal_mix.wav (両話者混合) と bt_mix.wav (BTイヤホン装着者の発話区間のみ音声あり)
を生成する。

前提: VOICEVOX engine が http://localhost:50021 で起動していること
    docker run --rm -d --name voicevox-tts -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu20.04-latest

使い方:
    .venv/Scripts/python.exe tts/generate_audio.py
"""
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voicevox_client import VoicevoxClient

TESTING_DIR = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml(TESTING_DIR / "config.yaml")
    script = load_yaml(TESTING_DIR / cfg["paths"]["script"])
    voice_map = load_yaml(TESTING_DIR / cfg["paths"]["voice_map"])["voices"]

    bt_wearer = script["meta"]["bt_wearer"]
    gap_lo, gap_hi = cfg["audio"]["turn_gap_seconds"]

    out_dir = TESTING_DIR / cfg["paths"]["audio_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    client = VoicevoxClient(base_url=cfg["voicevox"]["base_url"])

    internal_chunks = []
    bt_chunks = []
    timeline = []
    cursor = 0.0
    samplerate = None

    turns = script["turns"]
    for i, turn in enumerate(turns):
        speaker = turn["speaker"]
        role = script["speakers"][speaker]
        voice = voice_map[speaker]

        print(f"[{i+1}/{len(turns)}] synthesizing turn {turn['id']} ({speaker}): {turn['text'][:20]}...")
        wav_bytes = client.synthesize(
            turn["text"], speaker_id=voice["speaker_id"], speed_scale=voice.get("speedScale", 1.0)
        )
        samples, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samplerate is None:
            samplerate = sr
        elif sr != samplerate:
            raise RuntimeError(f"サンプルレート不一致: {sr} != {samplerate}")

        duration = len(samples) / samplerate

        internal_chunks.append(samples)
        if speaker == bt_wearer:
            bt_chunks.append(samples)
        else:
            bt_chunks.append(np.zeros_like(samples))

        timeline.append(
            {
                "turn_id": turn["id"],
                "speaker": speaker,
                "role": role,
                "is_bt_wearer": speaker == bt_wearer,
                "start_sec": round(cursor, 3),
                "end_sec": round(cursor + duration, 3),
            }
        )

        gap = random.uniform(gap_lo, gap_hi)
        cursor += duration + gap
        if i < len(turns) - 1:
            silence = np.zeros(int(gap * samplerate), dtype="float32")
            internal_chunks.append(silence)
            bt_chunks.append(silence)

    client.close()

    internal_audio = np.concatenate(internal_chunks)
    bt_audio = np.concatenate(bt_chunks)
    assert len(internal_audio) == len(bt_audio), "internal/btの長さが一致しません"

    internal_path = out_dir / "internal_mix.wav"
    bt_path = out_dir / "bt_mix.wav"
    timeline_path = out_dir / "timeline.json"

    sf.write(internal_path, internal_audio, samplerate)
    sf.write(bt_path, bt_audio, samplerate)
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(
            {"samplerate": samplerate, "total_duration_sec": round(cursor, 3), "turns": timeline},
            f,
            ensure_ascii=False,
            indent=2,
        )

    total_min = cursor / 60
    print(f"\n完了: {internal_path}")
    print(f"完了: {bt_path}")
    print(f"完了: {timeline_path}")
    print(f"総尺: {total_min:.1f}分 ({cursor:.1f}秒)")


if __name__ == "__main__":
    main()
