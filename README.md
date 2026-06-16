# LR-system

-**立会試験 議事録システム**-

工場での立会試験の音声をリアルタイムで文字起こしし、試験終了後すぐにAIが議事録を自動生成するシステムです。  
すべての処理はローカルで完結し、外部ネットワークへのデータ送信は行いません。

---

## 目次

**使い方（前半）**

1. [動作環境](#1-動作環境)
2. [初回セットアップ](#2-初回セットアップ)
3. [試験前の準備](#3-試験前の準備)
4. [試験当日の操作手順](#4-試験当日の操作手順)
5. [議事録の生成とダウンロード](#5-議事録の生成とダウンロード)
6. [設定の変更](#6-設定の変更)
7. [よくあるトラブル](#7-よくあるトラブル)

**技術詳細（後半）**

8. [システム構成](#8-システム構成)
9. [音声処理パイプライン](#9-音声処理パイプライン)
10. [話者判定ロジック](#10-話者判定ロジック)
11. [LLM処理パイプライン](#11-llm処理パイプライン)
12. [VRAM管理](#12-vram管理)
13. [WebSocket通信プロトコル](#13-websocket通信プロトコル)
14. [ファイル構成と各ファイルの役割](#14-ファイル構成と各ファイルの役割)
15. [議事録フォーマットの変更方法](#15-議事録フォーマットの変更方法)
16. [ログの見方と問題診断](#16-ログの見方と問題診断)

---

# 使い方

---

## 1. 動作環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11 |
| GPU | NVIDIA RTX 3060 Laptop（VRAM 6GB）以上 |
| RAM | 16GB以上推奨（WSL2に12GB割り当て） |
| ストレージ | 20GB以上の空き（モデルファイル込み） |
| ソフトウェア | Docker Desktop（WSL2バックエンド）、NVIDIAドライバー最新版 |
| ブラウザ | Google Chrome / Microsoft Edge（最新版） |

---

## 2. 初回セットアップ

### 2-1. WSL2のメモリ設定

Windowsのホームディレクトリに `.wslconfig` ファイルを作成します。

```
C:\Users\<ユーザー名>\.wslconfig
```

```ini
[wsl2]
memory=12GB
processors=8
```

作成後、PowerShellで WSL2 を再起動します。

```powershell
wsl --shutdown
```

### 2-2. Docker Desktopの設定確認

Docker Desktop を起動し、以下を確認します。

- Settings → General → 「Use the WSL 2 based engine」が有効
- Settings → Resources → WSL Integration → 使用するディストリビューションが有効

### 2-3. システムの起動

PowerShell またはコマンドプロンプトで実行します。

```powershell
cd minutes-system
docker compose up --build
```

**初回起動時の注意**  
Ollamaコンテナが `qwen2.5:7b`（約4.4GB）を自動ダウンロードします。  
インターネット接続のある環境で実行してください。完了後はオフライン環境でも動作します。  
ダウンロードには10〜30分程度かかります。

### 2-4. 起動確認

以下のログが表示されれば起動完了です。

```
minutes-app  | STTモデルを初期化中...
minutes-app  | STTモデルの初期化完了
minutes-app  | Application startup complete.
```

ブラウザで `http://localhost:8000` を開いて画面が表示されることを確認します。

---

## 3. 試験前の準備

### 3-1. マイクの接続

試験開始前に以下を確認します。

- BTイヤホンを Bluetooth でペアリングする
- Windowsの「サウンドの設定」→「入力」でBTイヤホンが録音デバイスとして表示されていることを確認する
- BTイヤホンが表示されない場合は、いったん切断して再ペアリングする（通話モード/HFPモードで接続されることを確認）

### 3-2. 2つのウィンドウを開く

| ウィンドウ | URL | 役割 |
|-----------|-----|------|
| メイン（Window 1） | `http://localhost:8000` | 内蔵マイクの録音・文字起こし表示・議事録生成 |
| BTウィンドウ（Window 2） | メイン画面の「BTウィンドウを開く」ボタンで開く | BTイヤホンの録音のみ |

> **重要**  
> 必ずメイン画面の「BTウィンドウを開く」ボタンを使って開いてください。  
> セッションIDが自動で引き継がれます。

### 3-3. マイクの選択

**メイン画面（Window 1）**

1. 「マイク一覧を取得」ボタンを押す
2. ブラウザのマイク許可ダイアログで「許可」を選ぶ
3. PC内蔵マイクをドロップダウンから選択する

**BTウィンドウ（Window 2）**

1. 「マイク一覧を取得」ボタンを押す
2. BTイヤホンをドロップダウンから選択する

### 3-4. 試験情報の入力

メイン画面のサイドバーに以下を入力します。

- **試験名**：例「〇〇装置 第1回立会試験」
- **場所**：例「第1試験室」
- **相手方参加者**：カンマ区切りで入力（例「山田 太郎, 鈴木 一郎」）
- **自社参加者**：カンマ区切りで入力

---

## 4. 試験当日の操作手順

```
① BTウィンドウ（Window 2）で「録音開始」を押す

② メイン画面（Window 1）で「録音開始」を押す

③ 試験を実施する
   → 文字起こしが画面右側にリアルタイムで表示される

④ 試験終了後、両方のウィンドウで「停止」を押す
```

### 話者ラベルの見方

| ラベル | 色 | 意味 |
|--------|------|------|
| our_side | 青 | 自社（BTイヤホンを装着した試験員の発言） |
| client | オレンジ | 相手方の発言 |

### 話者判定がずれる場合

サイドバーの「BTマイク 音量閾値」スライダーを調整します。

- 相手方の発言が our_side と表示される → **閾値を上げる**
- 自社の発言が client と表示される → **閾値を下げる**

閾値はリアルタイムで反映されます。録音中でも調整可能です。

---

## 5. 議事録の生成とダウンロード

1. 「停止」ボタンで録音を終了する
2. メイン画面下部の「📄 議事録を生成」ボタンを押す
3. 生成中はボタン横に進捗が表示される（LLM処理で2〜5分かかる場合があります）
4. 完了後「⬇ ダウンロード」ボタンが有効になる
5. ボタンを押すと `.docx` ファイルがダウンロードされる

生成されたファイルはコンテナ内の `/output/` フォルダにも保存されます。  
ホスト側では `minutes-system/output/` から直接取得することもできます。

---

## 6. 設定の変更

`app/config/settings.yaml` を編集し、コンテナを再起動すると反映されます。

```yaml
stt:
  model_size: large-v3   # large-v3（高精度）/ medium（高速）/ small（最速）
  chunk_seconds: 5       # 文字起こしの処理間隔（秒）

speaker:
  bt_rms_threshold: 800  # BTマイクの話者判定閾値（初期値）

llm:
  model: qwen2.5:7b      # 使用するLLMモデル
  chunk_chars: 2000      # LLMに一度に渡すテキストの長さ
```

設定変更後の再起動：

```powershell
docker compose restart minutes-app
```

---

## 7. よくあるトラブル

### 起動しない / コンテナがすぐ終了する

```powershell
# ログを確認する
docker compose logs minutes-app
docker compose logs minutes-ollama
```

GPUが認識されていない場合：

```powershell
# ホストでGPUが見えているか確認
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### マイクが1件しか表示されない

BTイヤホンが録音デバイスとして認識されていません。

1. タスクバーの🔊を右クリック →「サウンドの設定」
2. 「入力」セクションにBTイヤホンが表示されているか確認
3. 表示されていない場合はBTイヤホンを切断→再ペアリング

### 文字起こしが空になる / 精度が低い

- マイクがPCから離れすぎていないか確認する
- `settings.yaml` の `vad_threshold` を下げる（デフォルト `0.3` → `0.2`）
- モデルを `large-v3` に設定しているか確認する（`medium` や `small` は精度が下がる）

### 議事録生成でエラーが出る

- Ollamaが起動しているか確認する：`docker compose logs minutes-ollama`
- VRAMが足りない可能性：ブラウザをリロードしてコンテナを再起動する

### 生成された議事録の内容がおかしい

`app/llm/prompts.py` の Stage 1 / Stage 2 プロンプトを調整してください。  
詳細は[LLM処理パイプライン](#11-llm処理パイプライン)を参照してください。

---

---

# 技術詳細

---

## 8. システム構成

### コンテナ構成

```
Windows ホスト
│
├── Docker コンテナ: minutes-app  (port 8000)
│   ├── FastAPI（WebサーバーおよびWebSocket）
│   ├── faster-whisper（音声文字起こし）
│   └── 議事録生成ロジック
│
├── Docker コンテナ: minutes-ollama  (port 11434)
│   └── Ollama（LLM APIサーバー）
│
├── ボリューム: ./output    → 生成済み議事録
├── ボリューム: ./models    → HuggingFaceモデルキャッシュ
└── ボリューム: ollama-data → Ollamaモデルキャッシュ
```

2コンテナを分離している理由は VRAM の逐次管理のためです。  
STT（文字起こし）と LLM は同時にVRAMに乗らないため、処理が終わったモデルを解放してから次のモデルをロードします。

### ネットワーク

```
ブラウザ（Windows）
  ↕ HTTP/WebSocket  port 8000
minutes-app コンテナ
  ↕ HTTP  port 11434
minutes-ollama コンテナ
```

両コンテナは Docker 内部ネットワーク（`minutes-system_default`）で通信します。  
Ollamaの 11434 ポートはホストにも公開されていますが、外部からのアクセスが不要な場合は  
`docker-compose.yml` の `ports` を `expose` に変更してください。

---

## 9. 音声処理パイプライン

### 音声の流れ

```
ブラウザ（MediaRecorder）
  │  WebM/Opus  5秒チャンク
  │  WebSocket バイナリフレーム
  ↓
app/api/websocket.py  （受信）
  ↓
app/stt/transcriber.py
  │  ffmpeg（WebM → 16kHz モノラル WAV）
  ↓
faster-whisper large-v3
  │  VADフィルター → 文字起こし
  ↓
WebSocket テキストフレーム（JSON）
  ↓
ブラウザ（画面に表示）
```

### MediaRecorder の設定

`index.html` 内で以下のように設定されています。

```javascript
const recorder = new MediaRecorder(stream, {
    mimeType: 'audio/webm;codecs=opus'
});
recorder.start(5000); // 5000ms = 5秒ごとにチャンクを送信
```

5秒ごとにブラウザが音声データ（WebMブロブ）を生成し、WebSocket経由でサーバーに送信します。  
この間隔は `index.html` の `recorder.start(5000)` の数値で変更できます。  
短くするとリアルタイム性が上がりますが、文字起こしの精度が下がります。

### ffmpeg による変換

faster-whisper は WAV（PCM）形式を要求します。  
ブラウザから送られてくる WebM/Opus を ffmpeg でリアルタイム変換します。

```python
subprocess.run([
    "ffmpeg", "-y",
    "-i", "pipe:0",      # 標準入力から受け取る
    "-ar", "16000",      # サンプリングレート 16kHz（Whisper推奨）
    "-ac", "1",          # モノラル
    "-f", "wav",
    "pipe:1",            # 標準出力に出力
], input=audio_bytes, capture_output=True)
```

### VAD（音声区間検出）フィルター

工場ノイズ対策として faster-whisper 内蔵の Silero-VAD を使用します。

```yaml
# settings.yaml
stt:
  vad_threshold: 0.3         # 低いほど小さな音も拾う（0.1〜0.9）
  vad_min_silence_ms: 300    # この時間以上の無音を区切りとみなす
  vad_speech_pad_ms: 300     # 発話区間の前後にパディングを追加
```

工場ノイズが大きい場合は `vad_threshold` を `0.4`〜`0.5` に上げると  
ノイズを音声と誤認識しにくくなります。

---

## 10. 話者判定ロジック

### 基本原理

BTイヤホン（自社試験員が装着）のマイクに音声が入っていれば our_side、  
無音であれば client が話していると判定します。

```
BTマイクのRMS値 > 閾値  →  our_side
BTマイクのRMS値 ≤ 閾値  →  client
```

### RMS（二乗平均平方根）の計算

RMS は音量の物理的な指標です。値が大きいほど音量が大きいことを示します。

```python
audio = numpy.frombuffer(wav_pcm_bytes, dtype=numpy.int16).astype(numpy.float32)
rms = float(numpy.sqrt(numpy.mean(audio ** 2)))
```

int16 PCM データの場合、無音時は RMS ≈ 0、通常会話時は RMS ≈ 500〜3000 程度になります。

### 時間窓による判定

内蔵マイクの5秒チャンクと、BTマイクのRMS記録を時刻で突き合わせます。

```
内蔵マイクチャンク（T=10.0 〜 T=15.0）
  ↓
SpeakerDetector.get_speaker(start_time=10.0, end_time=15.0)
  ↓
BTマイクのRMS履歴から T=10.0〜15.0 のレコードを取り出す
  ↓
その中の最大RMSが閾値を超えていれば our_side、超えていなければ client
```

### RMS履歴の管理

```python
# app/stt/speaker.py
self._history: deque[RMSRecord] = deque()

# BTチャンクを受信するたびに追加
self._history.append(RMSRecord(timestamp=time.time(), rms=rms))

# window_seconds * 2 より古いレコードは自動削除
```

### 閾値のチューニング方法

1. 録音を開始し、自社側だけが話す
2. ログで BT の RMS 値を確認する（`docker compose logs -f minutes-app`）
3. 発話中の RMS 値の最小値を確認し、その 70% 程度を閾値として設定する

```
[DEBUG] [BT] RMS=1240.0   ← 発話中
[DEBUG] [BT] RMS=1150.0   ← 発話中
[DEBUG] [BT] RMS=85.0     ← 無音
[DEBUG] [BT] RMS=92.0     ← 無音

→ 発話中の最小 = 1150、その70% = 805
→ 閾値 = 800 が適切
```

---

## 11. LLM処理パイプライン

### 2段階処理の概要

文字起こし全文を一度にLLMに渡すとコンテキスト長を超えるため、2段階で処理します。

```
【Stage 1】チャンク分割 → 各チャンクから要点JSON抽出
  全文字起こし（N行）
    ↓ chunker.py（2000文字ごとに分割）
  チャンク1, チャンク2, ... チャンクN
    ↓ LLM × N回（並列ではなく順次）
  要点JSON 1, 要点JSON 2, ... 要点JSON N

【Stage 2】要点JSONの統合 → 最終議事録JSON生成
  要点JSON 1〜N をすべて連結
    ↓ LLM × 1回
  最終議事録JSON
    ↓ formatter.py
  議事録.docx
```

### Stage 1 出力フォーマット（要点JSON）

```json
{
  "discussions": [
    {
      "topic": "測定値の余裕度について",
      "client_request": "測定値が仕様上限に近いため余裕を持たせてほしい",
      "our_response": "承認。次回試験までに±5%のマージンを設ける",
      "status": "合意済み"
    }
  ],
  "action_items": [
    {
      "content": "マージン設定の変更",
      "owner": "our_side",
      "deadline": "2025年3月末"
    }
  ]
}
```

### Stage 2 出力フォーマット（最終議事録JSON）

```json
{
  "trial_name": "〇〇装置 第1回立会試験",
  "date": "2025年03月15日",
  "location": "第1試験室",
  "attendees": {
    "client": ["山田 太郎", "鈴木 一郎"],
    "our_side": ["田中 花子"]
  },
  "discussions": [ ... ],
  "action_items": [ ... ]
}
```

### プロンプトの調整

プロンプトは `app/llm/prompts.py` に定義されています。  
議事録の内容が期待どおりにならない場合はここを修正します。

主な調整ポイント：

```python
# Stage1プロンプト（抽出精度の改善）
# 「status」の判定基準を明示する例
"""
statusの判定基準:
- 合意済み: 双方が明示的に同意した
- 継続協議: 次回以降も議論が必要
- 保留: 決定を保留した・情報収集中
"""

# Stage2プロンプト（統合精度の改善）
# 試験の種類が固定されている場合は事前定義を追加できる
"""
この試験は以下の項目を確認する試験です:
- 外観検査
- 寸法確認
- 動作試験
上記項目に関する協議を優先的に抽出してください。
"""
```

---

## 12. VRAM管理

RTX 3060 Laptop の VRAM は 6GB のため、STT と LLM を同時にロードできません。  
以下の順序で逐次管理しています。

```
試験中（録音中）
  └─ faster-whisper large-v3 が常駐（~3GB使用）
  └─ Ollama はモデル未ロード状態

「議事録を生成」ボタン押下
  └─ 1. faster-whisper のモデルをメモリから削除
  └─ 2. torch.cuda.empty_cache() でVRAMを解放
  └─ 3. Ollama へ HTTP リクエスト（Ollamaが qwen2.5:7b をロード ~4.4GB）
  └─ 4. LLM処理（Stage1 × N + Stage2 × 1）
  └─ 5. 処理完了後、Ollama はモデルを保持（次回生成が速くなる）

次回録音開始時
  └─ Ollamaにアンロードを指示（OLLAMA_MAX_LOADED_MODELS=1 で自動管理）
  └─ faster-whisper を再ロード
```

### VRAM使用量の確認

```powershell
# Windows ホストで実行
nvidia-smi
# または Docker コンテナ内で
docker exec minutes-app nvidia-smi
```

### VRAM不足エラーが出る場合

ROS2などほかのDockerコンテナが VRAM を消費している可能性があります。

```powershell
# 他のコンテナのGPU使用状況を確認
nvidia-smi

# minutes-system 以外のコンテナを停止
docker stop <コンテナ名>
```

---

## 13. WebSocket通信プロトコル

### エンドポイント

| エンドポイント | 方向 | 説明 |
|---|---|---|
| `/ws/audio/bt?session_id=xxx` | 双方向 | BTイヤホンの音声ストリーム |
| `/ws/audio/internal?session_id=xxx` | 双方向 | 内蔵マイクの音声ストリーム・文字起こし受信 |

### メッセージ形式

**ブラウザ → サーバー（バイナリ）**

```
WebM/Opus 形式のバイト列（MediaRecorderのチャンク）
5秒ごとに送信
```

**サーバー → ブラウザ（テキスト JSON）**

```jsonc
// 文字起こし結果（internal エンドポイント）
{
  "type": "transcript",
  "speaker": "our_side",       // "our_side" | "client"
  "text": "測定値は規格内です",
  "timestamp": 1710000015.234
}

// RMS値（bt エンドポイント）
{
  "type": "rms",
  "value": 1240.5
}
```

### セッションID

セッションIDはブラウザ側で生成されるランダム文字列（7文字）です。  
メイン画面を開いたときに自動生成され、URLに付与されます。  
「BTウィンドウを開く」ボタンを使うと同じセッションIDが引き継がれます。

```
http://localhost:8000/?session=a3f9b2c
                               ↑ これがセッションID
http://localhost:8000/?role=bt&session=a3f9b2c
                                        ↑ 同じIDをBTウィンドウに渡す
```

セッションデータはコンテナのメモリ上に保持されます。  
コンテナを再起動するとセッションデータは消えます。

---

## 14. ファイル構成と各ファイルの役割

```
minutes-system/
├── docker-compose.yml         コンテナ定義・GPU設定・ボリュームマウント
├── .env.example               環境変数サンプル（HF_TOKENなど）
│
├── app/
│   ├── Dockerfile             Ubuntu22.04 + CUDA + Python + ffmpeg
│   ├── requirements.txt       Pythonパッケージ一覧
│   ├── main.py                FastAPIアプリ起動・STTモデル初期化
│   │
│   ├── config/
│   │   └── settings.yaml      モデル名・閾値・チャンク設定
│   │
│   ├── api/
│   │   ├── websocket.py       WebSocketエンドポイント・セッション管理
│   │   └── routes.py          REST API（生成・DL・メタ情報登録）
│   │
│   ├── stt/
│   │   ├── transcriber.py     faster-whisper ラッパー・ffmpeg変換
│   │   └── speaker.py         BTマイクRMSによる話者判定
│   │
│   ├── llm/
│   │   ├── client.py          Ollama HTTP クライアント
│   │   └── prompts.py         Stage1/Stage2 プロンプトテンプレート
│   │
│   ├── postprocess/
│   │   ├── chunker.py         文字起こしテキストの分割
│   │   └── formatter.py       議事録JSON → docx 変換
│   │                          ★社内テンプレートへの差し替えはここ
│   │
│   └── static/
│       └── index.html         ブラウザUI（録音・表示・生成）
│
├── ollama/
│   ├── Dockerfile             ollama/ollamaベースイメージ
│   └── entrypoint.sh          起動時にqwen2.5:7bを自動pull
│
├── output/                    生成済み議事録の保存先（ホストと共有）
└── models/                    HuggingFaceキャッシュ（将来のモデル追加用）
```

---

## 15. 議事録フォーマットの変更方法

現在は `formatter.py` に一般的なフォーマットが実装されています。  
社内テンプレートが決まったら以下の2つを変更します。

### パターンA：docxテンプレートファイルを使う場合

```python
# formatter.py を以下のように変更する

from docx import Document

def generate_docx(minutes: dict, output_path: str):
    # テンプレートを読み込む
    doc = Document("postprocess/template.docx")

    # テンプレート内のプレースホルダーを置換する
    for paragraph in doc.paragraphs:
        if "{{trial_name}}" in paragraph.text:
            paragraph.text = paragraph.text.replace("{{trial_name}}", minutes["trial_name"])
        # ... 他のフィールドも同様
    
    doc.save(output_path)
```

### パターンB：フォーマットをコードで定義する場合

`formatter.py` の `generate_docx` 関数を直接編集します。  
テーブルの列構成・見出しの文言・フォントサイズなどはすべてコードで制御されています。

変更が必要な主な箇所：

```python
# 基本情報テーブルの行定義（formatter.py 内）
info_rows = [
    ("試験名",   minutes.get("trial_name", "")),
    ("日時",     minutes.get("date", "")),
    ("場所",     minutes.get("location", "")),
    ("参加者",   f"相手方: {client_names}\n自社: {our_names}"),
    # ← ここに行を追加できる（例：「文書番号」「改訂番号」など）
]

# 協議事項テーブルの列定義
headers = ["No", "議題", "相手方 要望・確認事項", "自社 回答・対応", "状態"]
# ← 列名・列数を変更できる
```

---

## 16. ログの見方と問題診断

### ログの表示方法

```powershell
# 全ログをリアルタイムで表示
docker compose logs -f

# app コンテナのみ
docker compose logs -f minutes-app

# 直近100行のみ
docker compose logs --tail=100 minutes-app
```

### 主なログメッセージと意味

| ログメッセージ | 意味 |
|---|---|
| `STTモデルの初期化完了` | faster-whisperが正常にGPUにロードされた |
| `新規セッション: a3f9b2c` | ブラウザが初めてWebSocketに接続した |
| `[BT] 接続: a3f9b2c` | BTウィンドウが接続された |
| `[Internal] 接続: a3f9b2c` | メインウィンドウが接続された |
| `[BT] RMS=1240.0` | BTマイクの音量値（DEBUG レベル）|
| `[our_side] 測定値は...` | 文字起こし結果と話者ラベル |
| `Stage1 [1/3] 完了` | LLMのチャンク処理進捗 |
| `docx 生成完了` | 議事録ファイルが正常に生成された |
| `ffmpeg stderr: ...` | 音声変換の警告（多少のエラーは正常）|

### DEBUGログを有効にする方法

`app/main.py` の `logging.basicConfig` を変更します。

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

または `docker-compose.yml` に環境変数を追加します。

```yaml
environment:
  - LOG_LEVEL=DEBUG
```

### よくあるエラーとその対処

| エラー | 原因 | 対処 |
|---|---|---|
| `CUDA out of memory` | VRAMが不足している | 他のGPU使用プロセスを終了してコンテナを再起動 |
| `ffmpeg stderr: Invalid data found` | ブラウザから不正な音声データが送られてきた | 通常は自動回復。続く場合はブラウザをリロード |
| `Connection refused` (Ollama) | Ollamaコンテナが未起動 | `docker compose up minutes-ollama` |
| `JSONDecodeError` | LLMがJSON以外を返した | プロンプトの末尾に「JSONのみ出力」の指示を強化する |
| `WebSocket切断` | 録音中にPCがスリープした | スリープ設定を無効にして再録音 |