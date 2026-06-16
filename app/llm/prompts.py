import json


def build_stage1_prompt(chunk_text: str) -> str:
    """
    Stage1: 会話チャンクから要点JSONを抽出するプロンプト
    """
    return f"""あなたは工場立会試験の議事録作成者です。
以下の会話録から情報を抽出し、JSON形式のみで出力してください。

話者ルール:
- our_side: 自社（承認・回答・対応を行う側）
- client: 相手方（要望・確認・指摘を行う側）

抽出するJSON構造:
{{
  "discussions": [
    {{
      "topic": "話題の件名（簡潔に）",
      "client_request": "相手方の要望・質問・指摘（なければnull）",
      "our_response": "自社の回答・承認・対応（なければnull）",
      "status": "合意済み または 継続協議 または 保留"
    }}
  ],
  "action_items": [
    {{
      "content": "対応内容",
      "owner": "our_side または client",
      "deadline": "期限（言及があれば記載、なければnull）"
    }}
  ]
}}

会話:
{chunk_text}

JSONのみ出力。前置き・説明・コードブロック記号は不要。"""


def build_stage2_prompt(summaries: list[str], meta: dict) -> str:
    """
    Stage2: 複数の要点JSONを統合して最終議事録JSONを生成するプロンプト
    """
    summaries_text = "\n---\n".join(summaries)

    return f"""以下の複数の要点JSONを統合し、最終議事録JSONを生成してください。

統合ルール:
- 同一内容の重複を排除する
- 話題は時系列順に並べる
- statusが矛盾する場合は最終的な合意状態を優先する

メタ情報:
- 試験名: {meta.get('trial_name', '')}
- 日時: {meta.get('date', '')}
- 場所: {meta.get('location', '')}
- 相手方参加者: {', '.join(meta.get('attendees', {}).get('client', []))}
- 自社参加者: {', '.join(meta.get('attendees', {}).get('our_side', []))}

要点JSONs:
{summaries_text}

出力するJSON構造:
{{
  "trial_name": "試験名",
  "date": "日時",
  "location": "場所",
  "attendees": {{
    "client": ["氏名"],
    "our_side": ["氏名"]
  }},
  "discussions": [
    {{
      "topic": "議題",
      "client_request": "相手方要望・確認事項",
      "our_response": "自社回答・対応",
      "status": "合意済み または 継続協議 または 保留"
    }}
  ],
  "action_items": [
    {{
      "content": "対応内容",
      "owner": "our_side または client",
      "deadline": "期限またはnull"
    }}
  ]
}}

JSONのみ出力。前置き・説明・コードブロック記号は不要。"""
