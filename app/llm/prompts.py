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

重要: 会話中で期限や納期（「〜日までに」「〜までに対応」等）を伴う約束・対応が述べられた場合は、
その協議事項がすでに"合意済み"であっても、必ずaction_itemsにも同じ内容を1件追加すること。
「合意した」ことと「まだ実行していない対応が残っている」ことは別であり、期限付きの
未実施対応はdiscussionsのstatusに関わらずすべてaction_itemsに含める。

抽出するJSON構造:
{{
  "discussions": [
    {{
      "topic": "話題の件名（5〜15文字程度の名詞句。例:「絶縁抵抗試験」「表示灯の色」「銘板の誤記」。同じ話題が会話中に複数回登場する場合は、毎回まったく同じ表現を使うこと。言い換えや装飾語の追加はしない）",
      "client_request": "相手方の要望・質問・指摘（なければnull）",
      "our_response": "自社の回答・承認・対応（なければnull）",
      "status": "合意済み または 継続協議 または 保留"
    }}
  ],
  "action_items": [
    {{
      "content": "対応内容（対応する discussions の topic と同じ名詞句を含めること）",
      "owner": "our_side または client（実際にその対応を実施する側。相手方に何かを提出・送付するのは自社なら owner は our_side）",
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
- topicの文字列が完全一致していなくても、同一の試験項目・指摘事項・議題を指している場合は
  同一の協議事項とみなし、1件に統合すること（例:「絶縁抵抗試験」と「絶縁抵抗の確認」は同じ話題）
- 同一の協議事項が複数のJSONにまたがって登場する場合、client_request/our_responseは
  最も情報量が多い（具体的な）記述を採用し、statusは会話の時系列で最後に確定した状態を
  必ず採用すること。特に、最初「継続協議」だったものが後から「合意済み」に変わった場合は、
  最終出力のstatusは必ず「合意済み」にすること（逆に後戻りすることはない）
- action_itemsについても同様に、同一の対応内容が複数回言及されている場合は1件に統合する
- 話題は時系列順に並べる

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
