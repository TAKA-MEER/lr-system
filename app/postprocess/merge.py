"""Stage1の要点JSON(チャンクごと)を、Stage2に渡す前にプログラムで名寄せする。

長時間の試験では1つの協議事項が複数チャンクに分かれてStage1の話題が本来の数倍に膨らみ、
LLMが1回で統合しきれずに重複・出力の打ち切りが起きた(testing/CHANGELOG.md 第2期 追加調査)。
ここで同じ話題をまとめてからStage2に渡し、LLMには表現の整理だけを任せる。

- 同じ話題の判定: 話題名を正規化した文字列の類似度、または一方が他方を含む
- ただし英数字などの識別子(A盤/B盤、1号機/2号機 等)が異なるものは統合しない
- 状態は時系列で最後に述べられたものを採用し、要望・回答は最も情報量の多い(長い)記述を採用する
"""
import re
import unicodedata
from difflib import SequenceMatcher

_PUNCT = re.compile(r"[\s、。・,.:：;；「」『』（）()\[\]【】〔〕…ー―\-~〜]")
_ID = re.compile(r"[A-Za-z]+|\d+|[一二三四五六七八九十]+号|[甲乙丙]")


def _norm(text: str) -> str:
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text or "")).lower()


def _ids(text: str) -> set[str]:
    return set(_ID.findall(unicodedata.normalize("NFKC", text or "").upper()))


# 話題名に広く付く語(これだけが共通でも同じ話題とはみなさない)
_GENERIC = re.compile(r"試験|測定値|測定|結果|確認|再|について|変更|対応|追加|修正|の件|件|値|の")


def same_topic(a: str, b: str) -> bool:
    """同じ協議事項か。一方が他方を含む(3文字以上)、共通部分が4文字以上、または全体がほぼ同じ場合。
    英数字などの識別子(A盤/B盤 等)が違えば別の話題"""
    if _ids(a) != _ids(b):
        return False
    na, nb = _GENERIC.sub("", _norm(a)), _GENERIC.sub("", _norm(b))
    if not na or not nb:
        return _norm(a) == _norm(b)
    short, long_ = sorted((na, nb), key=len)
    if len(short) >= 3 and short in long_:
        return True
    sm = SequenceMatcher(None, na, nb)
    m = sm.find_longest_match(0, len(na), 0, len(nb))
    return m.size >= 4 or sm.ratio() >= 0.75


def _longer(a: str, b: str) -> str:
    return b if len(b or "") > len(a or "") else (a or "")


def merge_discussions(items: list[dict]) -> list[dict]:
    """items は時系列順。同じ話題を1件にまとめる"""
    groups: list[dict] = []
    for d in items:
        if not isinstance(d, dict) or not d.get("topic"):
            continue
        target = next((g for g in groups if any(same_topic(d["topic"], t) for t in g["_topics"])), None)
        if target is None:
            groups.append({"topic": d["topic"], "client_request": d.get("client_request") or "",
                           "our_response": d.get("our_response") or "", "status": d.get("status") or "",
                           "_topics": [d["topic"]]})
            continue
        target["_topics"].append(d["topic"])
        target["client_request"] = _longer(target["client_request"], d.get("client_request") or "")
        target["our_response"] = _longer(target["our_response"], d.get("our_response") or "")
        if d.get("status"):
            target["status"] = d["status"]  # 時系列で後の記述を優先
    return [{k: v for k, v in g.items() if not k.startswith("_")} for g in groups]


def _same_sentence(a: str, b: str) -> bool:
    if _ids(a) != _ids(b):
        return False
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= 0.8


def merge_actions(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for a in items:
        if not isinstance(a, dict) or not a.get("content"):
            continue
        # 対応内容は文章が長く共通の語句(「見積もりを提出」等)を含みやすいため、ほぼ同じ文だけをまとめる
        # (話題名と同じ基準では別の対応まで統合してしまった。比較実験 exp/prog-merge)
        prev = next((x for x in out if x.get("owner") == a.get("owner") and _same_sentence(x["content"], a["content"])), None)
        if prev is None:
            out.append(dict(a))
        else:
            prev["content"] = _longer(prev["content"], a["content"])
            prev["deadline"] = a.get("deadline") or prev.get("deadline") or ""
    return out


def merge_summaries(summaries: list[dict]) -> dict:
    """Stage1の要点JSONのリスト(時系列順)を1つの要点JSONにまとめる"""
    ds = [d for s in summaries for d in (s.get("discussions") or [])]
    acts = [a for s in summaries for a in (s.get("action_items") or [])]
    return {"discussions": merge_discussions(ds), "action_items": merge_actions(acts)}
