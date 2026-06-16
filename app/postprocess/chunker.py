def split_transcript(text: str, max_chars: int = 2000) -> list[str]:
    """
    文字起こしテキストを max_chars 文字以下のチャンクに分割する。
    話者の発言単位（行）で分割してチャンクを作る。
    """
    lines = text.strip().split("\n")
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        if not line.strip():
            continue
        line_len = len(line)

        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks
