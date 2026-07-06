"""WebM(Matroska)ストリームの先頭要素を辿るための最小限のEBMLユーティリティ。

ブラウザのMediaRecorderが出す最初のBlobは
EBMLヘッダー + Segment(Info/Tracks) + 最初のCluster
という構造を持つ。真のコンテナヘッダー(音声フレームを含まない部分)は
最初のCluster要素の開始位置までである。
"""

EBML_HEADER_ID = 0x1A45DFA3
SEGMENT_ID = 0x18538067
CLUSTER_ID = 0x1F43B675
UNKNOWN_SIZE = -1


def _read_vint(data: bytes, pos: int, mask_first_byte: bool) -> tuple[int, int]:
    first = data[pos]
    if first == 0:
        raise ValueError(f"不正なvint(先頭バイトが0): pos={pos}")
    length = 1
    mask = 0x80
    while not (first & mask):
        length += 1
        mask >>= 1
        if length > 8:
            raise ValueError(f"不正なvint(長さ超過): pos={pos}")

    raw = data[pos : pos + length]
    if mask_first_byte:
        value = int.from_bytes(raw, "big")
    else:
        first_masked = first & (mask - 1)
        value = first_masked
        for b in raw[1:]:
            value = (value << 8) | b
        all_ones = (1 << (7 * length)) - 1
        if value == all_ones:
            value = UNKNOWN_SIZE

    return value, length


def _read_element_header(data: bytes, pos: int) -> tuple[int, int, int]:
    elem_id, id_len = _read_vint(data, pos, mask_first_byte=True)
    size, size_len = _read_vint(data, pos + id_len, mask_first_byte=False)
    return elem_id, size, id_len + size_len


def find_first_cluster_offset(data: bytes) -> int | None:
    """最初のCluster要素の開始バイトオフセット(=音声フレームを含まない真のヘッダー長)を返す。
    解析できない場合はNoneを返す(呼び出し側でフォールバック処理する)。"""
    try:
        pos = 0
        elem_id, size, hdr_len = _read_element_header(data, pos)
        if elem_id != EBML_HEADER_ID:
            return None
        pos += hdr_len + size

        elem_id, size, hdr_len = _read_element_header(data, pos)
        if elem_id != SEGMENT_ID:
            return None
        seg_start = pos + hdr_len
        seg_end = len(data) if size == UNKNOWN_SIZE else seg_start + size

        p = seg_start
        while p < seg_end and p < len(data):
            elem_id, size, hdr_len = _read_element_header(data, p)
            if elem_id == CLUSTER_ID:
                return p
            if size == UNKNOWN_SIZE:
                p += hdr_len
                continue
            p += hdr_len + size
        return None
    except (ValueError, IndexError):
        return None
