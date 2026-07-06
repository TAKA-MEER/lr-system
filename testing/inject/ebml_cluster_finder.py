"""
単一の連続したWebM(Matroska)バイト列から、トップレベル要素を辿って
Segment内の Cluster(0x1F43B675) 要素の開始オフセットを列挙する簡易EBMLウォーカー。

これにより、本物のChrome MediaRecorderが5秒タイムスライスで出すBlob列
(1個目=ヘッダー+Cluster#1、2個目以降=Cluster単体)と構造的に同一な
チャンク列を、事前に1回でエンコードした連続WebMファイルから再現できる。
"""

EBML_HEADER_ID = 0x1A45DFA3
SEGMENT_ID = 0x18538067
CLUSTER_ID = 0x1F43B675
UNKNOWN_SIZE = -1


def _read_vint(data: bytes, pos: int, mask_first_byte: bool) -> tuple[int, int]:
    """EBML可変長整数を読む。(値, 消費バイト数) を返す。
    mask_first_byte=True の場合はID読み取り用(先頭バイトそのまま値に含める)。
    mask_first_byte=False の場合はサイズ読み取り用(長さを示す先頭ビットを取り除く)。
    """
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
        # ID: 先頭バイトも含めてそのまま値とする(EBML仕様上、IDはマーカービット込みでユニーク)
        value = int.from_bytes(raw, "big")
    else:
        # サイズ: 先頭バイトからマーカービットを取り除く
        first_masked = first & (mask - 1)
        value = first_masked
        for b in raw[1:]:
            value = (value << 8) | b
        # all-1のvintは「サイズ不明」を意味する(ライブストリーミング時に使われる)
        all_ones = (1 << (7 * length)) - 1
        if value == all_ones:
            value = UNKNOWN_SIZE

    return value, length


def _read_element_header(data: bytes, pos: int) -> tuple[int, int, int]:
    """(要素ID, サイズ, ヘッダー長) を返す。サイズ不明(-1)は呼び出し側で処理する。"""
    elem_id, id_len = _read_vint(data, pos, mask_first_byte=True)
    size, size_len = _read_vint(data, pos + id_len, mask_first_byte=False)
    return elem_id, size, id_len + size_len


def find_cluster_offsets(data: bytes) -> list[int]:
    """Segment直下の各Clusterの開始オフセット(EBML要素IDの先頭位置)を出現順に返す。"""
    pos = 0
    n = len(data)

    # 1. トップレベル: EBMLヘッダーをスキップ
    elem_id, size, hdr_len = _read_element_header(data, pos)
    if elem_id != EBML_HEADER_ID:
        raise ValueError(f"先頭がEBMLヘッダーではありません: id=0x{elem_id:X}")
    pos += hdr_len + size

    # 2. トップレベル: Segment要素に入る(サイズ不明=ライブストリームの場合はEOFまでとする)
    elem_id, size, hdr_len = _read_element_header(data, pos)
    if elem_id != SEGMENT_ID:
        raise ValueError(f"Segment要素が見つかりません: id=0x{elem_id:X}")
    seg_content_start = pos + hdr_len
    seg_content_end = n if size == UNKNOWN_SIZE else seg_content_start + size

    # 3. Segment直下の子要素を順に辿り、Clusterの開始位置を記録する
    offsets = []
    p = seg_content_start
    while p < seg_content_end and p < n:
        try:
            elem_id, size, hdr_len = _read_element_header(data, p)
        except (ValueError, IndexError):
            break
        if elem_id == CLUSTER_ID:
            offsets.append(p)
        if size == UNKNOWN_SIZE:
            # サイズ不明要素(通常はCluster自身がlive時にこうなる)は次のトップレベル
            # 要素IDが出現するまで中身とみなす、という厳密な処理は複雑なため、
            # 本ツールの用途(Segment直下がCluster/Info/Tracksのみ)では
            # 次の既知IDが見つかるまで1バイトずつ進めるフォールバックにする。
            p += hdr_len
            continue
        p += hdr_len + size

    return offsets


def split_into_mediarecorder_like_chunks(data: bytes) -> list[bytes]:
    """
    Cluster境界で分割し、本物のMediaRecorderのタイムスライスBlob列と
    構造的に同一なチャンク列(1個目=ヘッダー+Cluster#1、以降=Cluster単体)を返す。
    """
    offsets = find_cluster_offsets(data)
    if not offsets:
        raise ValueError("Cluster要素が1つも見つかりませんでした")

    bounds = [0] + offsets[1:] + [len(data)]
    chunks = [data[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    return chunks
