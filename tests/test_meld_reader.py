from pathlib import Path

import pytest

from game.tiles import parse
from perception.meld_reader import describe, group_melds


def P(text):
    return parse(text)[0]


def test_group_melds_finds_kongs_pungs_and_chows_in_order():
    melds, leftover = group_melds(parse("5555m777z546p1s"))
    assert [m.kind for m in melds] == ["kong", "pung", "chow"]
    assert melds[2].tiles == tuple(parse("456p"))
    assert leftover == [P("1s")]
    # 跨花色不是吃
    melds, leftover = group_melds(parse("89m1p"))
    assert melds == [] and len(leftover) == 3


@pytest.mark.parametrize("name, expected", [
    ("經典_副露_上家碰白_下家碰一筒.png", {"left": "777z", "right": "111p", "top": ""}),
    ("經典_副露_下家吃四五六筒.png", {"left": "", "right": "654p", "top": ""}),
    ("經典_副露_對家碰發.png", {"left": "", "right": "", "top": "666z"}),
])
def test_reads_opponent_melds_from_screenshots(name, expected):
    from perception.capture import read_image
    from perception.meld_reader import MeldReader

    path = Path("train") / name
    reader = MeldReader.from_directory()
    if not path.exists() or reader is None:
        pytest.skip("缺少訓練截圖或牌河模板")
    read = reader.read(read_image(path))
    for seat, tiles in expected.items():
        assert sorted(hit.tile for hit in read[seat]) == sorted(parse(tiles) if tiles else []), describe(read)
