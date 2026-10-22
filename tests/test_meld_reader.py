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


def test_top_seat_reading_stays_accurate_on_labelled_screenshots():
    """對家（倒著擺）的已知答案截圖（10/22 目視核對）：33 張牌至少對 30、錯不超過 3（TOP_ASPECT 0.9 時只對 22、錯 8）。"""
    from collections import Counter

    from perception.capture import read_image
    from perception.meld_reader import MeldReader

    truth = {"經典_副露_驗證_202741.png": "444m", "經典_副露_驗證_203038.png": "222p",
             "經典_副露_驗證_203436.png": "789p111s", "經典_副露_驗證_203544.png": "123m111s",
             "經典_副露_驗證_200831.png": "789m111s666z", "經典_副露_驗證_193456.png": "456p",
             "經典_副露_對家碰發.png": "666z", "經典_副露_上家碰白_下家碰一筒.png": ""}
    paths = {name: Path("train") / name for name in truth}
    reader = MeldReader.from_directory()
    if reader is None or not all(path.exists() for path in paths.values()):
        pytest.skip("缺少訓練截圖或牌河模板")
    good = wrong = 0
    for name, answer in truth.items():
        got = Counter(hit.tile for hit in reader.read(read_image(paths[name]))["top"])
        want = Counter(parse(answer)) if answer else Counter()
        good += sum((got & want).values())
        wrong += sum((got - want).values())
    assert good >= 30 and wrong <= 3, (good, wrong)
