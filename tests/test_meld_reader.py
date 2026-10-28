from pathlib import Path

import pytest

from game.tiles import parse
from perception.meld_reader import describe, group_melds


def P(text):
    return parse(text)[0]


def test_group_melds_finds_kongs_pungs_and_chows_in_order():
    # 四張並排不是槓（遊戲把第四張疊在中間那張上）：碰加一張分不出的牌
    melds, leftover = group_melds(parse("5555m777z546p1s"))
    assert [m.kind for m in melds] == ["pung", "pung", "chow"]
    assert melds[2].tiles == tuple(parse("456p"))
    assert leftover == [P("5m"), P("1s")]
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


def test_top_seat_reading_stays_accurate_on_labelled_screenshots(tmp_path):
    """對家（倒著擺）的已知答案截圖（10/22 目視核對）：33 張牌至少對 32、錯不超過 1（TOP_ASPECT 0.9 時只對 22、錯 8；
    10/28 萬子數字只和牌河模板比、容許偏幾像素之前對 30、錯 3）。人工標註的模板就是從這幾張切的，不讀。"""
    import shutil
    from collections import Counter

    from perception.capture import read_image
    from perception.meld_reader import MeldReader

    truth = {"經典_副露_驗證_202741.png": "444m", "經典_副露_驗證_203038.png": "222p",
             "經典_副露_驗證_203436.png": "789p111s", "經典_副露_驗證_203544.png": "123m111s",
             "經典_副露_驗證_200831.png": "789m111s666z", "經典_副露_驗證_193456.png": "456p",
             "經典_副露_對家碰發.png": "666z", "經典_副露_上家碰白_下家碰一筒.png": ""}
    paths = {name: Path("train") / name for name in truth}
    seat_dir = tmp_path / "meld_templates"
    if Path("train/meld_templates").exists():
        shutil.copytree("train/meld_templates", seat_dir, ignore=shutil.ignore_patterns("manual_*"))
    reader = MeldReader.from_directory(seat_dir=seat_dir)
    if reader is None or not all(path.exists() for path in paths.values()):
        pytest.skip("缺少訓練截圖或牌河模板")
    good = wrong = 0
    for name, answer in truth.items():
        got = Counter(hit.tile for hit in reader.read(read_image(paths[name]))["top"])
        want = Counter(parse(answer)) if answer else Counter()
        good += sum((got & want).values())
        wrong += sum((got - want).values())
    assert good >= 32 and wrong <= 1, (good, wrong)


@pytest.mark.parametrize("name, seat, tile", [
    ("經典_槓_對家八萬.png", "top", "8m"),        # 10/28 21:37 中間那張比兩旁高
    ("經典_槓_上家八萬.png", "left", "8m"),       # 10/28 21:54 兩張八萬之間露出藍色牌側
    ("經典_副露_驗證_203436.png", "left", "1z"),  # 10/22 上家東槓
])
def test_stacked_fourth_tile_marks_a_kong(name, seat, tile):
    from perception.capture import read_image
    from perception.meld_reader import MeldReader

    path = Path("train") / name
    reader = MeldReader.from_directory()
    if not path.exists() or reader is None:
        pytest.skip("缺少訓練截圖或牌河模板")
    hits = reader.read(read_image(path))[seat]
    stacked = [hit for hit in hits if hit.stacked]
    assert len(stacked) == 3 and {hit.tile for hit in stacked} == {P(tile)}


def test_no_kong_on_screens_without_one():
    from perception.capture import read_image
    from perception.meld_reader import MeldReader

    reader = MeldReader.from_directory()
    names = ["經典_副露_上家碰白_下家碰一筒.png", "經典_副露_對家碰發.png", "經典_副露_下家吃四五六筒.png",
             "經典_副露_驗證_193456.png", "經典_副露_驗證_200831.png", "經典_副露_驗證_202741.png",
             "經典_副露_驗證_203038.png", "經典_副露_驗證_203544.png"]
    paths = [Path("train") / name for name in names]
    if reader is None or not all(path.exists() for path in paths):
        pytest.skip("缺少訓練截圖或牌河模板")
    for path in paths:
        read = reader.read(read_image(path))
        assert not any(hit.stacked for hits in read.values() for hit in hits), path.name
