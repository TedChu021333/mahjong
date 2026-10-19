import numpy as np
import pytest

from game.tiles import parse
from perception.capture import read_image
from perception.config import RIVER_REGIONS, RIVER_ROW_TOPS, TRAIN_DIR
from perception.river_reader import (
    RiverReader,
    locate_river_tiles,
    extract_templates,
)
from perception.hand_reader import TileClassifier

# 影片 290 秒（第二局末）與 95 秒（第一局中）的牌河，依偵測順序（由上而下、由左而右）
LABELS_290 = {
    "top": "3z1p6z2p3z1z2s9s5z",
    "me": "5z7m5z3m9p7p9p2z",
    "left": "9s4s5s3m",
    "right": "2p9s6z1p1s6z1s9p",
}
COUNTS_95 = {"me": 3, "right": 0, "top": 2, "left": 4}  # 對家最新那張被橘色箭頭壓到上緣


def screenshot(name):
    path = TRAIN_DIR / f"{name}.png"
    if not path.exists():
        pytest.skip("缺少訓練截圖")
    return read_image(path)


def test_empty_table_has_no_river_tiles():
    green = np.zeros((1080, 1920, 3), np.uint8)
    green[:] = (40, 90, 30)
    assert all(not boxes for boxes in locate_river_tiles(green).values())


def test_synthetic_tile_faces_are_found_in_each_region():
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[:] = (40, 90, 30)
    for seat, (x0, y0, x1, y1) in RIVER_REGIONS.items():
        top = RIVER_ROW_TOPS[seat][0]
        image[top:top + 60, x0 + 30:x0 + 90] = 250
        image[y1 - 70:y1 - 10, x1 - 90:x1 - 30] = 250  # 不在已知的列上：當成動畫忽略
    found = locate_river_tiles(image)
    for seat in RIVER_REGIONS:
        assert [box[1] for box in found[seat]] == [RIVER_ROW_TOPS[seat][0]]


def test_real_rivers_are_located():
    counts = {seat: len(boxes) for seat, boxes in locate_river_tiles(screenshot("經典_牌河290")).items()}
    assert counts == {seat: len(parse(tiles)) for seat, tiles in LABELS_290.items()}
    counts = {seat: len(boxes) for seat, boxes in locate_river_tiles(screenshot("經典_牌河95")).items()}
    assert counts == COUNTS_95


def test_templates_round_trip(tmp_path):
    image = screenshot("經典_牌河290")
    labels = {seat: parse(tiles) for seat, tiles in LABELS_290.items()}
    extract_templates(image, labels, "t", tmp_path)
    reader = RiverReader(TileClassifier.from_directory(tmp_path, min_score=0.7))
    rivers = reader.read(image)
    assert {seat: [t.tile for t in tiles] for seat, tiles in rivers.items()} == labels


def test_reader_without_templates_still_locates_tiles(tmp_path):
    reader = RiverReader.from_directory(tmp_path / "missing")
    rivers = reader.read(screenshot("經典_牌河95"))
    assert {seat: len(tiles) for seat, tiles in rivers.items()} == COUNTS_95
    assert all(t.tile is None for tiles in rivers.values() for t in tiles)


# 10/19 使用者標註的局末截圖（偵測順序）：自己第二排在上方、對家第二排在下方、上家一列第 4 張
LABELS_SECOND_ROWS = {
    "經典_牌河第二排": {   # 局末_20261019_190558
        "me": "676m6z44z3z9p5p8m4z2z1p",
        "right": "1s1p6p2m6z2p9m7z7p1p2p8m",
        "top": "9p7p2z2s2z6m7z4p7p7p",
        "left": "7z6z9m3p9m5z9m3z9p7z",
    },
    "經典_牌河對家第二排": {   # 局末_20261019_182648
        "me": "3z8m1z1z2m7m1s8m4m1m",
        "right": "4p6z1m9p5s2m5m2s8m2z2m",
        "top": "1p2z7m1z6z5m2z2m2s7m3s3s",
        "left": "1m4m1z2z1s6z1s9m2s3z",
    },
}


@pytest.mark.parametrize("name", sorted(LABELS_SECOND_ROWS))
def test_second_rows_are_located_and_read(name):
    image = screenshot(name)
    rivers = RiverReader.from_directory().read(image)
    for seat, tiles in LABELS_SECOND_ROWS[name].items():
        assert [tile.tile for tile in rivers[seat]] == parse(tiles), seat
