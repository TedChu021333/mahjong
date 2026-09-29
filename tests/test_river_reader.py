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
