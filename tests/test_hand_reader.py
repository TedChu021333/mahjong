from pathlib import Path

import cv2
import numpy as np
import pytest

from game.tiles import parse
from perception.capture import read_image
from perception.config import (
    TEMPLATE_DIR,
    HAND_TILE_BOTTOM,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TILE_RAISE,
    TRAIN_DIR,
)
from perception.extract_templates import extract_templates
from perception.hand_reader import (
    TileClassifier,
    _cell_box,
    crop_tile,
    drawn_tile_box,
    locate_hand_tiles,
    read_hand,
)

TABLE_COLOR = (80, 60, 40)


def blank_screen() -> np.ndarray:
    screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
    screen[:] = TABLE_COLOR
    return screen


def draw_tile(screen: np.ndarray, box, seed: int, top: int = 836, raise_by: int = 0,
              bottom: int | None = None) -> None:
    """畫一張白色牌面，中間放由 seed 決定的彩色方塊圖樣。"""
    left, _, right, box_bottom = box
    bottom = box_bottom if bottom is None else bottom
    top, bottom = top - raise_by, bottom - raise_by
    screen[top:bottom - 3, left + 3:right - 3] = (235, 235, 235)
    pattern = np.random.default_rng(seed).integers(0, 200, (6, 4, 3), dtype=np.uint8)
    pattern = cv2.resize(pattern, (right - left - 30, bottom - top - 50),
                         interpolation=cv2.INTER_NEAREST)
    screen[top + 25:top + 25 + pattern.shape[0], left + 15:left + 15 + pattern.shape[1]] = pattern


def screen_with_hand(seeds: list[int]) -> np.ndarray:
    screen = blank_screen()
    for index, seed in enumerate(reversed(seeds)):
        draw_tile(screen, _cell_box(index), seed)
    return screen


def test_locates_hand_and_ignores_upright_meld_tiles():
    screen = screen_with_hand([1, 2, 3, 4, 5, 6, 7])
    # 副露牌直立在暗手牌左側，但白色牌面只到 y≈960，比暗手牌短
    for index in range(7, 10):
        draw_tile(screen, _cell_box(index), 99, bottom=963)
    boxes = locate_hand_tiles(screen)
    assert len(boxes) == 7
    assert boxes == sorted(boxes)
    assert boxes[-1] == _cell_box(0)


def test_isolated_tile_left_of_a_gap_is_not_a_hand_tile():
    # 動畫或副露偶爾在左側單獨被偵測到；它和暗手牌之間隔著空格
    screen = screen_with_hand([1, 2, 3, 4, 5])
    draw_tile(screen, _cell_box(14), 9)
    boxes = locate_hand_tiles(screen)
    assert len(boxes) == 5
    assert _cell_box(14) not in boxes


def test_flower_display_overlapping_tile_top_does_not_hide_tile():
    screen = screen_with_hand(list(range(16)))
    left, *_ = _cell_box(12)
    screen[780:860, left + 5:left + 80] = (200, 120, 40)  # 壓在手牌頂端的花牌
    assert len(locate_hand_tiles(screen)) == 16


def test_partly_filled_grid_and_drawn_tile():
    # 張數不足時由左邊排起、右側格子空著；摸進的牌在獨立位置，排在最後
    screen = blank_screen()
    for index in range(15, 0, -1):
        draw_tile(screen, _cell_box(index), index)
    draw_tile(screen, drawn_tile_box(), 50)
    boxes = locate_hand_tiles(screen)
    assert len(boxes) == 16
    assert _cell_box(0) not in boxes
    assert boxes[-1] == drawn_tile_box()


def test_raised_tile_is_found_and_cropped_at_raised_position():
    screen = screen_with_hand([1, 2, 3, 4])
    raised = _cell_box(1)
    screen[raised[1] - TILE_RAISE:raised[3], raised[0]:raised[2]] = TABLE_COLOR
    draw_tile(screen, raised, 2, raise_by=TILE_RAISE)
    boxes = locate_hand_tiles(screen)
    assert len(boxes) == 4
    # 合成牌面下緣比真實牌少 3px，量到的位移會多幾個像素
    assert abs(raised[1] - boxes[2][1] - TILE_RAISE) <= 4
    assert [box[1] for box in boxes].count(raised[1]) == 3


def test_empty_table_has_no_hand_tiles():
    assert locate_hand_tiles(blank_screen()) == []


def test_classifier_matches_templates_under_dimming():
    seeds = [10, 11, 12, 13]
    screen = screen_with_hand(seeds)
    boxes = locate_hand_tiles(screen)
    tiles = parse("1m5p9s7z")
    classifier = TileClassifier([(tile, crop_tile(screen, box)) for tile, box in zip(tiles, boxes)])
    dimmed = (screen.astype(np.float32) * 0.6 + 15).astype(np.uint8)
    reading = read_hand(dimmed, classifier)
    assert reading.tiles == tuple(tiles)
    assert reading.known_tiles() == tuple(tiles)


def test_unknown_tile_is_reported_as_none():
    screen = screen_with_hand([20, 21])
    boxes = locate_hand_tiles(screen)
    classifier = TileClassifier([(0, crop_tile(screen, boxes[0]))])
    reading = read_hand(screen, classifier)
    assert reading.tiles == (0, None)
    assert not reading.complete
    with pytest.raises(ValueError):
        reading.known_tiles()


def test_extracted_templates_round_trip(tmp_path: Path):
    screen = screen_with_hand([30, 31, 32])
    labels = parse("123m")
    paths = extract_templates(screen, labels, "合成", tmp_path)
    assert [path.parent.name for path in paths] == ["1m", "2m", "3m"]
    classifier = TileClassifier.from_directory(tmp_path)
    assert read_hand(screen, classifier).tiles == tuple(labels)
    with pytest.raises(ValueError):
        extract_templates(screen, parse("12m"), "合成", tmp_path)


def test_tile_crop_stays_inside_hand_row():
    box = _cell_box(0)
    assert box[3] == HAND_TILE_BOTTOM
    assert box[2] <= SCREEN_WIDTH


# 經典版綠色桌面的人工標註（暗手牌由左到右，含第 17 張）；由影片擷取，
# train/ 不進版控，沒有截圖時略過。名稱：a/b = 兩支影片，數字 = 秒。
SCREENSHOT_LABELS = {
    "經典_a20": "1123789m4p567789s37z",
    "經典_a45": "112237899m4p567789s",
    "經典_a75": "115789m4p567789s",
    "經典_a124": "2337m7p12234688s775z",
    "經典_a133": "2337m7p12234688s",
    "經典_a200": "7p1223489s77z",
    "經典_a280": "12289s77z",
    "經典_a283": "12289s772z",
    "經典_b10": "34489m1789p2458s465z",
    "經典_b43": "3344899m57889p2458s",
    "經典_b72": "3344899m557889p458s2p",
    "經典_b99": "3344899m557889p458s4m",
    "經典_c67": "22344688m22345567p",
}


def _nearby(a: str, b: str) -> bool:
    """同一支影片相隔 15 秒內的畫面幾乎一樣，留一驗證時一起排除。"""
    video_a, second_a = a[3], int(a[4:])
    video_b, second_b = b[3], int(b[4:])
    return video_a == video_b and abs(second_a - second_b) <= 15


@pytest.mark.skipif(
    not all((TRAIN_DIR / f"{name}.png").exists() for name in SCREENSHOT_LABELS),
    reason="缺少訓練截圖",
)
def test_real_screenshots_leave_one_out():
    """每張截圖只用其他（不相鄰）截圖的模板辨識，模擬遇到新畫面的情況。"""
    data = {}
    for name, labels in SCREENSHOT_LABELS.items():
        image = read_image(TRAIN_DIR / f"{name}.png")
        boxes = locate_hand_tiles(image)
        assert len(boxes) == len(parse(labels)), name
        data[name] = [crop_tile(image, box) for box in boxes], parse(labels)
    for name, (crops, labels) in data.items():
        classifier = TileClassifier([
            (tile, crop)
            for other, (other_crops, other_labels) in data.items() if not _nearby(name, other)
            for tile, crop in zip(other_labels, other_crops)
        ])
        for crop, tile in zip(crops, labels):
            predicted, _ = classifier.classify(crop)
            if tile in classifier.known_tiles:
                assert predicted == tile, name
            else:
                assert predicted is None, name


@pytest.mark.skipif(
    not (TRAIN_DIR / "經典_c69_變暗.png").exists() or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_dimmed_hand_while_claim_buttons_are_shown():
    """按鈕出現時不能吃碰的牌會變暗；排除同一時刻的模板後仍要全部認得。"""
    templates = [
        (parse(path.parent.name)[0], read_image(path))
        for path in TEMPLATE_DIR.glob("*/*.png")
        if not path.name.startswith(("經典_c6", "經典_c7"))
    ]
    classifier = TileClassifier(templates)
    reading = read_hand(read_image(TRAIN_DIR / "經典_c69_變暗.png"), classifier)
    expected = parse("22344688m22345567p")
    assert len(reading.tiles) == len(expected)
    for predicted, tile in zip(reading.tiles, expected):
        # 只在這一刻出現過的牌種（6萬）沒有其他模板，應回報無法辨識而不是猜錯
        assert predicted == (tile if tile in classifier.known_tiles else None)
