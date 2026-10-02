import numpy as np
import pytest

from game.tiles import parse
from perception.capture import read_image
from perception.config import (
    ACTION_BUTTON_CENTERS,
    CANCEL_BUTTON_REGION,
    CLAIM_TEMPLATE_BOX,
    DECLARE_TEMPLATE,
    GOLD_MIN_SCORE,
    GOLD_TEMPLATE_BOX,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TEMPLATE_DIR,
    TRAIN_DIR,
)
from perception.hand_reader import TileClassifier
from perception.table_reader import read_buttons, read_claim_tile, read_gold_tiles


def screen_with_buttons(lit: set[str]) -> np.ndarray:
    screen = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 60, np.uint8)
    left, top, right, bottom = CANCEL_BUTTON_REGION
    screen[top:bottom, left:right] = (30, 30, 230)  # 紅色「取消」
    for name, (x, y) in ACTION_BUTTON_CENTERS.items():
        screen[y - 30:y + 30, x - 30:x + 30] = 220 if name in lit else 70
    return screen


def test_buttons_report_only_lit_actions():
    assert read_buttons(screen_with_buttons({"pung"})) == {"pung"}
    assert read_buttons(screen_with_buttons({"chow", "win"})) == {"chow", "win"}


def test_no_button_row_without_cancel_button():
    screen = screen_with_buttons({"pung"})
    left, top, right, bottom = CANCEL_BUTTON_REGION
    screen[top:bottom, left:right] = 60
    assert read_buttons(screen) is None


def test_all_lit_is_not_a_button_row():
    # 配桌畫面同位置剛好有紅色按鈕，後面也是亮的
    assert read_buttons(screen_with_buttons(set(ACTION_BUTTON_CENTERS))) is None


# 真實截圖：(按鈕, 可吃碰的牌, 來源, 金牌)
SCREENSHOTS = {
    "經典_碰": ({"chow", "pung"}, "2m", "left", "1s9p7s"),
    "經典_對家碰": ({"chow", "pung"}, "2s", "top", "4z7p4p"),
    "經典_下家打牌": (None, "5z", "right", "1s9p7s"),
}


@pytest.mark.skipif(
    not all((TRAIN_DIR / f"{name}.png").exists() for name in SCREENSHOTS)
    or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_real_screenshots():
    claim_reader = TileClassifier.from_directory(template_box=CLAIM_TEMPLATE_BOX)
    gold = TileClassifier.from_directory(template_box=GOLD_TEMPLATE_BOX, min_score=GOLD_MIN_SCORE)
    for name, (buttons, claim, source, gold_tiles) in SCREENSHOTS.items():
        image = read_image(TRAIN_DIR / f"{name}.png")
        assert read_buttons(image) == (None if buttons is None else frozenset(buttons)), name
        reading = read_claim_tile(image, claim_reader)
        assert (reading.tile, reading.source) == (parse(claim)[0], source), name
        assert read_gold_tiles(image, gold) == parse(gold_tiles), name


def test_chow_panels_are_found_only_on_the_choice_screen():
    from perception.table_reader import read_chow_panels
    blank = np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 60, np.uint8)
    assert read_chow_panels(blank) == ()
    for left in (700, 1060):
        blank[650:790, left:left + 280] = 235
    centers = read_chow_panels(blank)
    assert [x for x, _ in centers] == [839, 1199]


@pytest.mark.skipif(
    not (TRAIN_DIR / "吃牌2.png").exists() or not (TRAIN_DIR / "經典_碰.png").exists(),
    reason="缺少訓練截圖",
)
def test_real_chow_choice_screen():
    from perception.table_reader import read_chow_panels
    image = read_image(TRAIN_DIR / "吃牌2.png")
    assert len(read_chow_panels(image)) == 2
    assert read_chow_panels(read_image(TRAIN_DIR / "經典_碰.png")) == ()


@pytest.mark.skipif(
    not (TRAIN_DIR / "經典_牌河290.png").exists() or not DECLARE_TEMPLATE.exists(),
    reason="缺少訓練截圖或聽牌標記模板",
)
def test_declared_markers():
    import cv2
    from perception.table_reader import read_declared

    template = cv2.cvtColor(read_image(DECLARE_TEMPLATE), cv2.COLOR_BGR2GRAY)
    assert read_declared(read_image(TRAIN_DIR / "經典_牌河290.png"), template) == {"left"}
    # 95 秒是自己宣告聽牌，自己的標記不在對手的搜尋範圍內
    assert read_declared(read_image(TRAIN_DIR / "經典_牌河95.png"), template) == frozenset()
    assert read_declared(read_image(TRAIN_DIR / "經典_牌河95.png"), None) == frozenset()


# 上家牌河排到第二欄，圓框兩側都壓著牌河（2026-09-30 實戰），靠淡紫色外圈認出圓框
COVERED_CIRCLES = {"經典_上家第二欄吃": "1m", "經典_上家第二欄吃2": "8p"}


@pytest.mark.skipif(
    not all((TRAIN_DIR / f"{name}.png").exists() for name in COVERED_CIRCLES)
    or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_claim_tile_when_river_covers_both_sides_of_the_circle():
    claim_reader = TileClassifier.from_directory(template_box=CLAIM_TEMPLATE_BOX)
    for name, claim in COVERED_CIRCLES.items():
        reading = read_claim_tile(read_image(TRAIN_DIR / f"{name}.png"), claim_reader)
        assert (reading.tile, reading.source) == (parse(claim)[0], "left"), name


@pytest.mark.skipif(
    not (TRAIN_DIR / "經典_圓框蓋牌河.png").exists() or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_claim_tile_over_the_river_is_read_without_the_dark_circle():
    # 實戰 18:45：上家打的八筒圓框蓋在牌河上，兩側都是亮色的牌，圓框判斷失敗、吃碰沒處理
    claim_reader = TileClassifier.from_directory(template_box=CLAIM_TEMPLATE_BOX)
    reading = read_claim_tile(read_image(TRAIN_DIR / "經典_圓框蓋牌河.png"), claim_reader)
    assert (reading.tile, reading.source) == (parse("8p")[0], "left")


@pytest.mark.parametrize("seat, kind", [(seat, kind) for seat in ("left", "top", "right")
                                        for kind in ("chow", "pung")])
def test_claim_banner_on_real_screens(seat, kind):
    from perception.table_reader import read_claim_banner

    path = TRAIN_DIR / f"經典_吃碰字_{seat}_{kind}.png"
    if not path.exists():
        pytest.skip("缺少訓練截圖")
    assert read_claim_banner(read_image(path)) == (seat, kind)


def test_no_claim_banner_on_ordinary_screens():
    from perception.table_reader import read_claim_banner

    for name in ("經典_吃碰字_none", "經典_碰", "經典_暗槓", "經典_小結算", "經典_上訴"):
        path = TRAIN_DIR / f"{name}.png"
        if path.exists():
            assert read_claim_banner(read_image(path)) is None, name
