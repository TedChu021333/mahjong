import numpy as np
import pytest

from game.tiles import parse
from perception.capture import read_image
from perception.config import (
    ACTION_BUTTON_CENTERS,
    CANCEL_BUTTON_REGION,
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
    "吃牌畫面": ({"chow"}, "8m", "left", "8p7z2z"),
    "碰牌畫面": ({"pung"}, "1p", "right", "4m4s5s1m"),
    "胡牌畫面": ({"win"}, "4s", "left", "3z5m8s"),
    "副露區2": (None, None, None, "2s1p8s7z1s"),
}


@pytest.mark.skipif(
    not all((TRAIN_DIR / f"{name}.png").exists() for name in SCREENSHOTS)
    or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_real_screenshots():
    tiles = TileClassifier.from_directory()
    gold = TileClassifier.from_directory(template_box=GOLD_TEMPLATE_BOX)
    for name, (buttons, claim, source, gold_tiles) in SCREENSHOTS.items():
        image = read_image(TRAIN_DIR / f"{name}.png")
        assert read_buttons(image) == (None if buttons is None else frozenset(buttons)), name
        reading = read_claim_tile(image, tiles)
        if claim is None:
            assert reading is None, name
        else:
            assert (reading.tile, reading.source) == (parse(claim)[0], source), name
        assert read_gold_tiles(image, gold) == parse(gold_tiles), name
