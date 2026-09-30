import numpy as np
import pytest

from perception.capture import read_image
from perception.config import BUTTON_TEMPLATE_DIR, CONTINUE_BUTTONS, TEMPLATE_DIR, TRAIN_DIR
from perception.continue_button import ContinueButton, ContinueButtons, crop


def fake_screen(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)


def test_finds_button_even_if_shifted_a_few_pixels():
    screen = fake_screen(0)
    buttons = ContinueButtons({"小結算繼續": crop(screen, CONTINUE_BUTTONS["小結算繼續"])})
    shifted = np.roll(screen, (4, -3), axis=(0, 1))
    left, top, right, bottom = CONTINUE_BUTTONS["小結算繼續"]
    expected = ContinueButton("小結算繼續", ((left + right) // 2, (top + bottom) // 2))
    assert buttons.find(shifted) == expected
    assert buttons.find(fake_screen(1)) is None


@pytest.mark.skipif(
    not (TRAIN_DIR / "經典_小結算.png").exists() or not (BUTTON_TEMPLATE_DIR / "小結算繼續.png").exists()
    or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或按鈕參考圖",
)
def test_settlement_screen_has_continue_and_no_hand():
    from hint.advisor import Readers, observe

    buttons = ContinueButtons.from_directory()
    settlement = read_image(TRAIN_DIR / "經典_小結算.png")
    assert buttons.find(settlement).name == "小結算繼續"
    assert observe(settlement, Readers()) is None
    for name in ("經典_碰", "經典_a200", "經典_換三張"):
        assert buttons.find(read_image(TRAIN_DIR / f"{name}.png")) is None


@pytest.mark.skipif(
    not (TRAIN_DIR / "經典_再贏一局.png").exists()
    or not (BUTTON_TEMPLATE_DIR / "再贏一局.png").exists(),
    reason="缺少訓練截圖或按鈕參考圖",
)
def test_play_again_button_after_a_drawn_hand():
    # 流局沒有小結算，直接出現「再贏一局 (倒數)」
    buttons = ContinueButtons.from_directory()
    assert buttons.find(read_image(TRAIN_DIR / "經典_再贏一局.png")).name == "再贏一局"
    assert buttons.find(read_image(TRAIN_DIR / "經典_小結算.png")).name == "小結算繼續"


def test_end_of_round_buttons():
    # 打一圈結束：大結算「繼續戰鬥」，倒數結束後的「再玩一局」對話框
    expected = {"經典_大結算": "繼續戰鬥", "經典_再玩一局": "再玩一局",
                "經典_再贏一局": "再贏一局", "經典_小結算": "小結算繼續", "經典_碰": None,
                "經典_破產": "破產再玩一局", "經典_財產不足": "再拚一局"}
    buttons = ContinueButtons.from_directory()
    for name, button in expected.items():
        path = TRAIN_DIR / f"{name}.png"
        if not path.exists() or not (BUTTON_TEMPLATE_DIR / "繼續戰鬥.png").exists():
            pytest.skip("缺少訓練截圖或按鈕參考圖")
        found = buttons.find(read_image(path))
        assert (found.name if found else None) == button, name
