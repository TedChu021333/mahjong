from pathlib import Path

import pytest

from bigtwo.cards import cards_text, parse_cards
from bigtwo.hand_reader import TEMPLATE_DIR, read_hand, uniform_positions
from bigtwo.screen import read_image

TRAIN = Path(__file__).resolve().parent.parent / "train" / "bigtwo"

pytestmark = pytest.mark.skipif(not (TEMPLATE_DIR / "rank").exists(), reason="沒有大老二模板（train/ 不進版控）")

# 由 Google Play 版錄影 bigtwo_train/Google Play Games 2026.10.16 - 20.59.16.05.mp4 擷取，標註目視確認
CASES = {
    "手牌_24": ("3s 5c 6s 7c 7s Td Th Js Qc Qd Kh 2d 2s", ""),
    "手牌_56": ("3h 3s 4s 7s 8d Tc Th Js Kh Ks Ah 2c 2s", ""),
    "手牌_117": ("3c 5h 5s 6c 7c 8d 8s 9d 9s Jh Kd Ad Ah", ""),
    "手牌_240": ("3d 5d 5h 5s 6d 6s 8d 9d 9h 9s Jc Kh Ks", "3d 8d"),
    # 刁牌：自己拖曳整理，左邊 3 張疊得很緊、右邊分開
    "刁牌": ("5d 5h 5s 6d 6h 6s 9d 9h 9s Jc Kh", ""),
    "選起": ("3s 5h 7s 8d Tc Th Ts Js Kh Ks Ah 2c 2s", "Tc Th Ts"),
    "換三張": ("3s 5c 6s 7h 7s Td Th Js Qc Qd Kh 2d 2s", "7h"),
}


def _image(name):
    path = TRAIN / f"{name}.png"
    if not path.exists():
        pytest.skip(f"沒有 {path}")
    return read_image(path)


@pytest.mark.parametrize("name", CASES)
def test_read_hand(name):
    expected, raised = CASES[name]
    hand = read_hand(_image(name))
    assert hand is not None
    assert [c.card for c in hand] == parse_cards(expected)
    assert cards_text(c.card for c in hand if c.raised) == cards_text(parse_cards(raised))


def test_covered_hand_is_not_guessed():
    # 「你沒有更大的牌型可以出了！」提示字蓋住手牌下半：整手不採用
    assert read_hand(_image("變暗_提示字")) is None


def test_raised_five_not_read_as_three():
    # 選起的方塊 5 下方（平常牌的高度）曾被配成 3、分數比 5 還高（錄影 4:13）
    hand = read_hand(_image("選起5"))
    assert [c.card for c in hand if c.raised] == parse_cards("5d")
    assert parse_cards("3d")[0] not in [c.card for c in hand]


def test_uniform_positions_for_template_extraction():
    # 只在裁模板時用（一般排法）；刁牌時白色範圍可能剛好等於 13 張寬，執行時不靠它
    positions = uniform_positions(_image("手牌_240"))
    assert len(positions) == 13
    assert [top for _, top in positions].count(745) == 2


def test_no_hand_on_settlement():
    assert not read_hand(_image("結算"))
