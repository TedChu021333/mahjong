from pathlib import Path

import pytest

from bigtwo.cards import cards_text, parse_cards
from bigtwo.hand_reader import TEMPLATE_DIR, read_hand, uniform_positions
from bigtwo.screen import read_image

TRAIN = Path(__file__).resolve().parent.parent / "train" / "bigtwo"

pytestmark = pytest.mark.skipif(not (TEMPLATE_DIR / "rank").exists(), reason="沒有大老二模板（train/ 不進版控）")

# 由錄影 bigtwo_train/Desktop 2026.10.16 - 18.22.28.03.mp4 擷取，標註目視確認
CASES = {
    "手牌_a": ("3h 3s 4c 4d 5h 7d Ts Qd Qh Ks Ac Ah 2s", "3s 4d 5h Ah 2s"),
    "手牌_b": ("3c 3s 4s 5d 7h 7s 9d 9s Ts Ks Ac 2c 2s", ""),
    "手牌_c": ("3h 4c 5c 5s 6d 6s 8c Tc Jc Jh Qc 2c 2h", ""),
    # 刁牌：自己拖曳整理，左邊 5 張間距約 70px、右邊 5 張分開放
    "刁牌": ("3h 4c 5d 6s 2c 3s Ts Jc Jh 2h", ""),
    # 輪到自己、不能出的牌整張變暗；游標蓋住紅心 Q 的角落，只讀到 12 張
    "變暗": ("3h 3s 4c 4d 5h 7d Ts Qd Ks Ac Ah 2s", "Ac"),
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


def test_covered_suit_rejects_whole_hand():
    # 游標蓋住梅花 J 的花色：寧可整手不採用，也不要認成黑桃 J
    assert read_hand(_image("刁牌_變暗_游標")) is None


def test_uniform_positions_only_for_even_layout():
    assert len(uniform_positions(_image("手牌_b"))) == 13
    assert uniform_positions(_image("刁牌")) == []


def test_no_hand_on_settlement():
    assert not read_hand(_image("結算"))
