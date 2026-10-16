from pathlib import Path

import pytest

from bigtwo.buttons import BUTTONS, button_state, my_turn, swap_phase
from bigtwo.cards import parse_cards
from bigtwo.combos import classify
from bigtwo.counts import DIGIT_DIR, read_counts
from bigtwo.hand_reader import TEMPLATE_DIR
from bigtwo.live import Clicker, confirm_training_room
from bigtwo.hand_reader import HandCard
from bigtwo.screen import read_image
from bigtwo.table_reader import read_plays
from bigtwo.tracker import HAND_STABLE, Tracker

TRAIN = Path(__file__).resolve().parent.parent / "train" / "bigtwo"


def c(text):
    return classify(parse_cards(text))


def _image(name):
    path = TRAIN / f"{name}.png"
    if not path.exists() or not (TEMPLATE_DIR / "rank").exists():
        pytest.skip(f"沒有 {path} 或模板（train/ 不進版控）")
    return read_image(path)


# ---------- 出牌紀錄（不需要截圖） ----------

def feed_hand(tracker, text):
    out = []
    for _ in range(HAND_STABLE):
        out += tracker.my_hand(parse_cards(text))
    return out


def feed(tracker, plays, frames=2):
    out = []
    for _ in range(frames):
        out += tracker.update(plays)
    return out


def test_tracker_needs_two_frames_and_reports_last_play():
    t = Tracker()
    feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c")
    assert t.update({0: None, 1: c("5d5h"), 2: None, 3: None}) == []
    assert not t.settled
    assert [p.seat for p in t.update({0: None, 1: c("5d5h"), 2: None, 3: None})] == [1]
    assert t.settled
    assert t.to_beat() == (c("5d5h"), 1)


def test_tracker_own_play_from_hand_and_lead_after_everyone_passes():
    t = Tracker()
    feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c")
    feed(t, {0: None, 1: c("5d"), 2: None, 3: None})
    played = feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac")
    assert [(p.seat, p.combo) for p in played] == [(0, c("2c"))]
    # 桌面中間也顯示同一張：不重複記
    assert feed(t, {0: c("2c"), 1: None, 2: None, 3: None}) == []
    assert t.to_beat() == (None, None)          # 最後出牌的是自己：帶頭


def test_tracker_rejects_other_seat_showing_my_cards_or_replayed_cards():
    t = Tracker()
    feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c")
    assert feed(t, {0: None, 1: c("4c"), 2: None, 3: None}) == []        # 我手上的牌
    feed(t, {0: None, 1: c("4d"), 2: None, 3: None})
    assert feed(t, {0: None, 1: None, 2: c("4d"), 3: None}) == []        # 已經出過


def test_tracker_grows_partial_five_card_play():
    t = Tracker()
    feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c")
    feed(t, {0: None, 1: None, 2: None, 3: c("4d")})
    feed(t, {0: None, 1: None, 2: None, 3: c("4d 4h Jd Jh Js")})
    assert len(t.history) == 1 and t.history[0].combo == c("4d 4h Jd Jh Js")


def test_tracker_trick_closes_when_table_clears():
    t = Tracker()
    feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c")
    feed(t, {0: None, 1: c("9d"), 2: None, 3: None})
    t.update({s: None for s in range(4)})
    assert t.to_beat() == (None, None)


# ---------- 點擊（假的 click） ----------

def test_clicker_clicks_card_and_buttons_without_randomness():
    clicks = []
    clicker = Clicker(click=clicks.append, sleep=lambda s: None)
    clicker.card(HandCard(card=0, x=97, raised=False, score=1.0))
    clicker.card(HandCard(card=1, x=228, raised=True, score=1.0))
    clicker.button("出牌")
    assert clicks == [(162, 878), (293, 825), (1350, 630)]


def test_auto_mode_requires_explicit_yes():
    assert confirm_training_room(lambda prompt: "y")
    assert not confirm_training_room(lambda prompt: "")


# ---------- 畫面（需要 train/bigtwo/ 的截圖） ----------

@pytest.mark.parametrize("name, turn, swap, play", [
    ("變暗", True, False, "off"), ("出牌亮", True, False, "on"), ("帶頭", False, False, None),
    ("換三張", False, True, None), ("手牌_a", False, False, None), ("結算", False, False, None),
])
def test_buttons(name, turn, swap, play):
    image = _image(name)
    assert my_turn(image) == turn
    assert swap_phase(image) == swap
    assert button_state(image, "出牌") == play


def test_table_plays():
    plays = read_plays(_image("對手出牌"))
    assert plays == {0: c("Qd Qh"), 1: c("Kc Kh"), 2: None, 3: None}
    plays = read_plays(_image("帶頭"))
    assert plays[0] == c("Ac Ah") and plays[3] is None


def test_counts():
    if not DIGIT_DIR.exists():
        pytest.skip("沒有數字模板")
    assert read_counts(_image("帶頭")) == {1: 11, 2: 11, 3: 11}
    assert read_counts(_image("刁牌")) == {1: 9, 2: 7, 3: 13}
