from pathlib import Path

import pytest

from bigtwo.buttons import BUTTONS, button_state, center, my_turn, swap_phase
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
    assert clicks == [(167, 960), (298, 895), (1438, 670)]


def test_auto_mode_requires_explicit_yes():
    assert confirm_training_room(lambda prompt: "y")
    assert not confirm_training_room(lambda prompt: "")


# ---------- 畫面（需要 train/bigtwo/ 的截圖） ----------

@pytest.mark.parametrize("name, turn, swap, play", [
    ("輪到自己_出牌暗", True, False, "off"), ("出牌亮", True, False, "on"), ("帶頭_PASS暗", True, False, "on"),
    ("換三張", False, True, None), ("手牌_24", False, False, None), ("結算", False, False, None),
    ("對手出牌", False, False, None),
])
def test_buttons(name, turn, swap, play):
    image = _image(name)
    assert my_turn(image) == turn
    assert swap_phase(image) == swap
    assert button_state(image, "出牌") == play


def test_table_plays():
    # 上家的葫蘆散成扇形；下家的扇形左緣伸到對家的範圍也要整手算給下家
    assert read_plays(_image("對手出牌")) == {0: None, 1: None, 2: None, 3: c("7d 7h Ac Ah As")}
    assert read_plays(_image("下家葫蘆")) == {0: None, 1: c("6c 6d 6s Qc Qh"), 2: None, 3: None}
    assert read_plays(_image("三家單張")) == {0: c("Js"), 1: c("Ad"), 2: None, 3: c("Td")}
    assert read_plays(_image("帶頭_PASS暗")) == {s: None for s in range(4)}


def test_counts():
    if not DIGIT_DIR.exists():
        pytest.skip("沒有數字模板")
    assert read_counts(_image("張數")) == {1: 7, 2: 2, 3: 12}
    assert read_counts(_image("出牌亮")) == {1: 8, 2: 13, 3: 6}


# ---------- 小視窗文字 ----------

def test_overlay_text_uses_suit_symbols():
    from bigtwo.live import overlay_text, short
    assert overlay_text(c("3h 4c 5d 6s 2c")) == "出 ♥3 ♣4 ♦5 ♠6 ♣2"
    assert overlay_text(None) == "PASS"
    assert short(parse_cards("Th 2s")) == "♥10 ♠2"


def test_advisor_display_only_sends_changes(tmp_path):
    from bigtwo.live import Advisor, Log
    shown = []
    advisor = Advisor(Log(tmp_path, echo=False), show=shown.append)
    for text in ("出 ♣2", "出 ♣2", None, None, "PASS"):
        advisor.display(text)
    assert shown == ["出 ♣2", None, "PASS"]


# ---------- 自動點擊流程（假的畫面狀態） ----------

class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_auto(tmp_path, monkeypatch, states):
    import bigtwo.live as live
    monkeypatch.setattr(live, "button_state", lambda frame, name: states.get(name))
    clicks = []
    advisor = live.Advisor(live.Log(tmp_path, echo=False), live.Clicker(click=clicks.append, sleep=lambda s: None))
    advisor.clock = FakeClock()
    return advisor, clicks


def hand_cards(text, raised=""):
    up = set(parse_cards(raised))
    return [HandCard(card=card, x=100 + 131 * i, raised=card in up, score=1.0)
            for i, card in enumerate(parse_cards(text))]


def test_swap_clicks_cards_once_then_confirms(tmp_path, monkeypatch):
    advisor, clicks = make_auto(tmp_path, monkeypatch, {"換牌": "on", "不換": "on"})
    text = "3c 4d 5h 7d 9s Tc Jd Qc Kd Ac Ah 2s 2c"
    advisor.swap_target = frozenset(parse_cards("4d 5h 7d"))
    advisor.stable_hand = lambda frame: hand_cards(text)
    advisor.handle_swap("t", None)
    assert len(clicks) == 3
    advisor.handle_swap("t", None)                 # 冷卻中：牌還沒浮上來也不能再點（會取消選取）
    assert len(clicks) == 3
    advisor.clock.now = 1.0
    advisor.stable_hand = lambda frame: hand_cards(text, "4d 5h 7d")
    advisor.handle_swap("t", None)
    assert clicks[-1] == center("換牌") and advisor.swap_finished


def test_execute_clears_wrong_selection_before_choosing(tmp_path, monkeypatch):
    advisor, clicks = make_auto(tmp_path, monkeypatch, {"重選": "on", "出牌": "on", "PASS": "on"})
    advisor.choice = c("Ac Ah")
    advisor.execute(hand_cards("3c 4d Ac Ah", "3c"), None)
    assert clicks == [center("重選")]
    advisor.clock.now = 1.0
    advisor.execute(hand_cards("3c 4d Ac Ah"), None)
    assert len(clicks) == 3                        # 點 A♣、A♥
    advisor.clock.now = 2.0
    advisor.execute(hand_cards("3c 4d Ac Ah", "Ac Ah"), None)
    assert clicks[-1] == center("出牌")


def test_execute_pass(tmp_path, monkeypatch):
    advisor, clicks = make_auto(tmp_path, monkeypatch, {"PASS": "on"})
    advisor.choice = None
    advisor.execute(hand_cards("3c"), None)
    assert clicks == [center("PASS")]


def test_auto_stops_when_clicks_have_no_effect(tmp_path, monkeypatch):
    advisor, clicks = make_auto(tmp_path, monkeypatch, {"出牌": "on", "PASS": "on", "重選": "on"})
    advisor.choice = c("Ac Ah")
    hand = hand_cards("3c 4d Ac Ah")               # 怎麼點都沒有選起來
    for i in range(10):
        advisor.clock.now = i
        if advisor.clicker is None:
            break
        advisor.execute(hand, None)
    assert advisor.clicker is None
    assert len(clicks) == 2 * 3                    # 點了 3 輪（每輪 2 張）就停
    assert "不接受程式送出的點擊" in advisor.log.path.read_text(encoding="utf-8")


def test_successful_turns_do_not_trigger_no_effect_stop(tmp_path, monkeypatch):
    # 10/16 22:35：點牌成功、按出牌後沒清紀錄，三回合後被誤判成「點擊沒反應」而停止
    advisor, clicks = make_auto(tmp_path, monkeypatch, {"出牌": "on", "PASS": "on", "重選": "on"})
    for turn in range(5):
        advisor.choice = c("3c")
        advisor.clicked_raised = None if turn == 0 else advisor.clicked_raised
        advisor.clock.now = turn * 10
        advisor.execute(hand_cards("3c 4d 5h"), None)           # 點 3♣
        advisor.clock.now = turn * 10 + 1
        advisor.execute(hand_cards("3c 4d 5h", "3c"), None)     # 選起來了 → 出牌
        advisor.decided = None
    assert advisor.clicker is not None


def test_tracker_infers_passes_from_turn_order():
    t = Tracker()
    feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c")
    feed(t, {0: None, 1: c("5d"), 2: None, 3: None})          # 下家出
    feed(t, {0: None, 1: c("5d"), 2: None, 3: c("9d")})       # 上家壓：對家 PASS
    assert [(p.seat, p.combo) for p in t.history] == [(1, c("5d")), (2, None), (3, c("9d"))]
    t.update({s: None for s in range(4)})                      # 一墩結束
    feed(t, {0: None, 1: None, 2: None, 3: c("4d4h")})         # 上家贏了再帶頭：其他三家 PASS
    assert [p.seat for p in t.history[3:6]] == [0, 1, 2] and all(p.combo is None for p in t.history[3:6])


def test_mc_pass_constraints_from_history():
    from bigtwo.game import Play, View
    from bigtwo.mc_agent import pass_constraints
    history = [Play(1, c("5d")), Play(2, None), Play(3, c("9d")), Play(0, None)]
    view = View(0, tuple(parse_cards("3c")), (1, 12, 12, 12), c("9d"), 3, history, [], True)
    assert pass_constraints(view) == [(2, c("5d"), 1)]


def test_settlement_detected_and_saved_once(tmp_path, monkeypatch):
    import numpy as np
    import bigtwo.live as live
    from bigtwo.settlement import is_settlement
    assert is_settlement(_image("結算_48"))
    assert not is_settlement(_image("出牌亮")) and not is_settlement(_image("換三張"))
    saved = []
    monkeypatch.setattr(live, "write_image", lambda path, frame: saved.append(path))
    monkeypatch.setattr(live, "is_settlement", lambda frame: frame is not None and frame.mean() == 1)
    monkeypatch.setattr(live, "swap_phase", lambda frame: False)
    monkeypatch.setattr(live, "read_hand", lambda frame: None)
    monkeypatch.setattr(live, "read_plays", lambda frame: {s: None for s in range(4)})
    monkeypatch.setattr(live, "my_turn", lambda frame: False)
    advisor = live.Advisor(live.Log(tmp_path, echo=False), live=True)
    advisor.shot_dir = tmp_path
    for value in (0, 1, 1, 1, 0, 0):
        advisor.step("t", np.full((4, 4, 3), value, dtype=np.uint8))
    assert len(saved) == 1 and "結算" in saved[0].name


def test_match_end_screens():
    from bigtwo.match_end import find_continue
    assert find_continue(_image("再玩一局"))[0] == "再玩一局"
    assert find_continue(_image("社團廣告"))[0] == "廣告X"           # 廣告蓋著時先按 X
    for name in ("結算_48", "出牌亮", "換三張", "手牌_24"):
        assert find_continue(_image(name)) is None


def test_match_end_clicked_after_two_frames_in_auto_only(tmp_path, monkeypatch):
    import bigtwo.live as live
    monkeypatch.setattr(live, "find_continue", lambda frame: ("再玩一局", (1600, 937)))
    clicks = []
    auto = live.Advisor(live.Log(tmp_path, echo=False), live.Clicker(click=clicks.append, sleep=lambda s: None))
    auto.clock = FakeClock()
    auto.step("t", None)
    assert clicks == []                       # 第一幀先不按
    auto.step("t", None)
    assert clicks == [(1600, 937)]
    hint = live.Advisor(live.Log(tmp_path, echo=False))
    hint.step("t", None); hint.step("t", None)  # 提示模式不按，也不出錯


def test_pair_leaving_one_card_at_a_time_is_one_play():
    # 10/17 01:27：決定出的一對 K 一張張從手牌飛走，被記成兩個單張
    t = Tracker()
    feed_hand(t, "3d 4d Kd Ks Ac")
    t.expect = frozenset(parse_cards("Kd Ks"))
    assert feed_hand(t, "3d 4d Kd Ac") == []
    played = feed_hand(t, "3d 4d Ac")
    assert [(p.seat, p.combo) for p in played] == [(0, c("Kd Ks"))]
    assert t.expect is None


def test_counter_text_counts_cards_outside():
    from bigtwo.live import counter_text
    hand = parse_cards("2s Ah 3c")
    played = parse_cards("2c 2d Ac")
    text = counter_text(hand, played)
    assert "2:1" in text and "A:2" in text and "3:3" in text and "4:4" in text
    assert "2 ♥" in text and "A ♦♠" in text and "共 46 張" in text


def test_to_beat_follows_pass_button(tmp_path, monkeypatch):
    import bigtwo.live as live
    advisor = live.Advisor(live.Log(tmp_path, echo=False))
    t = advisor.tracker
    feed_hand(t, "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c")
    feed(t, {0: None, 1: c("2h"), 2: None, 3: None})
    t.update({s: None for s in range(4)})               # 動畫把桌面清空一下：紀錄以為一墩結束
    monkeypatch.setattr(live, "button_state", lambda frame, name: "on")
    assert advisor.to_beat(None) == (c("2h"), 1)      # PASS 亮著：其實在跟牌
    monkeypatch.setattr(live, "button_state", lambda frame, name: "off")
    assert advisor.to_beat(None) == (None, None)      # PASS 暗：帶頭
    assert live.button_state is not None
