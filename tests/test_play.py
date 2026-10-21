from random import Random
from types import SimpleNamespace

import pytest

from agent.rule_agent import choose_rule_action_for_player
from game.rules import Action, ActionType, Phase, initial_state, legal_actions
from play.game import HUMAN, HumanSeat, describe_action, play_round, score_lines, seat_policy


def first_choice(state, actions):
    """測試用的「使用者」：換三張不換，其他照規則式 AI 會做的（不在合法動作裡就選第一個）。"""
    if actions[0].kind == ActionType.SWAP:
        return Action(ActionType.SWAP)
    return actions[0]


def test_human_is_asked_only_when_there_is_a_choice():
    asked = []
    seat = HumanSeat(lambda state, actions: asked.append(actions) or actions[0])
    state = initial_state(Random(1), swap=True)
    assert state.phase == Phase.SWAP and state.current_player == HUMAN
    seat(state, HUMAN, Random())
    assert asked and asked[0][0].kind == ActionType.SWAP
    asked.clear()
    state = initial_state(Random(1), dealer=1)
    state.current_player, state.phase = HUMAN, Phase.DRAW
    assert seat(state, HUMAN, Random()).kind == ActionType.DRAW  # 摸牌不問
    assert not asked


def test_human_gets_a_copy_of_the_state():
    seen = []
    seat = HumanSeat(lambda state, actions: seen.append(state) or actions[0])
    state = initial_state(Random(2))
    seat(state, HUMAN, Random())
    assert seen and seen[0] is not state


def test_round_runs_to_the_end():
    hands, steps = [], []
    policy = seat_policy(HumanSeat(first_choice), choose_rule_action_for_player)
    totals = play_round(policy, Random(3), hands.append,
                        observer=lambda state, player, action: steps.append(action))
    assert len(hands) >= 4
    assert sum(totals) == 0
    assert hands[-1].totals == totals
    assert steps
    # 每家都當過莊
    assert {hand.dealer for hand in hands} == {0, 1, 2, 3}


def test_messages_and_score_lines():
    assert describe_action(1, Action(ActionType.PUNG, tile=4, tiles=(4, 4, 4))) == "下家 碰 五萬"
    assert describe_action(0, Action(ActionType.DRAW)) is None
    result = SimpleNamespace(score=SimpleNamespace(patterns=("金牌", "金牌", "自摸")))
    assert score_lines(result) == ["金牌 ×2　2 台", "自摸　1 台"]


def test_window_draws_every_phase(monkeypatch):
    tk = pytest.importorskip("tkinter")
    import play.__main__ as window
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("沒有顯示器")
    root.withdraw()
    monkeypatch.setattr(window.App, "game_thread", lambda self: None)
    app = window.App(root, SimpleNamespace(seed=None, bot="rule", model=window.MODEL))
    try:
        # 換三張：點兩張再按換牌
        state = initial_state(Random(5), swap=True)
        app.state, app.actions = state, legal_actions(state, HUMAN)
        app.draw()
        app.tile_clicked(0)
        app.tile_clicked(1)
        app.press("swap")
        reply = app.from_gui.get_nowait()
        assert reply.kind == ActionType.SWAP and len(reply.tiles) == 2
        # 打牌：點最右邊那張
        state = initial_state(Random(5))
        app.state, app.actions = state, legal_actions(state, HUMAN)
        app.draw()
        app.tile_clicked(len(app.hand_tiles) - 1)
        reply = app.from_gui.get_nowait()
        assert reply.kind == ActionType.DISCARD and reply.tile == app.hand_tiles[-1]
        # 一整圈的結算畫面都畫得出來
        hands = []
        play_round(seat_policy(HumanSeat(first_choice), choose_rule_action_for_player), Random(6),
                   hands.append)
        for summary in hands:
            app.state, app.summary, app.totals = summary.result.state, summary, summary.totals
            app.draw()
        app.round_totals = hands[-1].totals
        app.draw()
    finally:
        root.destroy()


def test_art_images_have_expected_sizes(monkeypatch, tmp_path):
    from play import art
    image = art.tile_image(4, 60, 100)
    assert image.size == (60 + art.SHADOW, 100 + art.SHADOW)
    assert art.tile_image(36, 40, 66).size == (40 + art.SHADOW, 66 + art.SHADOW)  # 花牌用字畫
    assert art.back_image(24, 36).size == (24 + art.SHADOW, 36 + art.SHADOW)
    # 沒有模板的電腦退回用字畫
    monkeypatch.setattr(art, "TEMPLATE_DIR", tmp_path)
    art.template.cache_clear()
    try:
        assert art.template(4) is None
        assert art.tile_image(4, 61, 101).size == (61 + art.SHADOW, 101 + art.SHADOW)
    finally:
        art.template.cache_clear()


def test_bot_waits_before_discarding():
    import play.__main__ as window
    slept = []
    discard = Action(ActionType.DISCARD, tile=3)
    bot = window.slow_bot(lambda state, player, rng: discard, delay=2.0,
                          sleep=slept.append, clock=iter([0.0, 0.5]).__next__)
    assert bot(None, 1, None) is discard and slept == [1.5]
    slept.clear()
    passing = window.slow_bot(lambda state, player, rng: Action(ActionType.PASS), delay=2.0,
                              sleep=slept.append, clock=iter([0.0, 0.1]).__next__)
    passing(None, 1, None)
    assert slept == []
