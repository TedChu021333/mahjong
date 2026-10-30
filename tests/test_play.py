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



def test_watch_mode_names_and_hand_order():
    import play.__main__ as window
    from play.game import WATCH_NAMES
    assert describe_action(2, Action(ActionType.DISCARD, tile=0), WATCH_NAMES) == "對面 打 一萬"
    state = initial_state(Random(3))
    seat = 2
    hand = state.players[seat].hand
    low = min(t for t in range(34) if hand[t])
    # 剛摸進最小的那張：放最右邊
    state.phase, state.current_player, state.claimed, state.last_drawn = Phase.DISCARD, seat, False, low
    app = SimpleNamespace(just_drew=window.App.just_drew)
    order = window.App.hand_order(app, state, seat)
    assert order[-1] == low and sorted(order) == [t for t in range(34) for _ in range(hand[t])]
    state.current_player = 1  # 不是他剛摸的：照順序排
    assert window.App.hand_order(app, state, seat) == sorted(order)


def test_recorder_logs_decisions_and_analyze_reads_them(tmp_path):
    from game.simulator import payments, play_game
    from play.analyze import judge, load_rounds, report
    from play.game import HandSummary
    from play.record import Recorder

    recorder = Recorder(choose_rule_action_for_player, HUMAN, "rule", directory=tmp_path)
    recorder.new_round()

    def ask(state, actions):
        recorder.thinking(state, actions)
        choice = choose_rule_action_for_player(state, HUMAN, Random(0))
        recorder.decided(choice)
        return choice

    policy = seat_policy(HumanSeat(ask), choose_rule_action_for_player)
    result = play_game(Random(7), player_policy=policy, swap=True,
                       observer=lambda state, player, action: recorder.event(player, action))
    pay = payments(result)
    recorder.hand_end(HandSummary(1, result, pay, list(pay), 0, 0))
    recorder.round_end(list(pay))

    rounds = load_rounds([recorder.path])
    hand = rounds[0][0]
    assert hand.end["pay"] == list(pay) and hand.events
    discards = [d for d in hand.decisions if d["view"]["phase"] == "DISCARD"]
    assert discards and all("options" in d and "error" not in d for d in discards)
    # 自己和 AI 是同一個規則式：每一步都一樣，打牌也不會退向聽
    assert all(d["ai"] == d["choice"] for d in hand.decisions)
    assert not any(p.startswith("退向聽") for d in discards for p in judge(d))
    assert "1 圈 1 局" in report(rounds, show_all=False)


def test_judge_flags_missed_win_and_retreat():
    from play.analyze import judge
    view = {"phase": "DISCARD", "declared": [False, True, False, False], "hand": [0] * 34, "melds": [[]] * 4}
    decision = {
        "view": view,
        "actions": [{"kind": "WIN"}, {"kind": "DISCARD", "tile": 0}, {"kind": "DISCARD", "tile": 1}],
        "choice": {"kind": "DISCARD", "tile": 0, "tiles": []},
        "ai": {"kind": "DISCARD", "tile": 1, "tiles": []},
        "options": {"0": {"shanten": 1, "outs": 20, "danger": 0.2}, "1": {"shanten": 0, "outs": 8, "danger": 0.02}},
    }
    problems = judge(decision)
    assert any(p.startswith("漏胡") for p in problems)
    assert any(p.startswith("退向聽") for p in problems)
    assert any(p.startswith("危險牌") for p in problems)


def test_human_is_asked_once_per_discard():
    """沒人吃碰時模擬器一步只收第一家的「過」，自己排在後面也只問一次。"""
    from game.simulator import play_game
    asked = []

    def ask(state, actions):
        if state.phase == Phase.RESPONSE:
            # 被吃碰的牌會從牌河拿掉，所以連副露數一起當作「哪一張打出的牌」
            key = (state.discard_player, tuple(len(p.discards) for p in state.players),
                   tuple(len(p.melds) for p in state.players))
            assert key not in asked
            asked.append(key)
            return next(a for a in actions if a.kind == ActionType.PASS)
        return choose_rule_action_for_player(state, HUMAN, Random(0))

    for seed in range(6):
        asked.clear()
        play_game(Random(seed), player_policy=seat_policy(HumanSeat(ask), choose_rule_action_for_player))
