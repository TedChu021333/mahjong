from random import Random

from agent.ev_agent import (
    choose_ev_action_for_player,
    discard_scores,
    honor_pung_value,
    honor_tai,
)
from game.rules import ActionType, GameState, Phase, PlayerState
from game.tiles import parse, to_counts
from tests.test_opponent_model import calibration


def P(value):
    return parse(value)[0]


def state_with(hand, **discards):
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse(hand))
    for seat, tiles in discards.items():
        players[int(seat[1])].discards = parse(tiles)
    return GameState(list(range(60)), players, phase=Phase.DISCARD)


def test_offense_decides_when_nobody_threatens():
    # 沒人聽牌：打孤張（九條、東、白）
    state = state_with("123456789m12345p9s17z")
    assert sum(state.players[0].hand) == 17
    scores = discard_scores(state, 0, Random(0), calibration(0.0), 20)
    assert max(scores, key=scores.get) in parse("9s17z")


def test_folds_against_a_declared_opponent_when_far_from_tenpai():
    # 手牌很差；下家宣告聽牌，打過的東、南、西對他絕對安全
    state = state_with("1479m258p369s1234567z", p1="1z2z3z")
    assert sum(state.players[0].hand) == 17
    state.declared[1] = True
    action = choose_ev_action_for_player(state, 0, Random(0), calibration(0.0), samples=150)
    assert action.kind == ActionType.DISCARD and action.tile in parse("123z")


def test_wins_are_never_skipped():
    state = state_with("123456789m123456p11z")
    state.last_drawn = P("1z")
    action = choose_ev_action_for_player(state, 0, Random(0), calibration(0.0), samples=10)
    assert action.kind == ActionType.WIN


def test_honor_tai_counts_round_wind_dragons_and_known_seat_wind():
    state = state_with("123456789m12345p9s17z")
    assert [honor_tai(state, 1, P(t)) for t in ("1z", "2z", "5z")] == [1, 1, 1]  # 下家門風南
    assert honor_tai(state, 1, P("3z")) == 0
    state.dealer_known = False  # 莊家不明：門風不算
    assert honor_tai(state, 1, P("2z")) == 0


def test_honor_pair_value_depends_on_unseen_copies():
    state = state_with("123456789m12345p9s17z")
    visible = [0] * 34
    hand = to_counts(parse("11z555z3z"))
    visible[P("1z")] = 2
    visible[P("5z")] = 3
    # 自己是莊（東風圈的東是圈風也是門風，2 台）：東對子剩 2 張 → 1 台；中刻子 1 台；西不加台
    assert honor_pung_value(state, 0, hand, visible) == 2.0
    visible[P("1z")] = 4
    assert honor_pung_value(state, 0, hand, visible) == 1.0
