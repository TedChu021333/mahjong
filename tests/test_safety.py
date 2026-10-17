from random import Random

import pytest

from agent.safety import claim_is_worse, safer_discard
from game.rules import Action, ActionType, GameState, Phase, PlayerState
from game.tiles import parse, to_counts


def P(value):
    return parse(value)[0]


def state_with(hand, **discards):
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse(hand))
    for seat, tiles in discards.items():
        players[int(seat[1])].discards = parse(tiles)
    return GameState(list(range(60)), players, phase=Phase.DISCARD)


def test_nothing_changes_without_a_declared_opponent():
    state = state_with("1479m258p369s1234567z", p1="1z2z3z")
    assert safer_discard(state, 0, P("5p"), Random(0)) == (P("5p"), None)


def test_far_from_tenpai_switches_to_a_safe_tile_against_a_declarer():
    # 手牌很差；下家宣告聽牌，打過的東、南、西對他絕對安全，五筒很危險
    state = state_with("1479m258p369s1234567z", p1="1z2z3z")
    state.declared[1] = True
    tile, note = safer_discard(state, 0, P("5p"), Random(0))
    assert tile in parse("123z") and "改打" in note


def test_already_safe_choice_is_kept():
    state = state_with("1479m258p369s1234567z", p1="1z2z3z")
    state.declared[1] = True
    assert safer_discard(state, 0, P("1z"), Random(0)) == (P("1z"), None)


def test_claim_is_not_checked_without_a_declared_opponent():
    state = state_with("1479m258p369s123456z", p1="3m")
    state.phase, state.last_discard, state.discard_player = Phase.RESPONSE, P("3m"), 3
    chow = Action(ActionType.CHOW, tile=P("3m"), tiles=tuple(parse("234m")), from_player=3)
    assert claim_is_worse(state, 0, chow, Random(0)) is None


def test_guard_only_declares_on_the_first_discard():
    pytest.importorskip("torch")
    from agent.nn_agent import NetworkPolicy

    policy = NetworkPolicy.__new__(NetworkPolicy)
    policy.last_note = None
    state = state_with("123456789m123p11z5s", p0="9s")
    declare = Action(ActionType.DECLARE, tile=P("5s"))
    action = policy._guarded(state, 0, declare, [declare], Random(0))
    assert action == Action(ActionType.DISCARD, tile=P("5s"))
    assert "不按聽" in policy.last_note
    state.players[0].discards = []  # 地聽照按
    assert policy._guarded(state, 0, declare, [declare], Random(0)) == declare
