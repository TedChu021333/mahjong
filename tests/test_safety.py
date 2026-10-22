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


def test_guard_declares_by_wait_shape():
    pytest.importorskip("torch")
    from agent.nn_agent import NetworkPolicy

    policy = NetworkPolicy.__new__(NetworkPolicy)
    policy.last_note = None
    # 嵌張只聽二條、摸四條就能改成聽兩種：網路想按也先不按
    state = state_with("123456789m12399p13s1z", p0="9s")
    declare = Action(ActionType.DECLARE, tile=P("1z"))
    action = policy._guarded(state, 0, declare, [declare], Random(0))
    assert action == Action(ActionType.DISCARD, tile=P("1z"))
    assert "不按聽" in policy.last_note
    state.players[0].discards = []  # 地聽照按
    assert policy._guarded(state, 0, declare, [declare], Random(0)) == declare
    # 聽三種（二五八條）：網路沒想按也按
    state = state_with("123456789m2345678s1z", p0="9s")
    discard = Action(ActionType.DISCARD, tile=P("1z"))
    declare = Action(ActionType.DECLARE, tile=P("1z"))
    assert policy._guarded(state, 0, discard, [discard, declare], Random(0)) == declare
    assert "按聽" in policy.last_note


def test_declare_rule_follows_the_wait_shape():
    from agent.rule_agent import declare_is_worth

    def worth(hand, **discards):
        state = state_with(hand, p0="9s", **discards)
        return declare_is_worth(state, 0, P("1z"))

    assert worth("123456789m2345678s1z")           # 聽二五八條三種
    assert worth("123456789m12345p55s1z")          # 聽三六筒兩種、剩 8 張
    assert not worth("123456789m12345p55s1z", p1="3p3p6p6p")  # 只剩 4 張
    assert not worth("123456789m12399p13s1z")      # 嵌張、摸四條能改成兩面


def test_first_discard_needs_no_counted_discards_either():
    from agent.rule_agent import declare_is_worth

    state = state_with("123456789m12399p13s1z")   # 嵌張、還能改善：平常不按
    assert declare_is_worth(state, 0, P("1z"))      # 但第一次打牌就聽（地聽）照按
    state.min_discards = 3  # 牌河漏記，但提示迴圈數到自己已經打過
    assert not declare_is_worth(state, 0, P("1z"))


def test_no_open_kong_while_concealed():
    from agent.rule_agent import choose_rule_action_for_player

    # 21:52 的手：門清聽三六筒，別人打七條
    state = state_with("123789m2245p234777s")
    state.phase, state.last_discard, state.discard_player = Phase.RESPONSE, P("7s"), 2
    state.response_players = [0]
    action = choose_rule_action_for_player(state, 0, Random(0))
    assert action.kind == ActionType.PASS


def test_open_kong_when_already_open():
    from agent.rule_agent import choose_rule_action_for_player
    from game.rules import Meld

    state = state_with("789m2245p234777s")
    state.players[0].melds = [Meld("chow", tuple(parse("123m")), 3)]
    state.phase, state.last_discard, state.discard_player = Phase.RESPONSE, P("7s"), 2
    state.response_players = [0]
    action = choose_rule_action_for_player(state, 0, Random(0))
    assert action.kind == ActionType.KONG


def test_guard_passes_on_open_kong_while_concealed():
    pytest.importorskip("torch")
    from agent.nn_agent import NetworkPolicy

    policy = NetworkPolicy.__new__(NetworkPolicy)
    policy.last_note = None
    state = state_with("123789m2245p234777s")
    state.phase, state.last_discard, state.discard_player = Phase.RESPONSE, P("7s"), 2
    kong = Action(ActionType.KONG, tile=P("7s"), tiles=(P("7s"),) * 4, from_player=2)
    passing = Action(ActionType.PASS, tile=P("7s"), from_player=2)
    assert policy._guarded(state, 0, kong, [passing, kong], Random(0)) == passing
    assert policy.last_note == "門清不明槓"


def test_efficiency_mode_prefers_two_sided_shapes_over_gold_on_ties():
    # 10/20 15:16：三四六七九筒（七、九筒是金牌）——打六筒和打九筒進張、改良牌都一樣，
    # 往後多看一步打九筒（留三四、六七兩面）較好；原本金牌不列入候選而打了六筒
    pytest.importorskip("torch")
    import numpy as np

    from agent.nn_agent import efficient_discard
    from game.rules import legal_actions
    from training.actions import legal_mask

    state = state_with("5567m34679p2236s8s66z1s")
    state.gold_tiles = parse("9p7p5m9m")
    _, lookup = legal_mask(legal_actions(state, 0))
    choice = Action(ActionType.DISCARD, tile=P("5m"))
    action = efficient_discard(state, 0, choice, lookup, np.zeros(200), turns=6)
    assert action == Action(ActionType.DISCARD, tile=P("9p"))


def test_guard_rejects_claims_that_do_not_improve_the_hand():
    pytest.importorskip("torch")
    from agent.nn_agent import NetworkPolicy

    policy = NetworkPolicy.__new__(NetworkPolicy)
    policy.last_note = None

    def respond(hand, tile):
        state = state_with(hand)
        state.phase, state.last_discard, state.discard_player = Phase.RESPONSE, P(tile), 2
        pung = Action(ActionType.PUNG, tile=P(tile), tiles=(P(tile),) * 3, from_player=2)
        passing = Action(ActionType.PASS, tile=P(tile), from_player=2)
        return policy._guarded(state, 0, pung, [passing, pung], Random(0))

    # 已經聽牌（一對五萬當眼、聽八條），碰了還是聽牌、沒變好：不碰
    assert respond("123456789m123p55m79s", "5m").kind == ActionType.PASS
    assert policy.last_note == "吃碰不會讓向聽變好"
    # 同樣的手、眼是中：碰中（三元牌加台）向聽不變也碰
    assert respond("123456789m123p55z79s", "5z").kind == ActionType.PUNG
