from random import Random

from agent.rule_agent import choose_discard, choose_rule_action, safe_tiles, visible_tile_counts
from game.rules import ActionType, GameState, Phase, PlayerState, apply_action, legal_actions
from game.tiles import parse, to_counts


def C(value: str) -> list[int]:
    return to_counts(parse(value))


def test_choose_discard_keeps_single_wait():
    hand = C("123456789m123456p12z")
    assert choose_discard(hand, rng=Random(1)) in parse("123456789m123456p12z")


def test_rule_agent_draws_and_discards():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p12z")
    state = GameState([0], players, phase=Phase.DISCARD)
    player, action = choose_rule_action(state, Random(1))
    assert player == 0
    assert action.kind == ActionType.DISCARD
    assert action.tile in parse("123456789m123456p12z")


def test_rule_agent_wins_before_discarding():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p11p")
    state = GameState([], players, phase=Phase.DISCARD)
    _, action = choose_rule_action(state, Random(1))
    assert action.kind == ActionType.WIN


def test_rule_agent_accepts_a_pung_without_worsening_shanten():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("111234567m1234567p")
    players[0].discards = parse("1m")
    state = GameState([], players, current_player=0, phase=Phase.RESPONSE,
                      last_discard=parse("1m")[0], discard_player=0,
                      response_player=1, response_players=[1])
    player, action = choose_rule_action(state, Random(1))
    assert player == 1
    assert action.kind == ActionType.PUNG
    apply_action(state, action, player)
    assert state.players[1].melds[0].kind == "pung"


def test_visible_tiles_are_used_to_weight_effective_draws():
    hand = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0,
            1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    visible = [2, 2, 2, 4, 0, 3, 1, 3, 1, 0, 0, 0, 0, 1, 4, 1,
               4, 0, 4, 3, 4, 1, 2, 0, 0, 4, 2, 3, 1, 3, 1, 1, 3, 3]
    assert choose_discard(hand, rng=Random(1)) == 16
    assert choose_discard(hand, rng=Random(1), visible_counts=visible) == 8


def test_visible_tile_counts_include_hands_discards_and_melds():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p12z")
    players[1].discards = parse("1m2m")
    state = GameState([], players)
    counts = visible_tile_counts(state)
    assert counts[parse("1m")[0]] == 2
    assert counts[parse("2m")[0]] == 2


    def test_safe_tiles_prefer_existing_opponent_discards_on_tie():
        hand = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0,
            1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
        players = [PlayerState() for _ in range(4)]
        players[1].discards = parse("1m")
        state = GameState([], players)
        assert parse("1m")[0] in safe_tiles(state, 0)
        assert choose_discard(hand, rng=Random(1), safe_tiles=safe_tiles(state, 0)) == 0