from random import Random

from agent.rule_agent import (
    choose_discard,
    choose_rule_action,
    choose_rule_action_for_player,
    danger_score,
    safe_tiles,
    suji_tiles,
    opponent_threat_score,
    should_fold,
    visible_tile_counts,
)
from game.rules import (
    Action,
    ActionType,
    GameState,
    Phase,
    PlayerState,
    apply_action,
    forbidden_after_claim,
    legal_actions,
)
from game.tiles import parse, to_counts


def C(value: str) -> list[int]:
    return to_counts(parse(value))


def P(value: str) -> int:
    return parse(value)[0]


def test_choose_discard_keeps_single_wait():
    hand = C("123456789m123456p12z")
    assert choose_discard(hand, rng=Random(1)) in parse("123456789m123456p12z")


def test_rule_agent_draws_and_discards():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p12z")
    state = GameState([0], players, phase=Phase.DISCARD)
    player, action = choose_rule_action(state, Random(1))
    assert player == 0
    # 打掉一張字牌後單吊聽牌，規則式 AI 會同時宣告聽牌
    assert action.kind == ActionType.DECLARE
    assert action.tile in parse("12z")


def test_rule_agent_wins_before_discarding():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p11p")
    state = GameState([], players, phase=Phase.DISCARD)
    _, action = choose_rule_action(state, Random(1))
    assert action.kind == ActionType.WIN


def test_player_policy_returns_only_that_players_action():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p12z")
    players[1].hand = C("123456789m123456p11p")
    state = GameState([0], players, phase=Phase.DISCARD)
    action = choose_rule_action_for_player(state, 0, Random(1))
    assert action is not None
    assert action.kind in {ActionType.DISCARD, ActionType.DECLARE}


def response_state(hand: str, discard: str, source: int = 0) -> GameState:
    """玩家 1 對 source 打出的牌做回應；source=0 是玩家 1 的上家，可以吃。"""
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C(hand)
    players[source].discards = parse(discard)
    return GameState([], players, current_player=source, phase=Phase.RESPONSE,
                     last_discard=parse(discard)[0], discard_player=source,
                     response_player=1, response_players=[1])


def test_rule_agent_skips_a_pung_that_does_not_help():
    # 已經有 111m，碰 1m 向聽與進張都沒有變好
    state = response_state("111234567m1234567p", "1m")
    assert choose_rule_action_for_player(state, 1).kind == ActionType.PASS


def test_rule_agent_pungs_when_it_widens_the_hand():
    # 經典_碰.png：碰二萬向聽不變，但有效進張由 7 張增加為 15 張
    state = response_state("11223789m99m4p56779s", "2m")
    _, action = choose_rule_action(state, Random(1))
    assert action.kind == ActionType.PUNG
    apply_action(state, action, 1)
    assert state.players[1].melds[0].kind == "pung"


def test_rule_agent_does_not_break_a_run_to_chow():
    # 手上 456789s 已成兩組順子，吃 5s 只是拆面子
    state = response_state("123m456789s1155p79p1z", "5s")
    assert any(action.kind == ActionType.CHOW for action in legal_actions(state, 1))
    assert choose_rule_action_for_player(state, 1).kind == ActionType.PASS


def test_forbidden_discards_after_claims():
    chow = lambda tiles: Action(ActionType.CHOW, tiles=tuple(parse(tiles)))
    assert forbidden_after_claim(chow("567s"), P("5s")) == (P("5s"), P("8s"))
    assert forbidden_after_claim(chow("345s"), P("5s")) == (P("5s"), P("2s"))
    assert forbidden_after_claim(chow("456s"), P("5s")) == (P("5s"),)
    assert forbidden_after_claim(chow("789s"), P("7s")) == (P("7s"),)   # 沒有 10
    assert forbidden_after_claim(chow("123s"), P("3s")) == (P("3s"),)   # 沒有 0
    assert forbidden_after_claim(Action(ActionType.PUNG, tiles=(1, 1, 1)), 1) == (1,)


def test_after_chow_the_engine_and_agent_avoid_forbidden_tiles():
    # 用 67s 吃 5s 後不能打 5s、8s，即使它們是最該打的孤張
    state = response_state("111m222m333m444p67s5s8s", "5s")
    chow = Action(ActionType.CHOW, tile=P("5s"), tiles=tuple(parse("567s")), from_player=0)
    apply_action(state, chow, 1)
    discards = {action.tile for action in legal_actions(state, 1)}
    assert P("5s") not in discards and P("8s") not in discards
    action = choose_rule_action_for_player(state, 1, Random(1))
    assert action.tile not in (P("5s"), P("8s"))
    apply_action(state, action, 1)
    assert state.forbidden_discards == ()


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
    counts = visible_tile_counts(state, 0)
    assert counts[parse("1m")[0]] == 2
    assert counts[parse("2m")[0]] == 2


def test_visible_tile_counts_do_not_include_opponent_hidden_hands():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p12z")
    players[1].hand = C("111222333m444p55z")
    state = GameState([], players)
    counts = visible_tile_counts(state, 0)
    assert counts[parse("1m")[0]] == 1
    assert counts[parse("5z")[0]] == 0


def test_safe_tiles_prefer_existing_opponent_discards_on_tie():
    hand = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0,
            1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    players = [PlayerState() for _ in range(4)]
    players[1].discards = parse("1m")
    players[2].discards = parse("1m")
    players[3].discards = parse("1m")
    state = GameState([], players)
    assert parse("1m")[0] in safe_tiles(state, 0)
    assert choose_discard(
        hand, rng=Random(1), safe_tiles=safe_tiles(state, 0), defensive=True
    ) == 0


def test_safe_tiles_require_all_opponents_to_have_discarded_the_tile():
    players = [PlayerState() for _ in range(4)]
    players[1].discards = parse("1m2m")
    players[2].discards = parse("1m3m")
    players[3].discards = parse("1m4m")
    state = GameState([], players)
    assert safe_tiles(state, 0) == {parse("1m")[0]}


def test_safe_tiles_focus_on_visible_threats_when_present():
    players = [PlayerState() for _ in range(4)]
    players[1].melds = [object(), object()]
    players[1].discards = parse("1m")
    players[2].discards = parse("2m")
    players[3].discards = parse("3m")
    state = GameState([], players)
    assert safe_tiles(state, 0) == {parse("1m")[0]}


def test_suji_tiles_only_include_suited_tiles_with_discarded_partners():
    players = [PlayerState() for _ in range(4)]
    players[1].discards = parse("1m4p9s")
    state = GameState([], players)
    suji = suji_tiles(state, 0)
    assert parse("4m")[0] in suji
    assert parse("1p")[0] in suji
    assert parse("7p")[0] in suji
    assert parse("6s")[0] in suji
    assert parse("1z")[0] not in suji


def test_danger_score_orders_safe_suji_and_unseen_honor():
    players = [PlayerState() for _ in range(4)]
    for player in (1, 2, 3):
        players[player].discards = parse("1m")
    players[1].discards += parse("4p")
    state = GameState([], players)
    assert danger_score(state, 0, parse("1m")[0]) == 0
    assert danger_score(state, 0, parse("1p")[0]) == 1
    assert danger_score(state, 0, parse("1z")[0]) > danger_score(state, 0, parse("5m")[0])


def test_defensive_mode_prioritizes_safe_tile_over_attack_score():
    hand = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 2, 0, 1, 0, 0, 0,
            1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]
    assert choose_discard(hand, rng=Random(1), safe_tiles={0}, defensive=True) == 0


def test_threat_score_uses_only_public_melds_and_discards():
    players = [PlayerState() for _ in range(4)]
    players[1].melds = [object(), object()]
    players[1].discards = parse("123456m")
    players[1].hand = C("111222333m444p55z")
    state = GameState([], players)
    assert opponent_threat_score(state, 0, 1) == 4


def test_should_fold_when_behind_against_high_public_threat():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("119m119p119s1234567z")
    players[1].melds = [object(), object()]
    players[1].discards = parse("123456m")
    state = GameState([], players)
    assert should_fold(state, 0)

def test_swap_chooses_isolated_tiles_and_keeps_good_hands():
    from agent.rule_agent import choose_swap_tiles
    assert set(choose_swap_tiles(C("123456789m1234p19s1z"))) == set(parse("19s1z"))
    assert choose_swap_tiles(C("1133557799m113355p")) == ()  # 已聽嚦咕嚦咕


def test_rule_agent_plays_full_games_with_swap():
    from game.simulator import play_game
    from agent.rule_agent import choose_rule_action_for_player
    for seed in range(3):
        result = play_game(Random(seed), player_policy=choose_rule_action_for_player, swap=True)
        assert result.state.phase.value == "ended"


def test_gold_tile_is_kept_when_another_discard_is_as_good():
    # 7s、東、南都是孤張，平常先打字牌；南是金牌時打東，東是金牌時打南
    hand = C("123456789m123p45p7s1z2z")
    assert sum(hand) == 17
    east, south = P("1z"), P("2z")
    assert choose_discard(hand, rng=Random(0), gold_tiles=(south,)) == east
    assert choose_discard(hand, rng=Random(0), gold_tiles=(east,)) == south


def test_gold_tile_is_kept_as_a_single_wait_instead_of_another_single():
    # 聽 1s 單騎摸進金牌 9s：打 1s 改聽金牌，同樣是聽牌但胡了多台
    hand = C("123456789m123456p1s9s")
    assert choose_discard(hand, rng=Random(0), gold_tiles=(P("9s"),)) == P("1s")


def test_gold_tile_is_discarded_if_keeping_it_costs_shanten():
    # 聽 3p、6p 兩面時摸進金牌 9s：打掉 9s 才能維持聽牌，向聽數優先
    hand = C("123456789m12345p11s9s")
    assert sum(hand) == 17
    tile = choose_discard(hand, rng=Random(0), gold_tiles=(P("9s"),))
    assert tile == P("9s")


def test_rule_agent_uses_the_tables_gold_tiles():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123p45p7s1z2z")
    state = GameState([0], players, phase=Phase.DISCARD, gold_tiles=[P("2z")])
    action = choose_rule_action_for_player(state, 0, Random(0))
    assert action.tile == P("1z")
