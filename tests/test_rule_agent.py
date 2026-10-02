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


def test_rule_agent_pungs_when_it_widens_the_hand(monkeypatch):
    # 經典_碰.png：碰二萬向聽不變，但有效進張由 7 張增加為 15 張；舊標準（loose）會碰，
    # 預設（improve）向聽沒變好就不碰
    import agent.rule_agent

    state = response_state("11223789m99m4p56779s", "2m")
    assert choose_rule_action(state, Random(1))[1].kind == ActionType.PASS
    monkeypatch.setattr(agent.rule_agent, "CLAIM_MODE", "loose")
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
    assert choose_discard(hand, rng=Random(1)) == 28  # 平手時先打孤張的南
    # 看得到的牌會改變進張數：這裡九萬、八筒、南、西加權後仍同分，平手照樣先打孤張字牌
    assert choose_discard(hand, rng=Random(1), visible_counts=visible) in (28, 29)


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
    assert choose_swap_tiles(C("123456789m123456p1z")) == ()  # 已聽牌


def test_rule_agent_plays_full_games_with_swap():
    from game.simulator import play_game
    from agent.rule_agent import choose_rule_action_for_player
    for seed in range(3):
        result = play_game(Random(seed), player_policy=choose_rule_action_for_player, swap=True)
        assert result.state.phase.value == "ended"


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


def test_declared_opponent_is_the_biggest_threat_and_triggers_folding():
    from agent.rule_agent import opponent_threat_score, safe_tiles, should_fold

    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("1479m258p369s1234567z")  # 很差的牌，向聽數高；手上有 9m、東可以棄
    players[2].discards = parse("9m1z")
    state = GameState([0], players, phase=Phase.DISCARD)
    assert opponent_threat_score(state, 0, 2) == 0 and not should_fold(state, 0)
    state.declared[2] = True
    assert opponent_threat_score(state, 0, 2) == 5 and should_fold(state, 0)
    assert safe_tiles(state, 0) == set(parse("9m1z"))
    action = choose_rule_action_for_player(state, 0, Random(0))
    assert action.tile in parse("9m1z")  # 棄胡：打對聽牌者安全的牌


def test_swap_never_gives_away_gold_tiles():
    from agent.rule_agent import choose_swap_tiles

    # 21:12 那局的起手：金牌是一條、南、西，原本會換掉南、西
    hand = C("22238m23458p35689s23z")
    assert set(choose_swap_tiles(hand)) >= set(parse("23z"))
    swapped = choose_swap_tiles(hand, gold_tiles=parse("1s23z"))
    assert not set(swapped) & set(parse("1s23z"))
    assert len(swapped) == 3


def test_tenpai_prefers_the_wait_with_more_live_tiles():
    # 五組面子加東、九條：打哪張都是單騎聽另一張。別家打過三張九條 → 聽九條已經沒牌，改聽東
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p1z9s")
    assert sum(players[0].hand) == 17
    players[1].discards = parse("9s9s9s")
    state = GameState([0], players, phase=Phase.DISCARD)
    assert choose_rule_action_for_player(state, 0, Random(0)).tile == P("9s")
    players[1].discards = parse("1z1z1z")
    assert choose_rule_action_for_player(state, 0, Random(0)).tile == P("1z")


def test_concealed_kong_when_it_does_not_hurt():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("1111m234567p456s11z67z")
    assert sum(players[0].hand) == 17
    state = GameState(list(range(30)), players, phase=Phase.DISCARD)
    action = choose_rule_action_for_player(state, 0, Random(0))
    assert action.kind == ActionType.KONG and action.tile == P("1m")


def test_no_kong_when_the_fourth_tile_is_needed_in_a_run():
    from agent.rule_agent import kong_is_worth

    # 一萬要和 23m 組搭子：暗槓會讓向聽數變差
    hand = C("1111m23m45p78p12s45s567z")
    assert sum(hand) == 17
    assert not kong_is_worth(hand, 0, P("1m"))


def test_kong_is_preferred_over_pung_from_a_discard():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("555p123m456m789m124s7z")
    assert sum(players[1].hand) == 16
    players[3].discards = parse("5p")  # 對家打的（上家打的不能明槓）
    state = GameState([0] * 30, players, current_player=3, phase=Phase.RESPONSE,
                      last_discard=P("5p"), discard_player=3, response_player=1,
                      response_players=[1])
    assert choose_rule_action_for_player(state, 1).kind == ActionType.KONG


def test_isolated_gold_tile_is_discarded_before_breaking_runs():
    # 12:49 實戰：南、九萬都是孤張金牌，原本留著它們去拆五六七八九條
    hand = C("345p89p56789s2z66z9m")
    gold = parse("9p2z1m9m")
    assert choose_discard(hand, 1, rng=Random(0), gold_tiles=gold) in parse("2z9m")


def test_gold_tiles_in_use():
    from agent.rule_agent import gold_in_use

    hand = C("11m35p8s1z2z")
    assert gold_in_use(hand, P("1m"))              # 對子
    assert gold_in_use(hand, P("3p"))              # 旁邊有 5p
    assert not gold_in_use(hand, P("8s"))          # 孤張數牌
    assert not gold_in_use(hand, P("1z"))          # 孤張字牌


def test_waiting_on_a_gold_tile_is_preferred():
    # 聽牌時單騎：打 1s 聽金牌 9s（胡金牌 +3 台），而不是打 9s 聽 1s
    hand = C("123456789m123456p1s9s")
    assert choose_discard(hand, rng=Random(0), gold_tiles=(P("9s"),)) == P("1s")


def test_swap_avoids_giving_away_middle_tiles():
    from agent.rule_agent import choose_swap_tiles, swap_gift_cost

    # 15:23 實戰：中、五筒、八筒原本分數相同，五筒被換出去；五筒對別人最好用
    hand = C("4557m25899p256677s5z")
    swapped = choose_swap_tiles(hand, gold_tiles=parse("3m9p7p"))
    assert P("5p") not in swapped and P("5z") in swapped
    assert swap_gift_cost(P("5p")) > swap_gift_cost(P("8p")) > swap_gift_cost(P("9p")) \
        > swap_gift_cost(P("5z"))


def test_ties_discard_the_most_isolated_tile():
    from agent.rule_agent import keep_value

    # 實戰 17:53：打六萬、南、七筒向聽與進張都相同，應先打孤張的南
    hand = C("1255678m7999p2335s2z9m")
    assert choose_discard(hand, rng=Random(0)) == P("2z")
    assert keep_value(hand, P("2z")) < keep_value(hand, P("9m")) < keep_value(hand, P("6m"))


def test_claim_modes():
    # 經典_碰.png：碰二萬向聽不變、進張變多 → 只有 loose 會碰
    state = response_state("11223789m99m4p56779s", "2m")
    assert choose_rule_action_for_player(state, 1, claim_mode="loose").kind == ActionType.PUNG
    assert choose_rule_action_for_player(state, 1, claim_mode="improve").kind == ActionType.PASS
    # 碰完聽牌（一向聽 → 聽牌）：near 也碰
    state = response_state("111222333m44p77p5s89s", "4p", source=2)
    assert choose_rule_action_for_player(state, 1, claim_mode="near").kind == ActionType.PUNG
    # 很散的手（八向聽 → 七向聽）：improve 碰；near 只碰會加台的中，不碰西（玩家 1 門風是南）
    def far(pair, tile):
        return response_state(f"1479m258p369s12z{pair}67z", tile, source=2)
    assert choose_rule_action_for_player(far("33z", "3z"), 1, claim_mode="improve").kind         == ActionType.PUNG
    assert choose_rule_action_for_player(far("33z", "3z"), 1, claim_mode="near").kind         == ActionType.PASS
    assert choose_rule_action_for_player(far("55z", "5z"), 1, claim_mode="near").kind         == ActionType.PUNG


def test_swap_keeps_a_valued_honor_pair():
    # 實戰 10/2 12:57：東風圈的一對東被換掉一張
    from agent.rule_agent import choose_swap_tiles

    hand = C("22m33m78m23456p45s11z2z")
    gold = parse("3s1p3m")
    assert P("1z") in choose_swap_tiles(hand, gold_tiles=gold)
    kept = choose_swap_tiles(hand, gold_tiles=gold, valued_honors=parse("1z5z6z7z"))
    assert P("1z") not in kept and P("2z") in kept  # 孤張南（不加台）照換
