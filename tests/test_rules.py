from random import Random

from game.rules import (
    DEAD_WALL_SIZE,
    GOLD_TILES_AT_START,
    Action,
    ActionType,
    Phase,
    GameState,
    PlayerState,
    apply_action,
    legal_actions,
    initial_state,
    seat_winds_for_dealer,
)
from game.rules import Meld
from game.tiles import parse, to_counts


def C(value: str) -> list[int]:
    return to_counts(parse(value))


def test_initial_state_deals_four_players_and_dealer_extra_tile():
    state = initial_state()
    assert len(state.players) == 4
    assert state.phase == Phase.DISCARD
    assert state.players[0].hand_size == 17
    assert [player.hand_size for player in state.players[1:]] == [16, 16, 16]
    assert sum(player.hand_size for player in state.players) == 65
    assert sum(player.hand_size + len(player.flowers) for player in state.players) + len(state.wall) == 144


def test_discard_actions_include_each_distinct_tile_and_self_draw_win():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p11z")
    state = GameState([], players, phase=Phase.DISCARD)
    actions = legal_actions(state)
    assert ActionType.WIN in [action.kind for action in actions]
    assert {action.tile for action in actions if action.kind == ActionType.DISCARD} == {
        *parse("123456789m123456p1z")
    }


def test_response_actions_for_next_player_include_chow_and_pass():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("2356m")
    players[0].discards = parse("4m")
    state = GameState([], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("4m")[0], discard_player=0,
                      response_player=1)
    actions = legal_actions(state, player=1)
    assert {action.kind for action in actions} == {ActionType.PASS, ActionType.CHOW}


def test_response_actions_for_non_next_player_cannot_chow():
    players = [PlayerState() for _ in range(4)]
    players[2].hand = C("2356m")
    state = GameState([], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("4m")[0], discard_player=0,
                      response_player=2)
    assert all(action.kind != ActionType.CHOW for action in legal_actions(state, player=2))


def test_all_three_other_players_can_pass_before_next_draw():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p11z")
    state = GameState([], players, phase=Phase.DISCARD)
    apply_action(state, Action(ActionType.DISCARD, tile=parse("1m")[0]))
    assert state.response_players == [1, 2, 3]
    for player in (1, 2):
        apply_action(state, Action(ActionType.PASS), player=player)
        assert state.phase == Phase.RESPONSE
    apply_action(state, Action(ActionType.PASS), player=3)
    assert state.phase == Phase.DRAW
    assert state.current_player == 1


def test_discard_then_pass_advances_to_next_player_draw():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123456p11z")
    state = GameState([0], players, phase=Phase.DISCARD)
    discard = Action(ActionType.DISCARD, tile=parse("1m")[0])
    apply_action(state, discard)
    assert state.phase == Phase.RESPONSE
    assert state.players[0].hand[0] == 0
    apply_action(state, Action(ActionType.PASS), player=1)
    apply_action(state, Action(ActionType.PASS), player=2)
    apply_action(state, Action(ActionType.PASS), player=3)
    assert state.phase == Phase.DRAW
    assert state.current_player == 1
    apply_action(state, Action(ActionType.DRAW))
    assert state.phase == Phase.DISCARD
    assert state.players[1].hand_size == 1


def test_chow_consumes_two_tiles_and_returns_to_discard():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("2356m")
    players[0].discards = parse("4m")
    state = GameState([], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("4m")[0], discard_player=0,
                      response_player=1)
    chow = next(action for action in legal_actions(state, player=1)
                if action.kind == ActionType.CHOW and action.tiles == (1, 2, 3))
    apply_action(state, chow, player=1)
    assert state.phase == Phase.DISCARD
    assert state.current_player == 1
    assert state.players[1].hand_size == 2
    assert state.players[1].melds[0].kind == "chow"


def test_discard_win_adds_claimed_tile_to_winner_hand():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("123456789m123456p1z")
    players[0].discards = parse("1z")
    state = GameState([], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("1z")[0], discard_player=0,
                      response_player=1, response_players=[1])
    win = next(action for action in legal_actions(state, player=1)
               if action.kind == ActionType.WIN)
    apply_action(state, win, player=1)
    assert state.phase == Phase.ENDED
    assert state.players[1].hand[parse("1z")[0]] == 2


def test_discard_kong_removes_claimed_tile_from_source_discard():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("111234567m1234567p")
    players[0].discards = parse("1m")
    state = GameState([5], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("1m")[0], discard_player=0,
                      response_player=1, response_players=[1])
    kong = next(action for action in legal_actions(state, player=1)
                if action.kind == ActionType.KONG)
    apply_action(state, kong, player=1)
    assert state.players[0].discards == []
    assert state.players[1].melds[0].tiles == (0, 0, 0, 0)


def test_taiwan_rules_have_no_furiten():
    # 台灣麻將沒有振聽：自己打過的牌，別人打出來仍可胡
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("123456789m123456p1z")
    players[1].discards = parse("1z")
    players[0].discards = parse("1z")
    state = GameState([], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("1z")[0], discard_player=0,
                      response_player=1, response_players=[1])
    assert any(action.kind == ActionType.WIN for action in legal_actions(state, player=1))


def test_flower_is_replaced_during_draw():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("123456789m123456p11z")
    state = GameState([0, 34], players, current_player=1, phase=Phase.DRAW)
    apply_action(state, Action(ActionType.DRAW))
    assert state.phase == Phase.DISCARD
    assert state.players[1].flowers == [34]
    assert state.players[1].hand[0] == 2

def test_seat_winds_rotate_with_dealer_and_wall_keeps_dead_tiles():
    state = initial_state(Random(1), dealer=2)
    assert state.seat_winds == seat_winds_for_dealer(2) == (29, 30, 27, 28)
    assert state.reserve == DEAD_WALL_SIZE
    players = [PlayerState() for _ in range(4)]
    stuck = GameState([0] * DEAD_WALL_SIZE, players, reserve=DEAD_WALL_SIZE)
    assert legal_actions(stuck) == []  # 只剩留牌，不能再摸


def test_cannot_win_right_after_claim():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("11m123456789p1234s")
    players[0].discards = parse("1m")
    state = GameState([5], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("1m")[0], discard_player=0,
                      response_player=1, response_players=[1])
    pung = next(a for a in legal_actions(state, 1) if a.kind == ActionType.PUNG)
    apply_action(state, pung, player=1)
    kinds = {a.kind for a in legal_actions(state)}
    assert kinds == {ActionType.DISCARD}


def test_added_kong_can_be_robbed():
    players = [PlayerState() for _ in range(4)]
    tile = parse("5m")[0]
    players[0].hand = C("5m123456789p1234s")
    players[0].melds = [Meld("pung", (tile,) * 3, from_player=2)]
    players[2].hand = C("46m123456789p111s22z")  # 聽 5m
    state = GameState([7, 8], players, phase=Phase.DISCARD)
    add_kong = next(a for a in legal_actions(state) if a.kind == ActionType.KONG)
    apply_action(state, add_kong)
    assert state.phase == Phase.RESPONSE and state.robbing_kong
    assert {a.kind for a in legal_actions(state, 1)} == {ActionType.PASS}
    win = next(a for a in legal_actions(state, 2) if a.kind == ActionType.WIN)
    apply_action(state, win, player=2)
    assert state.phase == Phase.ENDED
    assert state.players[0].melds[0].kind == "pung"  # 被搶槓，槓不成立


def test_added_kong_without_robber_draws_replacement():
    players = [PlayerState() for _ in range(4)]
    tile = parse("5m")[0]
    players[0].hand = C("5m123456789p1234s")
    players[0].melds = [Meld("pung", (tile,) * 3, from_player=2)]
    state = GameState([7, 8], players, phase=Phase.DISCARD)
    apply_action(state, next(a for a in legal_actions(state) if a.kind == ActionType.KONG))
    for player in (1, 2, 3):
        apply_action(state, Action(ActionType.PASS), player=player)
    assert state.phase == Phase.DRAW and state.current_player == 0 and state.after_kong
    assert state.players[0].melds[0].kind == "kong"


def test_eight_flowers_and_seven_rob_one():
    players = [PlayerState() for _ in range(4)]
    players[1].flowers = list(range(34, 41))
    players[1].hand = C("123456789m123456p1z")
    players[2].hand = C("123456789m123456p1z")
    state = GameState([0, 41], players, current_player=2, phase=Phase.DRAW)
    apply_action(state, Action(ActionType.DRAW))
    assert state.phase == Phase.ENDED
    assert (state.flower_win.winner, state.flower_win.kind, state.flower_win.from_player) == (1, "七搶一", 2)

    players = [PlayerState() for _ in range(4)]
    players[1].flowers = list(range(34, 41))
    players[1].hand = C("123456789m123456p1z")
    state = GameState([0, 41], players, current_player=1, phase=Phase.DRAW)
    apply_action(state, Action(ActionType.DRAW))
    assert (state.flower_win.winner, state.flower_win.kind) == (1, "八仙過海")


def test_gold_tiles_are_random_per_hand_and_kong_adds_one():
    state = initial_state(Random(5))
    assert len(state.gold_tiles) == GOLD_TILES_AT_START
    assert len(set(state.gold_tiles)) == GOLD_TILES_AT_START
    assert initial_state(Random(5)).gold_tiles == state.gold_tiles  # 同 seed 可重現

    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("1111m23456789p123s")
    state = GameState([7, 8], players, phase=Phase.DISCARD,
                      gold_tiles=[30, 31], gold_pool=[5, 6])
    apply_action(state, Action(ActionType.KONG, tile=0, tiles=(0,) * 4))
    assert state.gold_tiles == [30, 31, 5]


def test_declare_locks_hand_until_win():
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m123p45p9s11z")
    state = GameState([parse("9s")[0]], players, phase=Phase.DISCARD)
    declares = {a.tile for a in legal_actions(state) if a.kind == ActionType.DECLARE}
    assert parse("9s")[0] in declares      # 打 9s 聽 3p、6p
    assert parse("1z")[0] not in declares  # 打 1z 不聽
    apply_action(state, Action(ActionType.DECLARE, tile=parse("9s")[0]))
    assert state.declared[0] and state.phase == Phase.RESPONSE
    for player in (1, 2, 3):
        apply_action(state, Action(ActionType.PASS), player=player)
    state.current_player, state.phase = 0, Phase.DRAW
    apply_action(state, Action(ActionType.DRAW))
    # 摸到沒用的牌：只能打掉它
    assert legal_actions(state) == [Action(ActionType.DISCARD, tile=parse("9s")[0])]


def test_declared_player_cannot_claim():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("11m123456789p1234s")
    players[0].discards = parse("1m")
    state = GameState([5], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("1m")[0], discard_player=0,
                      response_player=1, response_players=[1],
                      declared=[False, True, False, False])
    assert {a.kind for a in legal_actions(state, 1)} == {ActionType.PASS}
