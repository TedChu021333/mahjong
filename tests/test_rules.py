from game.rules import (
    Action,
    ActionType,
    Phase,
    GameState,
    PlayerState,
    apply_action,
    legal_actions,
    initial_state,
)
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
    state = GameState([], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("1m")[0], discard_player=0,
                      response_player=1, response_players=[1])
    kong = next(action for action in legal_actions(state, player=1)
                if action.kind == ActionType.KONG)
    apply_action(state, kong, player=1)
    assert state.players[0].discards == []
    assert state.players[1].melds[0].tiles == (0, 0, 0, 0)


def test_player_cannot_ron_on_a_tile_already_discarded_by_self():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("123456789m123456p1z")
    players[1].discards = parse("1z")
    players[0].discards = parse("1z")
    state = GameState([], players, phase=Phase.RESPONSE, current_player=0,
                      last_discard=parse("1z")[0], discard_player=0,
                      response_player=1, response_players=[1])
    assert all(action.kind != ActionType.WIN for action in legal_actions(state, player=1))


def test_flower_is_replaced_during_draw():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = C("123456789m123456p11z")
    state = GameState([0, 34], players, current_player=1, phase=Phase.DRAW)
    apply_action(state, Action(ActionType.DRAW))
    assert state.phase == Phase.DISCARD
    assert state.players[1].flowers == [34]
    assert state.players[1].hand[0] == 2