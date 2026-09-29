import pytest

from game.rules import Meld, Phase
from game.tiles import parse
from perception.config import (
    ACTION_BUTTON_CENTERS,
    CANCEL_BUTTON_REGION,
    DRAWN_TILE_REGION,
    HAND_GRID_RIGHT,
    HAND_TILE_BOTTOM,
)
from perception.state_parser import ObservedPlayer, ObservedTable, parse_observed_table


def test_screen_regions_fit_1920x1080():
    for left, top, right, bottom in (DRAWN_TILE_REGION, CANCEL_BUTTON_REGION):
        assert 0 <= left < right <= 1920
        assert 0 <= top < bottom <= 1080
    assert HAND_GRID_RIGHT < DRAWN_TILE_REGION[0]
    assert HAND_TILE_BOTTOM <= 1080
    assert all(0 < x < 1920 and 0 < y < 1080 for x, y in ACTION_BUTTON_CENTERS.values())


def test_observed_table_becomes_game_state_without_unknown_wall():
    players = tuple(
        ObservedPlayer(hand=tuple(parse("123456789m123456p1z")))
        if player == 0 else ObservedPlayer()
        for player in range(4)
    )
    observed = ObservedTable(
        players=players,
        current_player=0,
        phase=Phase.DISCARD,
        round_wind=parse("1z")[0],
        seat_winds=tuple(parse(f"{wind}z")[0] for wind in (1, 2, 3, 4)),
    )
    state = parse_observed_table(observed)
    assert state.wall == []
    assert state.players[0].hand_size == 16
    assert state.current_player == 0


def test_parser_preserves_public_discards_and_melds():
    players = tuple(ObservedPlayer() for _ in range(4))
    players = list(players)
    players[0] = ObservedPlayer(
        discards=tuple(parse("123m")),
        melds=(Meld("chow", (0, 1, 2), from_player=1),),
    )
    state = parse_observed_table(ObservedTable(players=tuple(players)))
    assert state.players[0].discards == parse("123m")
    assert state.players[0].melds[0].kind == "chow"


def test_parser_rejects_flowers_in_hand_and_wrong_player_count():
    with pytest.raises(ValueError):
        parse_observed_table(ObservedTable(players=(ObservedPlayer(hand=(34,)),) * 4))
    with pytest.raises(ValueError):
        parse_observed_table(ObservedTable(players=(ObservedPlayer(),) * 3))