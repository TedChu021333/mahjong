import pytest

from agent.shanten import effective_tiles, seven_pairs_shanten, shanten, standard_shanten
from game.tiles import parse, to_counts


def C(value: str) -> list[int]:
    return to_counts(parse(value))


def test_complete_hand_is_minus_one_shanten():
    complete = C("123456789m123456p11p")
    assert standard_shanten(complete) == -1
    assert shanten(complete) == -1


def test_one_tile_away_hand_is_tenpai():
    hand = C("123456789m123456p1z")
    assert shanten(hand) == 0
    assert effective_tiles(hand) == parse("1z")


def test_open_meld_reduces_required_concealed_hand_size():
    assert standard_shanten(C("123456789m123p1z"), n_open_melds=1) == 0


def test_seven_pairs_shanten():
    assert seven_pairs_shanten(C("11223344556677m88p")) == -1


def test_invalid_hand_size_is_rejected():
    with pytest.raises(ValueError):
        shanten(C("123m"))