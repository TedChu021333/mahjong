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


def test_lickgu_shanten_needs_seven_pairs_and_one_triplet():
    # 16 張 8 對：再摸到任一對子牌成刻即胡，是聽牌而非胡牌
    assert seven_pairs_shanten(C("11223344556677m88p")) == 0
    assert shanten(C("11223344556677m88p")) == 0
    assert seven_pairs_shanten(C("11223344556677m888p")) == -1
    assert seven_pairs_shanten(C("1133557799m1133p59s")) == 1


def test_invalid_hand_size_is_rejected():
    with pytest.raises(ValueError):
        shanten(C("123m"))

def test_seven_pairs_do_not_count_as_tenpai_in_this_game():
    # 實戰 16:06：44m 88m 55p 66p 777p 44s 5s 88s 是嚦咕嚦咕聽 5s，但遊戲不能聽
    from agent.shanten import seven_pairs_shanten

    hand = to_counts(parse("44m88m55p66p777p44s5s88s"))
    assert seven_pairs_shanten(hand) == 0
    assert shanten(hand) > 0


def test_improvement_count_prefers_keeping_a_flexible_tile():
    # 實戰 10/2 14:07（副露 2 組）：進張相同時，打六萬後的改良牌比打北後少
    from agent.shanten import improvement_count

    hand = to_counts(parse("6m12p99p567s22z4z"))
    visible = list(hand)

    def after(tile):
        rest = list(hand)
        rest[parse(tile)[0]] -= 1
        return improvement_count(rest, 2, visible)

    assert after("4z") > after("6m")
