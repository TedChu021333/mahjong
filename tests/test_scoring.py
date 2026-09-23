import pytest

from game.scoring import score_hand
from game.tiles import parse, to_counts


def C(value: str) -> list[int]:
    return to_counts(parse(value))


def test_pinghu_and_menzen_are_scored():
    score = score_hand(C("123456789m123456p11p"))
    assert score.tai == 3
    assert "平胡" in score.patterns
    assert "門清" in score.patterns


def test_all_triplets_and_mixed_one_suit_are_scored():
    score = score_hand(C("111222333m444m555z66z"))
    assert score.tai == 9
    assert "碰碰胡" in score.patterns
    assert "混一色" in score.patterns


def test_highest_decomposition_is_selected():
    score = score_hand(C("111222333m444p555p66s"))
    assert "碰碰胡" in score.patterns


def test_lickgu_is_scored_and_can_be_disabled():
    hand = C("1133557799m1122555z")
    assert score_hand(hand).tai == 4
    with pytest.raises(ValueError):
        score_hand(hand, allow_lickgu=False)


def test_self_draw_and_flowers_add_context_tai():
    score = score_hand(C("123456789m123456p11p"), self_draw=True, flowers=2)
    assert score.tai == 6
    assert score.patterns.count("花牌") == 2