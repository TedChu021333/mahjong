import pytest
from game.tiles import parse, to_counts, tile_name, build_wall, hand_str
from game.win import (is_win, decompose, winning_tiles, is_lickgu, flower_win)


def C(s):
    return to_counts(parse(s))


def test_parse_and_names():
    assert hand_str(parse("19m5p3s1z7z")) == "一萬 九萬 五筒 三條 東 白"
    with pytest.raises(ValueError):
        parse("0m")
    with pytest.raises(ValueError):
        to_counts(parse("11111m"))


def test_wall():
    wall = build_wall()
    assert len(wall) == 144
    assert len(set(wall)) == 42


def test_basic_win():
    assert is_win(C("123456789m123456p11z"))


def test_win_with_open_melds():
    assert is_win(C("123456789m123p11z"), n_open_melds=1)
    assert not is_win(C("123456789m123p11z"), n_open_melds=0)


def test_not_win():
    assert not is_win(C("123456789m123457p11z"))


def test_honors_cannot_chow():
    assert not is_win(C("123z456m789m123p456p11s"))


def test_multiple_decompositions():
    # 111222333m 可拆成三刻或三順
    assert len(decompose(C("111222333m456p789p55s"))) >= 2


def test_single_wait():
    assert winning_tiles(C("123456789m123456p1z")) == parse("1z")


def test_nine_gates_waits():
    waits = winning_tiles(C("1112345678999m123p"))
    assert waits == parse("123456789m")


def test_lickgu():
    hand = C("1133557799m1122555z")
    assert is_lickgu(hand)
    assert is_win(hand)
    assert not is_win(hand, allow_lickgu=False)


def test_flower_win():
    assert flower_win(8) == "八仙過海"
    assert flower_win(7, other_holds_last_flower=True) == "七搶一"
    assert flower_win(7) is None
