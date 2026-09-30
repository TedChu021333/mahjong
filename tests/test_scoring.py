import pytest

from game.rules import Meld
from game.scoring import score_flower_win, score_hand
from game.tiles import parse, to_counts


def C(value: str) -> list[int]:
    return to_counts(parse(value))


def test_no_honors_no_flowers_and_menzen_are_scored():
    # 明星三缺一沒有平胡，改為無字無花 2 台（結算畫面 6 次）
    score = score_hand(C("123456789m123456p11p"), win_tile=parse("4p")[0])
    assert score.tai == 3
    assert "無字無花" in score.patterns and "門清" in score.patterns
    assert "平胡" not in score.patterns


def test_all_triplets_and_mixed_one_suit_are_scored():
    score = score_hand(C("111222333m444m555z66z"))
    # 門清 1 + 碰碰胡 4 + 五暗刻 8 + 混一色 4 + 三元牌(中) 1
    assert score.tai == 18
    assert "碰碰胡" in score.patterns
    assert "混一色" in score.patterns


def test_highest_decomposition_is_selected():
    score = score_hand(C("111222333m444p555p66s"))
    assert "碰碰胡" in score.patterns


def test_lickgu_is_off_by_default_but_can_be_enabled():
    hand = C("1133557799m1122555z")
    assert score_hand(hand, allow_lickgu=True).tai == 8
    with pytest.raises(ValueError):
        score_hand(hand)


def test_menzen_self_draw_and_every_flower_counts():
    # 明星三缺一每張花都算 1 台，不分是不是自己座位的花
    score = score_hand(C("123456789m123456p11p"), self_draw=True,
                       flowers=parse("125f"), seat_wind=27)
    assert "門清自摸" in score.patterns
    assert "平胡" not in score.patterns  # 自摸、有花都不算平胡
    assert score.patterns.count("花牌") == 3
    assert score.tai == 6


def test_honor_triplets_score_dragon_and_wind_tai():
    score = score_hand(
        C("111m222m333m111z555z66z"), seat_wind=27, round_wind=27
    )
    assert score.patterns.count("三元牌") == 1
    assert score.patterns.count("門風牌") == 1
    assert score.patterns.count("圈風牌") == 1
    # 門清 1 + 碰碰胡 4 + 五暗刻 8 + 混一色 4 + 三元 1 + 門風 1 + 圈風 1
    assert score.tai == 20

def test_concealed_kong_keeps_menzen_but_open_kong_does_not():
    hand = C("123456789m123p11z")
    concealed = score_hand(hand, [Meld("kong", (17,) * 4)], self_draw=True)
    assert "門清自摸" in concealed.patterns
    exposed = score_hand(hand, [Meld("kong", (17,) * 4, from_player=2)], self_draw=True)
    assert "門清自摸" not in exposed.patterns
    assert "自摸" in exposed.patterns


P = lambda value: parse(value)[0]  # noqa: E731


def test_no_honors_no_flowers_does_not_depend_on_the_wait():
    hand = C("123456789m123456p11p")
    for win_tile in ("3m", "5p", "1p"):
        assert "無字無花" in score_hand(hand, win_tile=P(win_tile)).patterns
    assert "無字無花" not in score_hand(hand, flowers=parse("1f")).patterns
    assert "無字無花" not in score_hand(C("123456789m123456p11z")).patterns


def test_kongs_score_extra_tai():
    # 163815 結算：暗槓 2 台、槓牌 1 台
    hand = C("123456789m11z")
    melds = [Meld("kong", (P("5p"),) * 4), Meld("kong", (P("7s"),) * 4, from_player=2)]
    score = score_hand(hand, melds, self_draw=True)
    assert "暗槓" in score.patterns and "槓牌" in score.patterns


def test_declaring_on_the_first_discard_is_di_ting():
    # 161823 結算：地聽 4 台（取代聽牌 1 台）
    hand = C("123456789m123456p11z")
    assert "聽牌" in score_hand(hand, declared=True).patterns
    early = score_hand(hand, declared=True, early_declared=True)
    assert "地聽" in early.patterns and "聽牌" not in early.patterns
    assert early.tai - score_hand(hand, declared=True).tai == 3


def test_single_wait_scores_du_ting():
    # 胡 1z 單吊，只聽這一張
    score = score_hand(C("123456789m123456p11z"), win_tile=P("1z"))
    assert "獨聽" in score.patterns
    assert "平胡" not in score.patterns


def test_concealed_triplets_and_ron_completed_pung():
    hand = C("111m222m333p456s789s77z")
    self_drawn = score_hand(hand, self_draw=True, win_tile=P("1m"))
    assert "三暗刻" in self_drawn.patterns
    # 胡別人打的 1m 完成的刻子算明刻，只剩兩暗刻
    ron = score_hand(hand, win_tile=P("1m"))
    assert not {"三暗刻", "四暗刻"} & set(ron.patterns)


def test_all_from_others_quan_qiu_ren():
    melds = [Meld("pung", (t,) * 3, from_player=1) for t in parse("159m")] + [
        Meld("chow", (9, 10, 11), from_player=3), Meld("pung", (P("3s"),) * 3, from_player=2)
    ]
    score = score_hand(C("11z"), melds, win_tile=P("1z"))
    assert "全求人" in score.patterns
    assert "獨聽" in score.patterns


def test_dragon_and_wind_limit_hands():
    assert "大三元" in score_hand(C("555666777z123m456p11s")).patterns
    small = score_hand(C("555666z77z123m111p789s"))
    assert "小三元" in small.patterns and "三元牌" not in small.patterns
    assert "大四喜" in score_hand(C("111222333444z123p11m")).patterns
    assert "小四喜" in score_hand(C("11122233344z111m123p")).patterns
    assert "字一色" in score_hand(C("11122233355566677z")).patterns


def test_dealer_streak_and_events():
    hand = C("123456789m123456p11z")
    score = score_hand(hand, self_draw=True, win_tile=P("1z"), dealer=True,
                       dealer_streak=2, events=["槓上開花"])
    assert {"莊家", "連2拉2", "槓上開花", "門清自摸", "獨聽"} <= set(score.patterns)
    assert score.tai == 1 + 4 + 1 + 3 + 1
    with pytest.raises(ValueError):
        score_hand(hand, events=["海底撈月"])  # 海底撈月必須是自摸


def test_flower_wins():
    assert score_flower_win("八仙過海").tai == 8
    assert score_flower_win("七搶一", dealer=True).tai == 9


def test_gold_tiles_and_declare_follow_game_settlement():
    # 對照影片 976 秒：金牌 西/五萬/八條，碰了西西西，胡 4s，花 3 張，有聽牌
    hand = C("22255p56s4s")
    melds = [Meld("pung", (P("3z"),) * 3, from_player=1),
             Meld("chow", (18, 19, 20), from_player=3),
             Meld("pung", (P("5z"),) * 3, from_player=2)]
    score = score_hand(hand, melds, flowers=parse("123f"), seat_wind=P("3z"),
                       round_wind=P("1z"), win_tile=P("4s"),
                       gold_tiles=parse("3z5m8s"), declared=True)
    assert score.patterns.count("花牌") == 3
    assert score.patterns.count("金牌") == 3
    assert "聽牌" in score.patterns
    assert score.tai == 9


def test_winning_on_gold_tile_adds_three():
    score = score_hand(C("123456789m123456p11z"), win_tile=P("1z"), gold_tiles=[P("1z")])
    assert score.patterns.count("金牌") == 2
    assert "胡金牌" in score.patterns
