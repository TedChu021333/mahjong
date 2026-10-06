from pathlib import Path

import pytest

from eval.deal_in import fill_hidden, is_suji, main_shape, parse_log, parse_tiles, wait_shapes
from eval.settlement import SettlementReader
from game.tiles import parse
from perception.capture import read_image

TRAIN_DIR = Path("train")


def test_parse_tiles_mixes_two_and_one_character_names():
    assert parse_tiles("一萬五萬東九條白") == tuple(parse("15m9s1z7z")[i] for i in (0, 1, 3, 2, 4))


LOG = r"""
00:40:33 [00:40:33] 換三張：換 北、發、二萬　（手牌 二萬五萬，金牌 七筒）
00:40:42 [00:40:41] 打 發　（手牌 一萬五萬二筒三筒四筒四筒五筒六筒九筒九筒四條四條六條八條九條發二萬，金牌 七筒六筒二條）
00:40:45 [00:40:45] 牌河 me discard 發
00:40:48 [00:40:48] 牌河 right discard 南
00:40:50 [00:40:50] 牌河 right claimed 南
00:40:53 [00:40:53] 牌河 top discard ?
00:41:22 [00:41:20] 打 西，並按「聽」　（手牌 一萬二萬二筒三筒四筒四筒五筒六筒四條四條六條八條九條西，金牌 七筒六筒二條）
00:42:04 [00:42:04] 結算截圖存到 result\20261005_004204.png
00:42:04 [00:42:04] 本局牌河：me 發西；right 南（副露 1）；top ?；left （累計自動收集 0 張牌河模板）
00:42:16 [00:42:16] 換三張：換 東、西、八條　（手牌 一萬，金牌 四萬）
"""


def test_parse_log_splits_games_and_keeps_rivers_before_each_discard(tmp_path):
    path = tmp_path / "log_20261005.txt"
    path.write_text(LOG, encoding="utf-8")
    first, second = parse_log(path)
    assert first.stamp == "20261005_004204" and second.stamp is None
    assert first.melds == (0, 1, 0, 0)
    opening, declare = first.decisions
    assert opening.rivers == ((), (), (), ()) and len(opening.hand) == 17
    assert declare.declare and declare.tile == parse("3z")[0]
    assert declare.rivers == ((parse("6z")[0],), (parse("2z")[0],), (), ())  # 認不出的牌略過
    assert declare.gold == tuple(parse("7p6p2s"))


@pytest.mark.parametrize("hand, win, shape", [
    ("23m456789p123777s11z", "1m", "兩面"),
    ("13m456789p123777s11z", "2m", "嵌張"),
    ("12m456789p123777s11z", "3m", "邊張"),
    ("11m456789p123s55s111z", "1m", "雙碰"),
    ("123m456789p123s5s111z", "5s", "單吊"),
])
def test_wait_shapes(hand, win, shape):
    assert main_shape(wait_shapes(parse(hand), parse(win)[0])) == shape


def test_fill_hidden_only_returns_winning_completions():
    # 一萬、蓋住的一張、胡三萬：只有補二萬能胡
    tiles = tuple(parse("1m456789p123777s11z")) + (None,)
    assert fill_hidden(tiles, parse("3m")[0]) == [list(tiles[:-1]) + [parse("2m")[0]]]


def test_suji_against_the_winners_river():
    river = tuple(parse("4m"))
    assert is_suji(parse("1m")[0], river) and is_suji(parse("7m")[0], river)
    assert not is_suji(parse("5m")[0], river) and not is_suji(parse("1z")[0], river)


def _reader_and(name):
    path = TRAIN_DIR / f"{name}.png"
    if not path.exists() or not (TRAIN_DIR / "settlement_templates").exists():
        pytest.skip("缺少訓練截圖或結算模板")
    return SettlementReader.load(), read_image(path)


def test_settlement_reads_winner_hand_dealer_and_declare():
    # 10/5 01:06：北家胡九萬（有聽牌台），莊家在南
    reader, image = _reader_and("經典_結算_胡牌者聽牌")
    settlement = reader.read(image)
    north = settlement.rows[3]
    assert north.tiles == tuple(parse("65711124335478m22s")[i] for i in range(16))
    assert north.win_tile == parse("9m")[0]
    assert [row.dealer for row in settlement.rows] == [False, True, False, False]
    assert settlement.winner_declared
    assert all(row.win_tile is None for row in settlement.rows[:3])


def test_settlement_tiles_under_the_reward_card_are_unknown():
    # 10/5 01:24：獎勵卡片蓋住南、西兩列左邊幾張；南家胡東、沒宣告
    reader, image = _reader_and("經典_結算_卡片蓋牌")
    settlement = reader.read(image)
    south = settlement.rows[1]
    assert south.win_tile == parse("1z")[0] and None in south.tiles
    assert not settlement.winner_declared
    hands = fill_hidden(south.tiles, south.win_tile)
    assert hands and {main_shape(wait_shapes(h, south.win_tile)) for h in hands} == {"雙碰"}


def test_parse_log_reads_live_declared_seats(tmp_path):
    path = tmp_path / "log_20261006.txt"
    hand = "一萬二萬三萬四萬五萬六萬七萬八萬九萬一筒二筒三筒四筒五筒六筒"
    lines = ["22:00:00 [22:00:00] 自動模式啟動，AI mc",
             "22:00:01 [22:00:01] 換三張：不換　（手牌 一萬）",
             f"22:00:05 [22:00:05] 打 東　（手牌 {hand}東東）",
             f"22:00:09 [22:00:09] 打 東　（手牌 {hand}東北，宣告 上家、下家）"]
    path.write_text("".join(line + chr(10) for line in lines), encoding="utf-8")
    game, = parse_log(path)
    assert [d.declared for d in game.decisions] == [frozenset(), frozenset({1, 3})]
