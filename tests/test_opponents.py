from eval.opponents import HandRecord, apply_settlement, metrics, parse_log, report, sim_hand

LOG = """\
00:02:24 [00:02:24] 自動模式啟動，AI nn，版本 3db4540，湊牌優先 6 手，護欄開
00:02:52 [00:02:52] 打 東　（手牌 三萬四萬，金牌 二筒）
00:02:59 [00:02:59] 牌河 right discard 九萬
00:03:05 [00:03:05] 牌河 top discard 二筒
00:03:06 [00:03:06] 牌河 right discard 白
00:03:07 [00:03:07] 牌河 right discard 一條
00:04:16 [00:04:15] 上家宣告聽牌（第一次認到；牌河 自己 6、下家 4、對家 4、上家 5）
00:05:37 [00:05:37] 結算截圖存到 result\\20261022_000537.png
00:05:37 [00:05:37] 本局牌河：me 東五條東；right 九萬白一條；top 二筒（副露 1）；left 八條二筒五筒三萬三條三筒五筒（累計自動收集 0 張）
00:06:10 [00:06:10] 打 南　（手牌 三萬四萬，金牌 二筒）
00:30:52 [00:30:52] 新的一局（上一局沒有結算畫面），重設牌河：me 南
00:31:00 [00:31:00] 提示模式啟動，AI nn
00:31:10 [00:31:10] 結算截圖存到 result\\20261022_003110.png
00:31:10 [00:31:10] 本局牌河：me 東；right ；top ；left 東
"""


def test_log_is_split_into_hands_and_hint_mode_is_skipped(tmp_path):
    path = tmp_path / "log_20261022.txt"
    path.write_text(LOG, encoding="utf-8")
    hands = parse_log(path)
    assert len(hands) == 2
    first, second = hands
    assert first.stamp == "20261022_000537" and not first.drawn
    assert first.declares == {3: 6}
    assert first.turns == 7                       # 上家牌河 7 張
    assert first.first_discards == {1: (8, 33), 2: (10,)}
    assert first.melds == (0, 0, 1, 0)            # 記錄檔推算的副露數
    assert second.drawn and second.stamp is None and second.turns == 1


def test_settlement_gives_winner_discarder_and_payment():
    read = {"amounts": [0, -270, 0, 270], "sizes": [16] * 4, "declared": True, "shape": "兩面"}
    hand = apply_settlement(HandRecord(stamp="x"), read)
    assert (hand.winner, hand.discarder, hand.self_draw, hand.paid) == (3, 1, False, 270)
    assert hand.winner_declared and hand.shape == "兩面"
    self_draw = apply_settlement(HandRecord(stamp="x"), {**read, "amounts": [-130, -130, 570, -310]})
    assert self_draw.self_draw and self_draw.paid == 130   # 自摸取付最少的（不是莊家）那家
    assert apply_settlement(HandRecord(stamp="x"), None).winner is None


def test_simulated_hand_and_report():
    hands = [sim_hand(seed, me="rule") for seed in range(3)]
    for hand in hands:
        assert hand.drawn or hand.winner is not None
        assert hand.turns and len(hand.melds) == 4
    text = report(hands, hands)
    assert "對手胡牌率" in text and "對手按聽巡目" in text
    assert dict(metrics(hands))["局數（有結果）"] == "3"
