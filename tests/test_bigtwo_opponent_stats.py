from bigtwo.cards import parse_cards
from bigtwo.combos import classify
from bigtwo.opponent_stats import events_of, games_from_logs, summarize


def c(text):
    return classify(parse_cards(text))


def test_passes_inferred_from_turn_order():
    plays = [(1, c("5d")), (3, c("9d")), (3, c("4c 4h"))]
    kinds = [(e.seat, e.kind) for e in events_of(plays)]
    assert kinds == [(1, "lead"), (2, "pass"), (3, "follow"), (0, "pass"), (1, "pass"), (2, "pass"), (3, "lead")]


def test_inconsistent_game_rejected():
    assert events_of([(1, c("Kc Kh")), (0, c("3s"))]) is None


def test_own_play_placed_at_button_press(tmp_path):
    log = tmp_path / "bigtwo_log_x.txt"
    log.write_text("\n".join([
        "[00:00:01] 換三張：手牌 … → 建議不換",
        "[00:00:02] 按「出牌」",
        "[00:00:03] 下家出 單張 方塊5",
        "[00:00:04] 自己出 單張 梅花3",
    ]), encoding="utf-8")
    games = games_from_logs([log])
    assert [seat for seat, _ in games[0]] == [0, 1]


def test_summarize_ratios():
    stats = summarize(events_of([(1, c("5d")), (3, c("9d")), (3, c("4c 4h"))]))
    assert stats["帶頭出單張"] == 0.5 and stats["面對單張時壓"] == 0.25   # 對家 PASS 5♦、上家壓、下家與對家 PASS 9♦
