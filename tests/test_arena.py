import csv

import pytest

from eval.arena import (
    PairedGame,
    compare,
    make_policy,
    payments,
    play_pair,
    summarize,
)
from game.scoring import Score
from game.simulator import GameResult


class _Table:
    def __init__(self, dealer):
        self.dealer = dealer


def result(winner, discarder, tai, dealer=3):
    return GameResult(state=_Table(dealer), steps=0, winner=winner,
                      win_by_discard=discarder is not None, score=Score(tai, ()), discarder=discarder)


def test_payments_follow_base_plus_tai():
    assert payments(result(1, 2, 5)) == [0, 7, -7, 0]            # 放槍：底 2 + 5 台
    assert payments(result(3, None, 2)) == [-4, -4, -4, 12]      # 莊家自摸：三家各付
    assert payments(GameResult(state=_Table(0), steps=0)) == [0, 0, 0, 0]  # 流局


def test_dealer_pays_the_dealer_tai_when_someone_else_self_draws():
    # 160658 結算：西家自摸 7 台（含莊家 1），閒家各付 400、南家（莊）付 450
    pay = payments(result(2, None, 6, dealer=1))
    assert pay == [-8, -9, 25, -8]


def test_policy_specs():
    assert all(make_policy(spec) is not None for spec in ("rule", "gold:8", "blind", "fold:1", "nodeclare", "ev", "ev:0.5", "mc", "mc:8"))
    for bad in ("gold", "rule:1", "nope"):
        with pytest.raises(ValueError):
            make_policy(bad)


def test_identical_policies_give_zero_difference():
    game = play_pair(5, "gold:0", "gold:0")
    assert game.seat == 1 and game.variant_net == game.baseline_net


def test_summary_statistics():
    games = [PairedGame(0, 0, 4, 0, True, False), PairedGame(1, 1, -2, -2, False, True)]
    summary = summarize(games)
    assert summary.difference == 2 and summary.variant_mean == 1
    assert summary.win_rate == 0.5 and summary.deal_in_rate == 0.5
    assert summary.margin > 0


def test_compare_writes_each_game_to_csv(tmp_path):
    out = tmp_path / "games.csv"
    summary = compare("gold:8", "gold:0", games=2, out=out, progress_every=0)
    rows = list(csv.reader(out.open(encoding="utf-8")))
    assert summary.games == 2 and len(rows) == 2
    assert rows[0][:2] == ["gold:8", "gold:0"]
