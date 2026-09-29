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


def result(winner, discarder, tai):
    return GameResult(state=None, steps=0, winner=winner, win_by_discard=discarder is not None,
                      score=Score(tai, ()), discarder=discarder)


def test_payments_follow_base_plus_tai():
    assert payments(result(1, 2, 5)) == [0, 8, -8, 0]            # 放槍：底 3 + 5 台
    assert payments(result(0, None, 2)) == [15, -5, -5, -5]      # 自摸：三家各付
    assert payments(GameResult(state=None, steps=0)) == [0, 0, 0, 0]  # 流局


def test_policy_specs():
    assert all(make_policy(spec) is not None for spec in ("rule", "gold:8", "blind", "fold:1", "nodeclare"))
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
