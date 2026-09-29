"""配對比較兩種策略：同一副牌各打一次，只差在其中一個座位用的策略。

    python -m eval.arena --variant gold:8 --baseline gold:0 --games 2000

每個種子打兩局：四家都是 baseline、以及把一個座位換成 variant（座位與莊家依種子輪替）；
比較該座位兩局的輸贏差。起手牌相同，運氣成分大多抵銷，比各打各的少很多局就能分出高下。
輸贏以台灣麻將「底 + 台」計（預設底 3、每台 1，對應 300/100）：放槍由放槍者付，
自摸三家各付；尚未計入非莊自摸時莊家多付的莊家台。

策略代號：
    rule          規則式 AI（目前的預設參數）
    gold:<n>      規則式 AI，打出金牌的代價視同少 n 張有效進張
    blind         規則式 AI，但忽略對手宣告聽牌（看不到「聽」標記時的行為）
    fold:<n>      規則式 AI，有對手宣告聽牌時，自己向聽數 ≥n 就棄胡
    nodeclare     規則式 AI，但聽牌時不宣告（不拿聽牌 1 台，手牌不鎖、還能防守）
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from random import Random

from agent.rule_agent import choose_rule_action_for_player
from game.rules import Action, ActionType
from game.simulator import GameResult, PlayerPolicy, play_game

BASE = 3
"""底，以「台」為單位（人氣館 300/100 → 底 3 台）。"""


def make_policy(spec: str) -> PlayerPolicy:
    name, _, arg = spec.partition(":")
    if name == "rule" and not arg:
        return choose_rule_action_for_player
    if name == "gold" and arg:
        return partial(_gold_policy, penalty=float(arg))
    if name == "blind" and not arg:
        return _blind_policy
    if name == "nodeclare" and not arg:
        return _no_declare_policy
    if name == "fold" and arg:
        return partial(_fold_policy, shanten=int(arg))
    raise ValueError(f"未知的策略代號：{spec}")


def _no_declare_policy(state, player, rng):
    action = choose_rule_action_for_player(state, player, rng)
    if action is not None and action.kind == ActionType.DECLARE:
        return Action(ActionType.DISCARD, tile=action.tile)
    return action


def _fold_policy(state, player, rng, shanten):
    return choose_rule_action_for_player(state, player, rng, declared_fold_shanten=shanten)


def _blind_policy(state, player, rng):
    return choose_rule_action_for_player(state, player, rng, use_declared=False)


def _gold_policy(state, player, rng, penalty):
    return choose_rule_action_for_player(state, player, rng, gold_penalty=penalty)


def payments(result: GameResult, base: int = BASE) -> list[int]:
    """每家這局的輸贏（台）。"""
    pay = [0, 0, 0, 0]
    if result.winner is None or result.score is None:
        return pay
    amount = base + result.score.tai
    losers = [result.discarder] if result.win_by_discard else \
        [player for player in range(4) if player != result.winner]
    for loser in losers:
        pay[loser] -= amount
        pay[result.winner] += amount
    return pay


@dataclass(frozen=True)
class PairedGame:
    seed: int
    seat: int
    variant_net: int
    baseline_net: int
    variant_won: bool
    variant_dealt_in: bool
    baseline_won: bool = False
    baseline_dealt_in: bool = False


def play_pair(seed: int, variant: str, baseline: str) -> PairedGame:
    seat, dealer = seed % 4, (seed // 4) % 4
    base_policy, var_policy = make_policy(baseline), make_policy(variant)

    def mixed(state, player, rng):
        return (var_policy if player == seat else base_policy)(state, player, rng)

    reference = play_game(Random(seed), player_policy=base_policy, dealer=dealer, swap=True)
    trial = play_game(Random(seed), player_policy=mixed, dealer=dealer, swap=True)
    return PairedGame(seed, seat, payments(trial)[seat], payments(reference)[seat],
                      trial.winner == seat, trial.discarder == seat,
                      reference.winner == seat, reference.discarder == seat)


@dataclass(frozen=True)
class Summary:
    games: int
    variant_mean: float
    baseline_mean: float
    difference: float
    margin: float
    """差值的 95% 信賴區間半寬。"""
    win_rate: float
    deal_in_rate: float
    baseline_win_rate: float = 0.0
    baseline_deal_in_rate: float = 0.0


def summarize(games: list[PairedGame]) -> Summary:
    n = len(games)
    if n < 2:
        raise ValueError("至少需要 2 局才能估計信賴區間")
    diffs = [game.variant_net - game.baseline_net for game in games]
    mean = sum(diffs) / n
    variance = sum((diff - mean) ** 2 for diff in diffs) / (n - 1)
    return Summary(
        games=n,
        variant_mean=sum(game.variant_net for game in games) / n,
        baseline_mean=sum(game.baseline_net for game in games) / n,
        difference=mean,
        margin=1.96 * (variance / n) ** 0.5,
        win_rate=sum(game.variant_won for game in games) / n,
        deal_in_rate=sum(game.variant_dealt_in for game in games) / n,
        baseline_win_rate=sum(game.baseline_won for game in games) / n,
        baseline_deal_in_rate=sum(game.baseline_dealt_in for game in games) / n,
    )


def describe(summary: Summary) -> str:
    return (f"{summary.games} 局：每局 {summary.variant_mean:+.3f} 台"
            f"（對照 {summary.baseline_mean:+.3f}），差 {summary.difference:+.3f}"
            f" ± {summary.margin:.3f}；胡牌率 {summary.win_rate:.1%}"
            f"（對照 {summary.baseline_win_rate:.1%}）、放槍率 {summary.deal_in_rate:.1%}"
            f"（對照 {summary.baseline_deal_in_rate:.1%}）")


def compare(variant: str, baseline: str, games: int, start: int = 0, workers: int = 1,
            out: Path | None = None, progress_every: int = 200) -> Summary:
    """out 有給時每局結果立即寫進 CSV，中斷後已跑完的部分不會遺失。"""
    make_policy(variant), make_policy(baseline)  # 先檢查代號
    task = partial(play_pair, variant=variant, baseline=baseline)
    seeds = range(start, start + games)
    results: list[PairedGame] = []
    writer = file = None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        file = out.open("a", newline="", encoding="utf-8")
        writer = csv.writer(file)
    began = time.time()
    try:
        with Pool(workers) if workers > 1 else _Serial() as pool:
            for game in pool.imap_unordered(task, seeds, chunksize=4):
                results.append(game)
                if writer is not None:
                    writer.writerow([variant, baseline, game.seed, game.seat, game.variant_net,
                                     game.baseline_net, int(game.variant_won),
                                     int(game.variant_dealt_in), int(game.baseline_won),
                                     int(game.baseline_dealt_in)])
                    file.flush()
                if progress_every and len(results) % progress_every == 0 and len(results) > 1:
                    print(f"[{time.time() - began:.0f}s] {describe(summarize(results))}",
                          flush=True)
    finally:
        if file is not None:
            file.close()
    return summarize(results)


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def imap_unordered(func, items, chunksize=1):
        return map(func, items)


def main() -> None:
    parser = argparse.ArgumentParser(description="配對比較兩種策略")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline", default="rule")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--start", type=int, default=0, help="第一個種子")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, help="逐局結果附加寫入的 CSV")
    args = parser.parse_args()
    summary = compare(args.variant, args.baseline, args.games, args.start, args.workers,
                      args.out)
    print(f"{args.variant} 對 {args.baseline}：{describe(summary)}")


if __name__ == "__main__":
    main()
