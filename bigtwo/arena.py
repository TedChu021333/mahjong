"""配對比較兩種大老二策略：同一副牌讓變體輪流坐四個座位（其他三家都是基準），
與四家都是基準時同座位的得分相減。

python -m bigtwo.arena --variant rule --baseline smallest --deals 2000
"""
from __future__ import annotations

import argparse
from math import sqrt
from random import Random

from .agents import random_policy, rule_policy, rule_swap, smallest_policy
from .cards import deal
from .game import no_swap, play_game
from .mc_agent import make_mc_policy

# 名稱 → (出牌策略, 換三張策略)
POLICIES = {
    "random": (random_policy, no_swap),
    "smallest": (smallest_policy, no_swap),
    "rule": (rule_policy, rule_swap),
    "rule-noswap": (rule_policy, no_swap),
    "mc": (make_mc_policy(), rule_swap),
    "mc-old": (make_mc_policy(samples=24, limit=8, z=0.0), rule_swap),   # 10/16 之前：直接取平均最高
}


def _play_deals(args):
    variant_name, baseline_name, jobs = args
    (variant, variant_swap), (baseline, baseline_swap) = POLICIES[variant_name], POLICIES[baseline_name]
    out = []
    for hands, game_seed in jobs:
        base = play_game([baseline] * 4, Random(game_seed), hands=hands, swappers=[baseline_swap] * 4)
        diffs, wins, base_wins = [], 0, 0
        for seat in range(4):
            policies, swappers = [baseline] * 4, [baseline_swap] * 4
            policies[seat], swappers[seat] = variant, variant_swap
            result = play_game(policies, Random(game_seed), hands=hands, swappers=swappers)
            diffs.append(result.scores[seat] - base.scores[seat])
            wins += result.winner == seat
            base_wins += base.winner == seat
        out.append((diffs, wins, base_wins))
    return out


def compare(variant: str, baseline: str, deals: int, seed: int = 0, workers: int = 1):
    """同一副牌讓 variant 輪流坐四個座位（其他三家 baseline），與四家都是 baseline 時同座位相減。"""
    rng = Random(seed)
    jobs = []
    for _ in range(deals):
        hands = deal(rng)
        jobs.append((hands, rng.random()))
    chunks = [(variant, baseline, jobs[k::workers]) for k in range(workers)]
    if workers == 1:
        results = [_play_deals(chunks[0])]
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            results = pool.map(_play_deals, chunks)
    rows = [row for chunk in results for row in chunk]
    diffs = [d for row in rows for d in row[0]]
    wins = sum(row[1] for row in rows)
    base_wins = sum(row[2] for row in rows)
    n = len(diffs)
    mean = sum(diffs) / n
    # 同一副牌四個座位的結果相關，誤差以「副」為單位計
    per_deal = [sum(row[0]) / 4 for row in rows]
    m = sum(per_deal) / deals
    se = sqrt(sum((d - m) ** 2 for d in per_deal) / (deals - 1) / deals)
    return mean, 1.96 * se, wins / n, base_wins / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="rule", choices=POLICIES)
    parser.add_argument("--baseline", default="smallest", choices=POLICIES)
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    mean, ci, win, base_win = compare(args.variant, args.baseline, args.deals, args.seed, args.workers)
    print(f"{args.variant} 對 {args.baseline}：{args.deals} 副 × 4 座位，每局差 {mean:+.2f} 張 ± {ci:.2f}；"
          f"勝率 {win:.1%}（對照 {base_win:.1%}）")


if __name__ == "__main__":
    main()
