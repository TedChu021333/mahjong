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
}


def compare(variant, baseline, deals: int, seed: int = 0):
    (variant, variant_swap), (baseline, baseline_swap) = variant, baseline
    rng = Random(seed)
    diffs, wins, base_wins = [], 0, 0
    for _ in range(deals):
        hands = deal(rng)
        game_seed = rng.random()
        base = play_game([baseline] * 4, Random(game_seed), hands=hands, swappers=[baseline_swap] * 4)
        for seat in range(4):
            policies, swappers = [baseline] * 4, [baseline_swap] * 4
            policies[seat], swappers[seat] = variant, variant_swap
            result = play_game(policies, Random(game_seed), hands=hands, swappers=swappers)
            diffs.append(result.scores[seat] - base.scores[seat])
            wins += result.winner == seat
            base_wins += base.winner == seat
    n = len(diffs)
    mean = sum(diffs) / n
    sd = sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    # 同一副牌四個座位的結果相關，誤差以「副」為單位計
    per_deal = [sum(diffs[i:i + 4]) / 4 for i in range(0, n, 4)]
    m = sum(per_deal) / deals
    se = sqrt(sum((d - m) ** 2 for d in per_deal) / (deals - 1) / deals)
    return mean, 1.96 * se, wins / n, base_wins / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="rule", choices=POLICIES)
    parser.add_argument("--baseline", default="smallest", choices=POLICIES)
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    mean, ci, win, base_win = compare(POLICIES[args.variant], POLICIES[args.baseline],
                                      args.deals, args.seed)
    print(f"{args.variant} 對 {args.baseline}：{args.deals} 副 × 4 座位，每局差 {mean:+.2f} 張 ± {ci:.2f}；"
          f"勝率 {win:.1%}（對照 {base_win:.1%}）")


if __name__ == "__main__":
    main()
