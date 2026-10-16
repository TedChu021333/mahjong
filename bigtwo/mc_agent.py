"""蒙地卡羅大老二 AI：把看不到的牌隨機分給三家（張數照實際），每個候選出法用同一批發牌
以規則式 AI 模擬到結束，選自己平均得分最高的。"""
from __future__ import annotations

from random import Random

from .agents import plan_hand, rule_policy
from .cards import NUM_CARDS
from .combos import Combo, Kind
from .game import PLAYERS, Play, View, continue_game


def _candidates(view: View, limit: int) -> list[Combo | None]:
    moves = list(view.moves)
    plan = set(plan_hand(view.hand))
    rule_choice = rule_policy(view, Random(0))

    def priority(m: Combo):
        # 拆牌規劃裡的組合、規則式的選擇優先，其餘依小到大
        return (m != rule_choice, m not in plan, m.size == 5 and m.kind != Kind.STRAIGHT, m.size != 5, m.kind, m.key)

    moves.sort(key=priority)
    picked: list[Combo | None] = []
    seen = set()
    for m in moves:
        sig = (m.kind, m.key[:1] if m.size == 5 else m.key)   # 同一種五張牌型只試一種湊法
        if sig in seen and m not in plan and m != rule_choice:
            continue
        seen.add(sig)
        picked.append(m)
        if len(picked) >= limit:
            break
    if view.can_pass:
        picked.append(None)
    return picked


def make_mc_policy(samples: int = 24, limit: int = 8, rollout=rule_policy):
    def mc_policy(view: View, rng: Random) -> Combo | None:
        candidates = _candidates(view, limit)
        if len(candidates) == 1:
            return candidates[0]
        played = view.played
        unseen = [c for c in range(NUM_CARDS) if c not in played and c not in view.hand]
        others = [(view.seat + k) % PLAYERS for k in range(1, PLAYERS)]
        totals = [0.0] * len(candidates)
        for _ in range(samples):
            rng.shuffle(unseen)
            hands: list[list[int]] = [[] for _ in range(PLAYERS)]
            hands[view.seat] = list(view.hand)
            start = 0
            for s in others:
                hands[s] = sorted(unseen[start:start + view.counts[s]])
                start += view.counts[s]
            seed = rng.random()
            for i, move in enumerate(candidates):
                result = continue_game(view, hands, move, [rollout] * PLAYERS, Random(seed))
                totals[i] += result.scores[view.seat]
        return candidates[max(range(len(candidates)), key=lambda i: totals[i])]

    return mc_policy
