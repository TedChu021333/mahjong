"""蒙地卡羅大老二 AI：把看不到的牌隨機分給三家（張數照實際），每個候選出法用同一批發牌
以規則式 AI 模擬到結束。

預設照規則式的選擇；別的出法要比它好超過 2 倍標準誤差才改，幾個都明顯較好時，在和最好的那個差不到
2 倍標準誤差的裡面選最小的一手。直接取平均最高時，10/16 實戰帶頭出了一對 9（一對 7、9、10 其實
差不多），跟牌時還拆一對 10 去壓 5（同一局面重算 6 次有 4 次選 6，是雜訊）。"""
from __future__ import annotations

from math import sqrt
from random import Random

from .agents import _small_groups, _strength, plan_hand, rule_policy
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
    # 不拆牌的單張／對子／三條（同點數剛好那幾張），每種最小的一手一定列入：
    # 一對 7 被規劃進葫蘆時，帶頭的候選曾只有一對 9、10、J
    legal = set(view.moves)
    lowest = {}
    for m in sorted((g for g in _small_groups(list(view.hand)) if g in legal), key=_strength):
        lowest.setdefault(m.kind, m)
    for m in [m for m in moves if m == rule_choice] + sorted(lowest.values(), key=_strength) + moves:
        if m in picked:
            continue
        sig = (m.kind, m.key[:1] if m.size == 5 else m.key)   # 同一種五張牌型只試一種湊法
        if sig in seen and m not in plan and m != rule_choice and m not in lowest.values():
            continue
        seen.add(sig)
        picked.append(m)
        if len(picked) >= limit:
            break
    if view.can_pass:
        picked.append(None)
    return picked


def _paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """a − b 的平均與標準誤差（同一批發牌）。"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    return mean, sqrt(var / n)


def choose(candidates: list[Combo | None], scores: list[list[float]], z: float = 2.0) -> Combo | None:
    """candidates[0] 是規則式的選擇。"""
    base = scores[0]
    better = [i for i in range(1, len(candidates))
              if (lambda m, se: m > z * se and m > 0)(*_paired(scores[i], base))]
    if not better:
        return candidates[0]
    top = max(better, key=lambda i: sum(scores[i]))
    close = [i for i in better if (lambda m, se: m <= z * se)(*_paired(scores[top], scores[i]))]
    return candidates[min(close, key=lambda i: (candidates[i] is not None, _strength(candidates[i]) if candidates[i] else ()))]


def make_mc_policy(samples: int = 64, limit: int = 10, rollout=rule_policy, z: float = 2.0):
    def mc_policy(view: View, rng: Random) -> Combo | None:
        candidates = _candidates(view, limit)
        if len(candidates) == 1:
            return candidates[0]
        played = view.played
        unseen = [c for c in range(NUM_CARDS) if c not in played and c not in view.hand]
        others = [(view.seat + k) % PLAYERS for k in range(1, PLAYERS)]
        scores: list[list[float]] = [[] for _ in candidates]
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
                scores[i].append(result.scores[view.seat])
        return choose(candidates, scores, z)

    return mc_policy
