"""蒙地卡羅大老二 AI：把看不到的牌隨機分給三家（張數照實際），每個候選出法用同一批發牌
以規則式 AI 模擬到結束。

預設照規則式的選擇；別的出法要比它好超過 2 倍標準誤差才改，幾個都明顯較好時，在和最好的那個差不到
2 倍標準誤差的裡面選最小的一手。直接取平均最高時，10/16 實戰帶頭出了一對 9（一對 7、9、10 其實
差不多），跟牌時還拆一對 10 去壓 5（同一局面重算 6 次有 4 次選 6，是雜訊）。"""
from __future__ import annotations

import time
from math import exp, sqrt
from random import Random

from .agents import _small_groups, _strength, plan_hand, rule_policy
from .cards import NUM_CARDS, rank_of
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


def pass_constraints(view: View) -> list[tuple[int, Combo, int]]:
    """(座位, 他當時沒壓的牌, 之後的紀錄起點)。只用單張、對子、三條（五張牌型太難判斷）。"""
    out = []
    last = None
    for i, play in enumerate(view.history):
        if play.combo is not None:
            last = play.combo
        elif last is not None and last.size <= 3 and play.seat != view.seat:
            out.append((play.seat, last, i))
    return out


def _can_beat(cards, combo: Combo) -> bool:
    if combo.size == 1:
        return any(c > combo.cards[0] for c in cards)
    groups: dict[int, list[int]] = {}
    for c in cards:
        groups.setdefault(rank_of(c), []).append(c)
    if combo.size == 2:
        return any(len(g) >= 2 and max(g) > combo.key[0] for g in groups.values())
    return any(len(g) >= 3 and r > combo.key[0] for r, g in groups.items())


def _violations(hands, constraints, history) -> int:
    count = 0
    for seat, combo, index in constraints:
        later = [c for p in history[index:] if p.seat == seat and p.combo for c in p.combo.cards]
        if _can_beat(list(hands[seat]) + later, combo):
            count += 1
    return count


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


PASS_REJECT = 0.7
"""抽到「某家明明壓得過、當時卻 PASS」的分法時，有這個機率丟掉重抽（有人會故意不壓，不全丟）。"""
MAX_TRIES = 30


def hand_strength(cards) -> float:
    """對手手牌的強度：2 各 1、A 各 0.5、三條 1（有對子湊葫蘆再 +0.5）、鐵支 2。"""
    counts = [0] * 13
    for c in cards:
        counts[rank_of(c)] += 1
    pairs = sum(1 for k in counts if k >= 2)
    score = counts[12] + 0.5 * counts[11]
    for k in counts:
        if k == 4:
            score += 2
        elif k == 3:
            score += 1 + (0.5 if pairs > 1 else 0)
    return score


STRONG_BIAS = 0.0
"""發牌偏向：每次抽 STRONG_PICK 種分法，以 exp(偏向 × 三家強度和) 的權重挑一種。0＝完全隨機。"""
STRONG_PICK = 4


def make_mc_policy(samples: int = 64, limit: int = 10, rollout=rule_policy, z: float = 2.0,
                   pass_reject: float = PASS_REJECT, time_limit: float | None = None,
                   max_samples: int = 2000, opponent_rollout=None, strong_bias: float = STRONG_BIAS):
    """time_limit：實戰用，至少模擬 samples 次，之後繼續到用完這麼多秒（最多 max_samples 次）。
    10/17 實戰大家還有 11~12 張時用 2♠ 壓 2♦：1500 次模擬 PASS 好 1.46±0.82 張，但 64 次分不出來、
    照規則式出 2；這個局面只有兩個候選，64 次只花 0.3 秒，限時 8 秒用不到一成。"""
    def mc_policy(view: View, rng: Random) -> Combo | None:
        candidates = _candidates(view, limit)
        if len(candidates) == 1:
            return candidates[0]
        played = view.played
        unseen = [c for c in range(NUM_CARDS) if c not in played and c not in view.hand]
        others = [(view.seat + k) % PLAYERS for k in range(1, PLAYERS)]
        scores: list[list[float]] = [[] for _ in candidates]
        constraints = pass_constraints(view) if pass_reject > 0 else []
        deadline = None if time_limit is None else time.monotonic() + time_limit
        done = 0
        policies = [opponent_rollout or rollout] * PLAYERS
        policies[view.seat] = rollout

        def draw() -> list[list[int]]:
            for _ in range(MAX_TRIES):
                rng.shuffle(unseen)
                hands: list[list[int]] = [[] for _ in range(PLAYERS)]
                hands[view.seat] = list(view.hand)
                start = 0
                for s in others:
                    hands[s] = sorted(unseen[start:start + view.counts[s]])
                    start += view.counts[s]
                bad = _violations(hands, constraints, view.history) if constraints else 0
                if not bad or rng.random() >= 1 - (1 - pass_reject) ** bad:
                    break
            return hands

        while done < samples or (deadline is not None and done < max_samples and time.monotonic() < deadline):
            done += 1
            if strong_bias > 0:
                options = [draw() for _ in range(STRONG_PICK)]
                weights = [exp(strong_bias * sum(hand_strength(h[s]) for s in others)) for h in options]
                hands = rng.choices(options, weights)[0]
            else:
                hands = draw()
            seed = rng.random()
            for i, move in enumerate(candidates):
                result = continue_game(view, hands, move, policies, Random(seed))
                scores[i].append(result.scores[view.seat])
        return choose(candidates, scores, z)

    return mc_policy
