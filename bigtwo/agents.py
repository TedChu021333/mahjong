"""大老二基準 AI：隨機、出最小、規則式（拆牌規劃）。"""
from __future__ import annotations

from random import Random

from .cards import NUM_RANKS, RANK_TWO, rank_of, suit_of
from .combos import STRAIGHTS, Combo, Kind, beats, classify
from .game import View


def random_policy(view: View, rng: Random) -> Combo | None:
    options: list[Combo | None] = list(view.moves)
    if view.can_pass:
        options.append(None)
    return rng.choice(options)


def smallest_policy(view: View, rng: Random) -> Combo | None:
    """能出就出最小的（帶頭時出含最小那張的單張）。"""
    return min(view.moves, key=_strength) if view.moves else None


def _strength(combo: Combo) -> tuple:
    return (combo.size == 5, combo.kind, combo.key)


# ---------- 規則式：把手牌拆成最少手數的組合 ----------

def _five_card_candidates(hand: list[int]) -> list[Combo]:
    """拆牌用的五張組合：每種點數組合只取一種挑法（盡量用落單、花色小的牌），避免組合爆炸。"""
    groups = [[] for _ in range(NUM_RANKS)]
    for c in sorted(hand):
        groups[rank_of(c)].append(c)
    out = []
    for ranks in STRAIGHTS:
        if all(groups[r] for r in ranks):
            plain = [min(groups[r], key=lambda c: (len(groups[rank_of(c)]) > 1, c)) for r in ranks]
            out.append(classify(plain))
            # 同花順另外試
            for suit in range(4):
                flush = [next((c for c in groups[r] if suit_of(c) == suit), None) for r in ranks]
                if None not in flush:
                    out.append(classify(flush))
    pairs = [g[:2] for g in groups if len(g) >= 2]
    for r, g in enumerate(groups):
        if len(g) >= 3:
            for p in pairs:
                if rank_of(p[0]) != r:
                    out.append(classify(g[:3] + p))
                    break
        if len(g) == 4:
            others = sorted((c for c in hand if rank_of(c) != r),
                            key=lambda c: (len(groups[rank_of(c)]) > 1, c))
            if others:
                out.append(classify(g + others[:1]))
    return [c for c in out if c is not None]


def _small_groups(hand: list[int]) -> list[Combo]:
    groups = [[] for _ in range(NUM_RANKS)]
    for c in sorted(hand):
        groups[rank_of(c)].append(c)
    out = []
    for g in groups:
        if len(g) == 4:
            out += [classify(g[:2]), classify(g[2:])]
        elif g:
            out.append(classify(g))
    return out


def plan_hand(hand) -> list[Combo]:
    """拆成手數最少的組合；同手數時落單小牌越少越好。"""
    hand = sorted(hand)
    best: tuple | None = None

    def score(combos: list[Combo]) -> tuple:
        low_singles = sum(1 for c in combos if c.kind == Kind.SINGLE and rank_of(c.cards[0]) < 8)
        return (len(combos), low_singles)

    def search(rest: list[int], chosen: list[Combo], depth: int):
        nonlocal best
        combos = chosen + _small_groups(rest)
        s = score(combos)
        if best is None or s < best[0]:
            best = (s, combos)
        if depth == 2:
            return
        for five in _five_card_candidates(rest):
            if chosen and _strength(five) < _strength(chosen[-1]):
                continue
            left = list(rest)
            for c in five.cards:
                left.remove(c)
            search(left, chosen + [five], depth + 1)

    search(hand, [], 0)
    return best[1]


def rule_policy(view: View, rng: Random) -> Combo | None:
    if not view.moves:
        return None
    moves = set(view.moves)
    plan = plan_hand(view.hand)
    nxt = view.counts[(view.seat + 1) % 4]
    danger = min(n for s, n in enumerate(view.counts) if s != view.seat)

    if view.last is None:
        legal_plan = [c for c in plan if c in moves]
        if len(plan) == 1 and legal_plan:
            return legal_plan[0]
        if len(plan) == 2 and len(legal_plan) == 2:   # 先出比較大的那手拿回出牌權，留最後一手
            return max(legal_plan, key=_strength)
        pool = legal_plan or list(view.moves)
        if nxt == 1:         # 下家剩一張：不要帶小單張
            non_single = [c for c in pool if c.kind != Kind.SINGLE]
            if non_single:
                return min(non_single, key=_strength)
            return max(view.moves, key=lambda c: (c.kind == Kind.SINGLE, c.key))
        # 帶含最小牌的那一手；五張牌型優先（一次出多張）
        lowest = min(view.hand)
        with_lowest = [c for c in pool if lowest in c.cards]
        if with_lowest:
            return max(with_lowest, key=lambda c: (c.size, -c.key[0] if c.size < 5 else 0))
        return min(pool, key=_strength)

    # 跟牌
    in_plan = sorted((c for c in plan if c in moves), key=_strength)
    if nxt == 1 and view.last.kind == Kind.SINGLE:
        return max(view.moves, key=_strength)
    if in_plan:
        choice = in_plan[0]
        # 別人還多張時，不急著用 2 壓小牌
        if (choice.kind == Kind.SINGLE and rank_of(choice.cards[0]) == RANK_TWO
                and danger > 4 and len(plan) > 3 and rank_of(view.last.cards[-1]) < 8):
            return None
        return choice
    if danger <= 3 or len(view.hand) <= 4:
        return min(view.moves, key=_strength)
    return None
