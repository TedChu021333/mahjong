"""大老二牌型：判定、比大小、列出手上所有能出的組合。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations, product

from .cards import CLUB_THREE, NUM_RANKS, rank_of, suit_of
from .rules import DEFAULT_RULES, Rules


class Kind(IntEnum):
    SINGLE = 0
    PAIR = 1
    TRIPLE = 2
    STRAIGHT = 3
    FLUSH = 4
    FULL_HOUSE = 5
    FOUR = 6           # 鐵支（四張＋一張）
    STRAIGHT_FLUSH = 7


KIND_NAMES = {
    Kind.SINGLE: "單張", Kind.PAIR: "一對", Kind.TRIPLE: "三條", Kind.STRAIGHT: "順子",
    Kind.FLUSH: "同花", Kind.FULL_HOUSE: "葫蘆", Kind.FOUR: "鐵支", Kind.STRAIGHT_FLUSH: "同花順",
}
FIVE_CARD = (Kind.STRAIGHT, Kind.FLUSH, Kind.FULL_HOUSE, Kind.FOUR, Kind.STRAIGHT_FLUSH)
BOMBS = (Kind.FOUR, Kind.STRAIGHT_FLUSH)

# 合法順子的點數組合（點數 0=3 … 11=A、12=2），由小到大：A2345 最小、23456 最大，
# 沒有 JQKA2、QKA23、KA234。
STRAIGHTS: tuple[tuple[int, ...], ...] = (
    (11, 12, 0, 1, 2),
    *(tuple(range(start, start + 5)) for start in range(8)),
    (12, 0, 1, 2, 3),
)
_STRAIGHT_INDEX = {frozenset(ranks): i for i, ranks in enumerate(STRAIGHTS)}


@dataclass(frozen=True)
class Combo:
    kind: Kind
    cards: tuple[int, ...]   # 由小到大
    key: tuple[int, ...]     # 同牌型比大小用

    @property
    def size(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        from .cards import card_name
        return f"{KIND_NAMES[self.kind]} " + " ".join(card_name(c) for c in self.cards)


def classify(cards, rules: Rules = DEFAULT_RULES) -> Combo | None:
    """判定一組牌是什麼牌型；不是合法牌型回傳 None。"""
    cs = tuple(sorted(cards))
    n = len(cs)
    if len(set(cs)) != n:
        return None
    ranks = [rank_of(c) for c in cs]
    if n == 1:
        return Combo(Kind.SINGLE, cs, (cs[0],))
    if n == 2:
        return Combo(Kind.PAIR, cs, (cs[1],)) if ranks[0] == ranks[1] else None
    if n == 3:
        return Combo(Kind.TRIPLE, cs, (ranks[0],)) if len(set(ranks)) == 1 else None
    if n != 5:
        return None
    counts = {r: ranks.count(r) for r in set(ranks)}
    flush = len({suit_of(c) for c in cs}) == 1
    straight = _STRAIGHT_INDEX.get(frozenset(ranks)) if len(counts) == 5 else None
    if straight is not None:
        # 同一種順子比「最大的數字」那張的花色：以牌力最大的那張（含 2 時就是 2）
        key = (straight, cs[-1])
        return Combo(Kind.STRAIGHT_FLUSH if flush else Kind.STRAIGHT, cs, key)
    if sorted(counts.values()) == [2, 3]:
        triple = next(r for r, k in counts.items() if k == 3)
        return Combo(Kind.FULL_HOUSE, cs, (triple,))
    if sorted(counts.values()) == [1, 4]:
        quad = next(r for r, k in counts.items() if k == 4)
        return Combo(Kind.FOUR, cs, (quad,))
    if flush and rules.flush_enabled:
        return Combo(Kind.FLUSH, cs, tuple(reversed(cs)))
    return None


def beats(a: Combo, b: Combo, rules: Rules = DEFAULT_RULES) -> bool:
    """a 能不能壓 b。"""
    if a.size == b.size:
        if a.size == 5:
            return (a.kind, a.key) > (b.kind, b.key)
        return a.key > b.key
    return rules.bombs_beat_smaller and a.kind in BOMBS and b.size < 5


def _by_rank(hand) -> list[list[int]]:
    groups = [[] for _ in range(NUM_RANKS)]
    for c in sorted(hand):
        groups[rank_of(c)].append(c)
    return groups


def all_combos(hand, rules: Rules = DEFAULT_RULES) -> list[Combo]:
    """手上所有能組成的牌型（不重複）。"""
    groups = _by_rank(hand)
    out: list[Combo] = []
    for c in sorted(hand):
        out.append(Combo(Kind.SINGLE, (c,), (c,)))
    for g in groups:
        for pair in combinations(g, 2):
            out.append(Combo(Kind.PAIR, pair, (pair[1],)))
        for triple in combinations(g, 3):
            out.append(Combo(Kind.TRIPLE, triple, (rank_of(triple[0]),)))
    for i, ranks in enumerate(STRAIGHTS):
        if all(groups[r] for r in ranks):
            for pick in product(*(groups[r] for r in ranks)):
                cs = tuple(sorted(pick))
                flush = len({suit_of(c) for c in cs}) == 1
                out.append(Combo(Kind.STRAIGHT_FLUSH if flush else Kind.STRAIGHT, cs, (i, cs[-1])))
    for r3, g3 in enumerate(groups):
        if len(g3) < 3:
            continue
        for triple in combinations(g3, 3):
            for r2, g2 in enumerate(groups):
                if r2 != r3:
                    for pair in combinations(g2, 2):
                        out.append(Combo(Kind.FULL_HOUSE, tuple(sorted(triple + pair)), (r3,)))
        if len(g3) == 4:
            for kicker in hand:
                if rank_of(kicker) != r3:
                    out.append(Combo(Kind.FOUR, tuple(sorted([*g3, kicker])), (r3,)))
    if rules.flush_enabled:
        for suit in range(4):
            same = [c for c in sorted(hand) if suit_of(c) == suit]
            for five in combinations(same, 5):
                combo = classify(five, rules)
                if combo is not None and combo.kind == Kind.FLUSH:
                    out.append(combo)
    return out


def legal_moves(hand, last: Combo | None, rules: Rules = DEFAULT_RULES,
                must_include: int | None = None) -> list[Combo]:
    """能出的牌（不含 PASS）。last 為 None 表示自己帶頭；must_include 是第一手要含的牌（梅花 3）。"""
    moves = all_combos(hand, rules)
    if must_include is not None:
        moves = [m for m in moves if must_include in m.cards]
    if last is not None:
        moves = [m for m in moves if beats(m, last, rules)]
    return moves


def first_card_required(rules: Rules) -> int | None:
    return CLUB_THREE if rules.first_must_include_club3 else None
