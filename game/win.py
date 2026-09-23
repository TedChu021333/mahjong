"""台灣16張麻將：胡牌判定

一般胡牌 = 5 組面子（順子/刻子）+ 1 對眼，共 17 張。
已經吃碰槓出去的面子數為 n_open，則暗手需有 17 - 3*n_open 張。
（槓子在手牌結構上視為一組面子，補牌另外處理。）
"""
from __future__ import annotations

from functools import lru_cache

from .tiles import NUM_TILE_TYPES, is_suited

HAND_SIZE = 16
NUM_MELDS = 5

Meld = tuple[str, int]  # ("pung", 起始牌) 或 ("chow", 起始牌)


@lru_cache(maxsize=None)
def _meld_decompositions(counts: tuple[int, ...]) -> tuple[tuple[Meld, ...], ...]:
    """把計數陣列完全拆成面子的所有方法；拆不完則回傳空 tuple"""
    i = next((k for k, c in enumerate(counts) if c), None)
    if i is None:
        return ((),)

    results: list[tuple[Meld, ...]] = []
    c = list(counts)

    # 最小的那張牌一定屬於「以它開頭的刻子」或「以它開頭的順子」
    if c[i] >= 3:
        c[i] -= 3
        for rest in _meld_decompositions(tuple(c)):
            results.append((("pung", i),) + rest)
        c[i] += 3

    if is_suited(i) and i % 9 <= 6 and c[i + 1] and c[i + 2]:
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        for rest in _meld_decompositions(tuple(c)):
            results.append((("chow", i),) + rest)

    return tuple(results)


def decompose(counts: list[int]) -> list[tuple[int, tuple[Meld, ...]]]:
    """列出所有「眼 + 面子」拆法，之後算台數（平胡、碰碰胡…）會用到"""
    if sum(counts) % 3 != 2:
        return []
    out = []
    c = list(counts)
    for pair in range(NUM_TILE_TYPES):
        if c[pair] >= 2:
            c[pair] -= 2
            for melds in _meld_decompositions(tuple(c)):
                out.append((pair, melds))
            c[pair] += 2
    return out


def _expected_size(n_open_melds: int) -> int:
    if not 0 <= n_open_melds <= NUM_MELDS:
        raise ValueError("副露數必須在 0-5 之間")
    return HAND_SIZE + 1 - 3 * n_open_melds


def is_standard_win(counts: list[int], n_open_melds: int = 0) -> bool:
    if sum(counts) != _expected_size(n_open_melds):
        return False
    c = list(counts)
    for pair in range(NUM_TILE_TYPES):
        if c[pair] >= 2:
            c[pair] -= 2
            ok = bool(_meld_decompositions(tuple(c)))
            c[pair] += 2
            if ok:
                return True
    return False


def is_lickgu(counts: list[int], n_open_melds: int = 0) -> bool:
    """嚦咕嚦咕：門清，7 對 + 1 刻（地方規則，可關閉）"""
    if n_open_melds != 0 or sum(counts) != 17:
        return False
    triplets = [c for c in counts if c == 3]
    if len(triplets) != 1:
        return False
    if any(c % 2 for c in counts if c != 3):
        return False
    return sum(c // 2 for c in counts if c != 3) == 7


def is_win(counts: list[int], n_open_melds: int = 0, allow_lickgu: bool = True) -> bool:
    if is_standard_win(counts, n_open_melds):
        return True
    return allow_lickgu and is_lickgu(counts, n_open_melds)


def winning_tiles(counts: list[int], n_open_melds: int = 0,
                  allow_lickgu: bool = True) -> list[int]:
    """聽哪些牌（手上已有 4 張的牌不算）"""
    if sum(counts) != _expected_size(n_open_melds) - 1:
        raise ValueError("聽牌判定時手牌張數不對")
    waits = []
    c = list(counts)
    for t in range(NUM_TILE_TYPES):
        if c[t] >= 4:
            continue
        c[t] += 1
        if is_win(c, n_open_melds, allow_lickgu):
            waits.append(t)
        c[t] -= 1
    return waits


def flower_win(my_flowers: int, other_holds_last_flower: bool = False) -> str | None:
    """花牌胡：八仙過海（集滿 8 張）、七搶一（自己 7 張，搶別人剛補的第 8 張）"""
    if my_flowers == 8:
        return "八仙過海"
    if my_flowers == 7 and other_holds_last_flower:
        return "七搶一"
    return None
