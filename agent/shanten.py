"""台灣 16 張麻將的向聽數與有效進張。"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from game import win as win_module
from game.tiles import NUM_TILE_TYPES, is_suited


def _validate_counts(counts: Sequence[int], n_open_melds: int) -> tuple[int, ...]:
    if len(counts) != NUM_TILE_TYPES:
        raise ValueError("手牌計數陣列必須長度為 34")
    if not 0 <= n_open_melds <= 5:
        raise ValueError("副露數必須在 0-5 之間")
    if any(count < 0 or count > 4 for count in counts):
        raise ValueError("每種牌的數量必須在 0-4 之間")
    expected = 16 - 3 * n_open_melds
    if sum(counts) not in (expected, expected + 1):
        raise ValueError(f"向聽數計算需要 {expected} 或 {expected + 1} 張暗手牌")
    return tuple(counts)


@lru_cache(maxsize=500_000)  # 不設上限時長時間模擬會用光記憶體
def _standard_shanten(
    counts: tuple[int, ...],
    index: int,
    melds: int,
    taatsu: int,
    pair: int,
) -> int:
    while index < NUM_TILE_TYPES and counts[index] == 0:
        index += 1
    if index == NUM_TILE_TYPES:
        usable_taatsu = min(taatsu, 5 - melds)
        return 8 - 2 * melds - usable_taatsu - pair

    isolated = list(counts)
    isolated[index] -= 1
    best = _standard_shanten(tuple(isolated), index, melds, taatsu, pair)
    mutable = list(counts)

    if counts[index] >= 3 and melds < 5:
        mutable[index] -= 3
        best = min(best, _standard_shanten(tuple(mutable), index, melds + 1, taatsu, pair))
        mutable[index] += 3

    if is_suited(index) and index % 9 <= 6 and counts[index + 1] and counts[index + 2]:
        mutable[index] -= 1
        mutable[index + 1] -= 1
        mutable[index + 2] -= 1
        best = min(best, _standard_shanten(tuple(mutable), index, melds + 1, taatsu, pair))

    if pair == 0 and counts[index] >= 2:
        mutable = list(counts)
        mutable[index] -= 2
        best = min(best, _standard_shanten(tuple(mutable), index, melds, taatsu, 1))

    if taatsu < 5:
        if counts[index] >= 2:
            mutable = list(counts)
            mutable[index] -= 2
            best = min(best, _standard_shanten(tuple(mutable), index, melds, taatsu + 1, pair))
        if is_suited(index) and index % 9 <= 7 and counts[index + 1]:
            mutable = list(counts)
            mutable[index] -= 1
            mutable[index + 1] -= 1
            best = min(best, _standard_shanten(tuple(mutable), index, melds, taatsu + 1, pair))
        if is_suited(index) and index % 9 <= 6 and counts[index + 2]:
            mutable = list(counts)
            mutable[index] -= 1
            mutable[index + 2] -= 1
            best = min(best, _standard_shanten(tuple(mutable), index, melds, taatsu + 1, pair))
    return best


def standard_shanten(counts: Sequence[int], n_open_melds: int = 0) -> int:
    """回傳標準五面子一對眼的向聽數；胡牌為 -1。

    台灣 16 張麻將的摸牌前手牌有 16-3*n 張，補牌後則多一張。
    """
    validated = _validate_counts(counts, n_open_melds)
    result = _standard_shanten(validated, 0, n_open_melds, 0, 0)
    return result + 2


def _seven_pairs_shanten_validated(counts: tuple[int, ...]) -> int:
    # 嚦咕嚦咕 = 1 刻 + 7 對，共 8 種牌 17 張；逐一假設哪種牌當刻子，
    # 其餘取張數最多的 7 種當對子，缺幾張就差幾步（胡牌 = -1）。
    pair_parts = sorted((min(count, 2) for count in counts), reverse=True)
    best = 0
    for count in counts:
        pairs = pair_parts.copy()
        pairs.remove(min(count, 2))
        best = max(best, min(count, 3) + sum(pairs[:7]))
    return 17 - best - 1


def seven_pairs_shanten(counts: Sequence[int]) -> int:
    """回傳嚦咕嚦咕（7 對 + 1 刻）的向聽數；此牌型限門清。"""
    validated = _validate_counts(counts, 0)
    return _seven_pairs_shanten_validated(validated)


@lru_cache(maxsize=200_000)  # 每筆約 0.5KB；100 萬筆時 15 個 arena 子程序用光記憶體
def _shanten_cached(counts: tuple[int, ...], n_open_melds: int) -> int:
    standard = standard_shanten(counts, n_open_melds)
    if n_open_melds or not win_module.LICKGU_ENABLED:
        return standard
    return min(standard, _seven_pairs_shanten_validated(counts))


def shanten(counts: Sequence[int], n_open_melds: int = 0) -> int:
    """回傳可用牌型中的最低向聽數。"""
    validated = _validate_counts(counts, n_open_melds)
    return _shanten_cached(validated, n_open_melds)


@lru_cache(maxsize=200_000)
def _effective_tiles_cached(counts: tuple[int, ...], n_open_melds: int) -> tuple[int, ...]:
    current = _shanten_cached(counts, n_open_melds)
    mutable = list(counts)
    effective = []
    for tile in range(NUM_TILE_TYPES):
        if mutable[tile] >= 4:
            continue
        mutable[tile] += 1
        if _shanten_cached(tuple(mutable), n_open_melds) < current:
            effective.append(tile)
        mutable[tile] -= 1
    return tuple(effective)


def effective_tiles(counts: Sequence[int], n_open_melds: int = 0) -> list[int]:
    """列出摸到後能降低向聽數的牌種。"""
    validated = _validate_counts(counts, n_open_melds)
    return list(_effective_tiles_cached(validated, n_open_melds))

def improvement_count(counts: Sequence[int], n_open_melds: int, visible: Sequence[int]) -> int:
    """改良牌：摸到後（再打掉一張）向聽數不變、但有效進張變多的牌，回傳看不到的張數。
    counts 是打完牌、等著摸牌的手牌；有效進張以 4 − visible 計。進張數相同時，
    改良牌多的手以後比較容易變好（例如留孤張六萬比留孤張字牌好）。"""
    hand = list(counts)
    level = shanten(hand, n_open_melds)
    effective = set(effective_tiles(hand, n_open_melds))
    base = sum(max(0, 4 - visible[tile]) for tile in effective)
    total = 0
    for drawn in range(len(hand)):
        live = 4 - visible[drawn]
        if live <= 0 or drawn in effective or hand[drawn] >= 4:
            continue
        hand[drawn] += 1
        better = False
        for discard in range(len(hand)):
            if not hand[discard] or discard == drawn:
                continue
            hand[discard] -= 1
            if shanten(hand, n_open_melds) == level:
                outs = sum(max(0, 4 - visible[tile] - (tile == drawn))
                           for tile in effective_tiles(hand, n_open_melds))
                better = outs > base
            hand[discard] += 1
            if better:
                break
        hand[drawn] -= 1
        if better:
            total += live
    return total
