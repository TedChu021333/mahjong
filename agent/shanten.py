"""台灣 16 張麻將的向聽數與有效進張。"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

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


@lru_cache(maxsize=None)
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


def seven_pairs_shanten(counts: Sequence[int]) -> int:
    """回傳一般七對子的向聽數；此變體限門清 16 張手牌。"""
    validated = _validate_counts(counts, 0)
    pairs = sum(count // 2 for count in validated)
    distinct = sum(count > 0 for count in validated)
    return 7 - pairs + max(0, 7 - distinct)


def shanten(counts: Sequence[int], n_open_melds: int = 0) -> int:
    """回傳可用牌型中的最低向聽數。"""
    standard = standard_shanten(counts, n_open_melds)
    if n_open_melds:
        return standard
    return min(standard, seven_pairs_shanten(counts))


def effective_tiles(counts: Sequence[int], n_open_melds: int = 0) -> list[int]:
    """列出摸到後能降低向聽數的牌種。"""
    validated = list(_validate_counts(counts, n_open_melds))
    current = shanten(validated, n_open_melds)
    effective = []
    for tile in range(NUM_TILE_TYPES):
        if validated[tile] >= 4:
            continue
        validated[tile] += 1
        if shanten(validated, n_open_melds) < current:
            effective.append(tile)
        validated[tile] -= 1
    return effective