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


def _better(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    return a != b and a[0] >= b[0] and a[1] >= b[1] and a[2] >= b[2]


@lru_cache(maxsize=200_000)
def _group_frontier(counts: tuple[int, ...], suited: bool) -> tuple[tuple[int, int, int], ...]:
    """一組牌（一種花色 9 種，或 7 種字牌）能拆出的（面子, 搭子, 眼）組合中不被其他組合
    全面勝過的那些。拆法與 _standard_shanten 相同，只是限定在同一組內；各組互不影響，
    所以整手的最佳值可以由各組的結果組合出來，而同一組牌在不同手牌間大量重複、可快取。"""
    found: set[tuple[int, int, int]] = set()

    def search(c: list[int], index: int, melds: int, taatsu: int, pair: int) -> None:
        while index < len(c) and c[index] == 0:
            index += 1
        if index == len(c):
            found.add((melds, taatsu, pair))
            return
        c[index] -= 1
        search(c, index, melds, taatsu, pair)  # 孤張
        c[index] += 1
        if c[index] >= 3:
            c[index] -= 3
            search(c, index, melds + 1, taatsu, pair)
            c[index] += 3
        sequence = suited and index <= 6
        if sequence and c[index + 1] and c[index + 2]:
            for k in range(3):
                c[index + k] -= 1
            search(c, index, melds + 1, taatsu, pair)
            for k in range(3):
                c[index + k] += 1
        if c[index] >= 2:
            c[index] -= 2
            if pair == 0:
                search(c, index, melds, taatsu, 1)
            search(c, index, melds, taatsu + 1, pair)
            c[index] += 2
        for gap in (1, 2):
            if suited and index + gap <= 8 and c[index + gap]:
                c[index] -= 1
                c[index + gap] -= 1
                search(c, index, melds, taatsu + 1, pair)
                c[index] += 1
                c[index + gap] += 1

    search(list(counts), 0, 0, 0, 0)
    return tuple(sorted(a for a in found if not any(_better(b, a) for b in found)))


def standard_shanten(counts: Sequence[int], n_open_melds: int = 0) -> int:
    """回傳標準五面子一對眼的向聽數；胡牌為 -1。

    台灣 16 張麻將的摸牌前手牌有 16-3*n 張，補牌後則多一張。
    向聽數 = 10 − 2×面子 − 可用搭子（最多補滿 5 組）− 眼；萬、筒、條、字四組分開拆
    （_group_frontier），再找最好的組合。結果與逐張遞迴的 _standard_shanten 相同
    （測試以隨機手牌比對），但快約 50 倍：模擬對局 95% 的時間原本都花在這裡。
    """
    validated = _validate_counts(counts, n_open_melds)
    groups = [_group_frontier(validated[start:start + 9], True) for start in (0, 9, 18)]
    groups.append(_group_frontier(validated[27:34], False))
    best = 0
    for m0, t0, p0 in groups[0]:
        for m1, t1, p1 in groups[1]:
            for m2, t2, p2 in groups[2]:
                m = n_open_melds + m0 + m1 + m2
                t = t0 + t1 + t2
                p = p0 + p1 + p2
                for m3, t3, p3 in groups[3]:
                    melds = min(5, m + m3)
                    value = 2 * melds + min(t + t3, 5 - melds) + min(1, p + p3)
                    if value > best:
                        best = value
    return 10 - best


def _reference_standard_shanten(counts: Sequence[int], n_open_melds: int = 0) -> int:
    """原本逐張遞迴的寫法，只留給測試比對。"""
    validated = _validate_counts(counts, n_open_melds)
    return _standard_shanten(validated, 0, n_open_melds, 0, 0) + 2


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


def next_step_outs(counts: Sequence[int], n_open_melds: int, visible: Sequence[int]) -> int:
    """牌型分數（往後多看一步）：每張有效進張摸到後、再打最好的一張，剩下的有效進張有幾張，
    依那張進張看不到的張數加權加總。進張、改良牌都一樣時用來分出牌型好壞：
    10/20 三四六七九筒打六筒（留三四、七九嵌張）和打九筒（留三四、六七兩面）進張 16、改良牌 56 都一樣，
    但摸到二五筒補完三四之後，前者只剩八筒、後者剩五八筒。"""
    hand = list(counts)
    total = 0
    for drawn in effective_tiles(hand, n_open_melds):
        live = 4 - visible[drawn]
        if live <= 0:
            continue
        hand[drawn] += 1
        best = None
        for discard in range(len(hand)):
            if not hand[discard]:
                continue
            hand[discard] -= 1
            level = shanten(hand, n_open_melds)
            outs = sum(max(0, 4 - visible[tile] - (tile == drawn))
                       for tile in effective_tiles(hand, n_open_melds))
            hand[discard] += 1
            if best is None or (level, -outs) < best:
                best = (level, -outs)
        hand[drawn] -= 1
        total += live * -best[1]
    return total
