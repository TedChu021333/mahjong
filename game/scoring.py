"""台灣 16 張麻將的基本台數計算。

這個模組只計算能由胡牌結構與明確情境得出的牌型；搶槓、海底等
需要對局事件的台數，留給 simulator 在事件發生時加上。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .rules import Meld
from .tiles import NUM_TILE_TYPES, is_honor, is_suited
from .win import decompose, is_lickgu, is_win


TAI = {
    "門清": 1,
    "自摸": 1,
    "平胡": 2,
    "碰碰胡": 4,
    "混一色": 4,
    "清一色": 8,
    "嚦咕嚦咕": 4,
    "花牌": 1,
    "三元牌": 1,
    "門風牌": 1,
    "圈風牌": 1,
}


@dataclass(frozen=True)
class Score:
    tai: int
    patterns: tuple[str, ...]
    pair: int | None = None
    melds: tuple[tuple[str, int], ...] = ()


def _validate_counts(counts: Sequence[int]) -> list[int]:
    if len(counts) != NUM_TILE_TYPES:
        raise ValueError("手牌計數陣列必須長度為 34")
    if any(count < 0 or count > 4 for count in counts):
        raise ValueError("每種牌的數量必須在 0-4 之間")
    return list(counts)


def _meld_tiles(melds: Sequence[Meld]) -> list[int]:
    tiles = []
    for meld in melds:
        if meld.kind not in {"chow", "pung", "kong"}:
            raise ValueError(f"未知副露種類：{meld.kind}")
        tiles.extend(meld.tiles)
    return tiles


def _suit_set(counts: Sequence[int], meld_tiles: Sequence[int]) -> set[int]:
    tiles = [tile for tile, count in enumerate(counts) if count]
    tiles.extend(meld_tiles)
    return {tile // 9 for tile in tiles if is_suited(tile)}


def _has_honors(counts: Sequence[int], meld_tiles: Sequence[int]) -> bool:
    return any(is_honor(tile) for tile, count in enumerate(counts) if count) or any(
        is_honor(tile) for tile in meld_tiles
    )


def _patterns_for_decomposition(
    counts: Sequence[int],
    melds: Sequence[Meld],
    pair: int,
    concealed_melds: Sequence[tuple[str, int]],
    self_draw: bool,
    flowers: int,
    seat_wind: int | None,
    round_wind: int | None,
) -> tuple[str, ...]:
    meld_tiles = _meld_tiles(melds)
    all_melds = [kind for kind, _ in concealed_melds]
    all_melds.extend(meld.kind for meld in melds)
    suits = _suit_set(counts, meld_tiles)
    honors = _has_honors(counts, meld_tiles) or is_honor(pair)
    patterns: list[str] = []

    if not melds:
        patterns.append("門清")
    if self_draw:
        patterns.append("自摸")
    if all(kind == "pung" for kind in all_melds) and all_melds:
        patterns.append("碰碰胡")
    elif all(kind == "chow" for kind in all_melds) and not is_honor(pair):
        patterns.append("平胡")
    if len(suits) == 1:
        patterns.append("混一色" if honors else "清一色")
    honor_pungs = [
        tile for kind, tile in concealed_melds if kind == "pung" and is_honor(tile)
    ]
    honor_pungs.extend(
        meld.tiles[0]
        for meld in melds
        if meld.kind in {"pung", "kong"} and is_honor(meld.tiles[0])
    )
    for tile in honor_pungs:
        if 31 <= tile <= 33:
            patterns.append("三元牌")
        if tile == seat_wind:
            patterns.append("門風牌")
        if tile == round_wind:
            patterns.append("圈風牌")
    if flowers:
        patterns.extend(["花牌"] * flowers)
    return tuple(patterns)


def _score_patterns(patterns: Sequence[str]) -> int:
    return sum(TAI[pattern] for pattern in patterns)


def score_hand(
    counts: Sequence[int],
    open_melds: Sequence[Meld] = (),
    flowers: int = 0,
    self_draw: bool = False,
    allow_lickgu: bool = True,
    seat_wind: int | None = None,
    round_wind: int | None = None,
) -> Score:
    """計算胡牌台數；多種拆法時選台數最高者。"""
    hand = _validate_counts(counts)
    if flowers < 0 or flowers > 8:
        raise ValueError("花牌數量必須在 0-8 之間")
    if not is_win(hand, len(open_melds), allow_lickgu):
        raise ValueError("不是合法胡牌")

    if is_lickgu(hand, len(open_melds)) and allow_lickgu:
        patterns = ["嚦咕嚦咕"]
        if self_draw:
            patterns.append("自摸")
        if flowers:
            patterns.extend(["花牌"] * flowers)
        return Score(_score_patterns(patterns), tuple(patterns))

    scores = []
    for pair, concealed_melds in decompose(hand):
        patterns = _patterns_for_decomposition(
            hand, open_melds, pair, concealed_melds, self_draw, flowers
            , seat_wind, round_wind
        )
        scores.append(
            Score(_score_patterns(patterns), patterns, pair, concealed_melds)
        )
    if not scores:
        raise ValueError("胡牌拆解失敗")
    best = max(scores, key=lambda score: score.tai)
    if not best.patterns:
        return Score(best.tai, ("基本胡",), best.pair, best.melds)
    return best