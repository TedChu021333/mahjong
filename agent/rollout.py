"""蒙地卡羅用的快速模擬：從某個局面把這一局簡化地打完，回傳每家的輸贏（台）。

完整規則引擎打一局約 1 秒，3 秒限時內模擬次數太少，所以這裡只保留決定輸贏的部分：
摸牌、自摸、放槍（依座位順序先搶先胡）、向聽數會變好的碰與吃、流局。不補花、不換牌；
台數用校正表的平均值（自摸三家各付、放槍一家付，有宣告聽牌再加 1 台）。宣告聽牌的人
手牌鎖住，摸到不胡就打掉；其他人從孤張分數最低的幾張中挑打掉後向聽數最低的。這個策略
比真正的 AI 弱（從隨機起手打，約九成的局有人胡），但各候選牌用同一批模擬比較，偏差大多抵銷。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from agent.shanten import shanten
from game.tiles import NUM_TILE_TYPES
from game.win import is_win

BASE = 3


@lru_cache(maxsize=200_000)
def _wins(counts: tuple[int, ...], open_melds: int) -> bool:
    return is_win(list(counts), open_melds, True)


def keep_score(hand: Sequence[int], tile: int) -> int:
    """這張牌留著的價值：同種越多、鄰近的牌越多越高；字牌只看同種。"""
    count = hand[tile]
    score = count * 6
    if tile < 27:
        rank = tile % 9
        for offset, weight in ((1, 3), (2, 1)):
            if rank - offset >= 0:
                score += weight * min(hand[tile - offset], 2)
            if rank + offset <= 8:
                score += weight * min(hand[tile + offset], 2)
        if rank in (0, 8):
            score -= 1
    return score


SHANTEN_CANDIDATES = 4
"""只對孤張分數最低的幾張算向聽數（全部算太慢）。"""


def heuristic_discard(hand: list[int], open_melds: int = 0) -> int:
    """孤張分數最低的幾張中，打掉後向聽數最低的那張。"""
    ranked = sorted((keep_score(hand, tile), tile) for tile in range(NUM_TILE_TYPES) if hand[tile])
    best = None
    for score, tile in ranked[:SHANTEN_CANDIDATES]:
        hand[tile] -= 1
        level = _shanten(tuple(hand), open_melds)
        hand[tile] += 1
        if best is None or (level, score) < best[0]:
            best = ((level, score), tile)
    return best[1]


@lru_cache(maxsize=200_000)
def _shanten(counts: tuple[int, ...], open_melds: int) -> int:
    return shanten(list(counts), open_melds)


@dataclass
class Seat:
    hand: list[int]
    open_melds: int
    declared: bool = False


def _claim(seat: Seat, tile: int, can_chow: bool) -> bool:
    """向聽數會變好才碰（任何人）或吃（只有下家）；成功時已從手牌拿掉用到的兩張、副露加一。"""
    if seat.declared:
        return False
    before = _shanten(tuple(seat.hand), seat.open_melds)
    options = []
    if seat.hand[tile] >= 2:
        options.append((tile, tile))
    if can_chow and tile < 27:
        rank = tile % 9
        for low in (tile - 2, tile - 1, tile):
            if low // 9 == tile // 9 and 0 <= low and low % 9 <= 6 and rank - (low % 9) <= 2:
                used = [t for t in (low, low + 1, low + 2) if t != tile]
                if all(seat.hand[t] for t in used):
                    options.append(tuple(used))
    for used in options:
        for t in used:
            seat.hand[t] -= 1
        # 吃碰後手上多一張要打：以打掉最差一張後的向聽數比較
        after = min(_shanten_after_discard(seat.hand, seat.open_melds + 1), 8)
        if after < before:
            seat.open_melds += 1
            return True
        for t in used:
            seat.hand[t] += 1
    return False


def _shanten_after_discard(hand: list[int], open_melds: int) -> int:
    tile = heuristic_discard(hand, open_melds)
    hand[tile] -= 1
    level = _shanten(tuple(hand), open_melds)
    hand[tile] += 1
    return level


def play_out(seats: list[Seat], wall: list[int], discarder: int, discard: int,
             tsumo_tai: float, ron_tai: float) -> list[float]:
    """discarder 剛打出 discard；從別家能不能胡這張開始模擬。回傳四家輸贏（台）。
    tsumo_tai、ron_tai 是沒宣告聽牌時的平均台數。會改動 seats。"""
    position = len(wall)
    current, tile = discarder, discard
    while True:
        # 放槍：依打牌者的下家順序
        for step in (1, 2, 3):
            other = (current + step) % 4
            seat = seats[other]
            seat.hand[tile] += 1
            won = _wins(tuple(seat.hand), seat.open_melds)
            seat.hand[tile] -= 1
            if won:
                result = [0.0] * 4
                amount = BASE + ron_tai + seat.declared
                result[other] += amount
                result[current] -= amount
                return result
        claimer = next((other for step in (1, 2, 3)
                        if _claim(seats[(other := (current + step) % 4)], tile, step == 1)), None)
        if claimer is not None:
            current = claimer
            seat = seats[current]
        else:
            if position == 0:
                return [0.0] * 4
            current = (current + 1) % 4
            position -= 1
            drawn = wall[position]
            seat = seats[current]
            seat.hand[drawn] += 1
            if _wins(tuple(seat.hand), seat.open_melds):
                amount = BASE + tsumo_tai + seat.declared
                result = [-amount] * 4
                result[current] = 3 * amount
                return result
            if seat.declared:
                seat.hand[drawn] -= 1
                tile = drawn
                continue
        tile = heuristic_discard(seat.hand, seat.open_melds)
        seat.hand[tile] -= 1
