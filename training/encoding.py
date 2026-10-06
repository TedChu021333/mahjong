"""把牌局狀態轉成網路的輸入：CHANNELS 張 34 格的平面（34 種牌），只用該玩家看得到的資訊。

座位一律換成相對位置：0 自己、1 下家、2 對家、3 上家（與 hint.advisor 相同）。

實戰的「眼睛」不完整（10/5 放槍分析）：牌河只記到約八成、別家副露只知道組數不知道內容、
宣告聽牌偶爾認錯。Noise 讓模擬器的狀態也有同樣的缺陷，網路才不會依賴實戰拿不到的資訊。
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np

from game.rules import GameState, Phase
from game.tiles import NUM_TILE_TYPES

RECENT_DISCARDS = 3
"""每家最近打的幾張各一張平面（越新越可能和聽牌有關）。"""
MAX_WALL = 60
MAX_TURNS = 20


@dataclass(frozen=True)
class Noise:
    drop_discard: float = 0.0
    """每張牌河的牌被漏記的機率（實戰約 0.2）。"""
    hide_meld_tiles: float = 0.0
    """別家每組副露內容看不到的機率（實戰目前 1.0：只推算組數）。"""
    miss_declare: float = 0.0
    """別家已宣告卻沒認到的機率。"""


LIVE_NOISE = Noise(drop_discard=0.2, hide_meld_tiles=0.8, miss_declare=0.1)
"""訓練時預設：接近實戰辨識（副露內容偶爾知道，例如推算出的碰）。"""

_PLANES = [
    ("hand", 4),            # 自己手牌 ≥1、≥2、≥3、=4 張
    ("melds", 4),           # 各家副露裡的牌（張數 / 4）
    ("discards", 4),        # 各家牌河（張數 / 4）
    ("recent", 4 * RECENT_DISCARDS),
    ("unseen", 1),          # 自己看不到的張數 / 4
    ("gold", 1),
    ("claim_tile", 1),      # 回應階段：別人剛打出、可以吃碰胡的那張
    ("drawn", 1),           # 剛摸進的牌
    ("forbidden", 1),       # 吃碰後不能打的牌
    ("seat_wind", 1),
    ("round_wind", 1),
    ("declared", 4),        # 以下為整張填同一值的數值資訊
    ("dealer", 4),
    ("meld_count", 4),
    ("flowers", 4),
    ("discard_count", 4),
    ("phase", 2),           # 打牌 / 回應
    ("claimed", 1),         # 剛吃碰完（這手不能胡）
    ("wall", 1),
    ("streak", 1),
    ("discarder", 4),       # 回應階段：打出那張牌的是誰
]
CHANNELS = sum(size for _, size in _PLANES)
OFFSET = {}
_start = 0
for _name, _size in _PLANES:
    OFFSET[_name] = _start
    _start += _size


def relative(seat: int, player: int) -> int:
    return (seat - player) % 4


def encode(state: GameState, player: int, noise: Noise | None = None,
           rng: Random | None = None) -> np.ndarray:
    """player 視角的輸入平面，形狀 (CHANNELS, 34)，float32。"""
    planes = np.zeros((CHANNELS, NUM_TILE_TYPES), dtype=np.float32)
    noise = noise or Noise()
    rng = rng or Random(0)
    me = state.players[player]
    hand = np.asarray(me.hand, dtype=np.int64)
    for level in range(4):
        planes[OFFSET["hand"] + level] = hand >= level + 1
    visible = hand.astype(np.float32).copy()

    for seat in range(4):
        rel = relative(seat, player)
        other = state.players[seat]
        meld_plane = planes[OFFSET["melds"] + rel]
        for meld in other.melds:
            if seat != player and noise.hide_meld_tiles and rng.random() < noise.hide_meld_tiles:
                continue
            for tile in meld.tiles:
                meld_plane[tile] += 0.25
                visible[tile] += 1
        discards = [tile for tile in other.discards
                    if not (noise.drop_discard and rng.random() < noise.drop_discard)]
        river = planes[OFFSET["discards"] + rel]
        for tile in discards:
            river[tile] += 0.25
            visible[tile] += 1
        for age, tile in enumerate(reversed(discards[-RECENT_DISCARDS:])):
            planes[OFFSET["recent"] + rel * RECENT_DISCARDS + age, tile] = 1
        declared = state.declared[seat]
        if declared and seat != player and noise.miss_declare and rng.random() < noise.miss_declare:
            declared = False
        planes[OFFSET["declared"] + rel] = float(declared)
        planes[OFFSET["meld_count"] + rel] = len(other.melds) / 5
        planes[OFFSET["flowers"] + rel] = len(other.flowers) / 8
        # 實戰牌河漏記：每家打過的張數至少算 min_discards（hint 由自己打了幾手推算）
        planes[OFFSET["discard_count"] + rel] = min(
            1.0, max(len(other.discards), state.min_discards) / MAX_TURNS)

    planes[OFFSET["unseen"]] = np.clip(4 - visible, 0, 4) / 4
    for tile in state.gold_tiles:
        planes[OFFSET["gold"], tile] = 1
    responding = state.phase == Phase.RESPONSE and state.last_discard is not None
    if responding:
        planes[OFFSET["claim_tile"], state.last_discard] = 1
        if state.discard_player is not None:
            planes[OFFSET["discarder"] + relative(state.discard_player, player)] = 1
    if state.phase == Phase.DISCARD and state.current_player == player:
        if state.last_drawn is not None and not state.claimed:
            planes[OFFSET["drawn"], state.last_drawn] = 1
        for tile in state.forbidden_discards if state.claimed else ():
            planes[OFFSET["forbidden"], tile] = 1
    planes[OFFSET["seat_wind"], state.seat_winds[player]] = 1
    planes[OFFSET["round_wind"], state.round_wind] = 1
    if state.known_dealer is not None:
        planes[OFFSET["dealer"] + relative(state.known_dealer, player)] = 1
    planes[OFFSET["phase"] + (1 if responding else 0)] = 1
    planes[OFFSET["claimed"]] = float(state.claimed)
    planes[OFFSET["wall"]] = min(1.0, state.drawable / MAX_WALL)
    planes[OFFSET["streak"]] = min(1.0, state.dealer_streak / 5)
    return planes
