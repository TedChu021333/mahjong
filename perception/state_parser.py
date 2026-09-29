"""把視覺層的牌面辨識結果轉成規則引擎的 GameState。

本模組不負責影像辨識；它只處理已辨識出的牌編碼，方便用截圖標註資料
測試狀態重建，也避免視覺模型直接依賴規則引擎內部實作。
"""
from __future__ import annotations

from dataclasses import dataclass

from game.rules import GameState, Meld, Phase, PlayerState
from game.tiles import NUM_TILE_TYPES, is_flower

UNKNOWN_MELD = "unknown"


@dataclass(frozen=True)
class ObservedPlayer:
    """單一玩家由畫面辨識出的資訊。"""

    hand: tuple[int, ...] = ()
    discards: tuple[int, ...] = ()
    melds: tuple[Meld, ...] = ()
    unknown_melds: int = 0
    """只知道組數、還沒辨識出內容的副露。"""


@dataclass(frozen=True)
class ObservedTable:
    players: tuple[ObservedPlayer, ...]
    current_player: int = 0
    phase: Phase = Phase.DRAW
    last_discard: int | None = None
    discard_player: int | None = None
    response_players: tuple[int, ...] = ()
    round_wind: int = 27
    seat_winds: tuple[int, int, int, int] = (27, 28, 29, 30)
    gold_tiles: tuple[int, ...] = ()
    last_drawn: int | None = None
    claimed: bool = False
    """目前玩家剛吃碰（這一手不能胡，只能打牌）。"""
    forbidden_discards: tuple[int, ...] = ()
    declared: tuple[bool, bool, bool, bool] = (False, False, False, False)
    """各家是否已宣告聽牌。"""


def _validate_tile(tile: int, allow_flower: bool = False) -> None:
    upper = 42 if allow_flower else NUM_TILE_TYPES
    if not 0 <= tile < upper:
        raise ValueError(f"無效牌編碼：{tile}")


def _validate_player(observed: ObservedPlayer) -> None:
    for tile in observed.hand:
        _validate_tile(tile)
        if is_flower(tile):
            raise ValueError("花牌必須放在 flowers，不可放入 hand")
    for tile in observed.discards:
        _validate_tile(tile)
    for meld in observed.melds:
        if meld.kind not in {"chow", "pung", "kong"}:
            raise ValueError(f"未知副露種類：{meld.kind}")
        for tile in meld.tiles:
            _validate_tile(tile)


def parse_observed_table(observed: ObservedTable) -> GameState:
    """將已辨識的桌面資訊轉成不含未知牌牆的 GameState。"""
    if len(observed.players) != 4:
        raise ValueError("需要 4 位玩家的辨識結果")
    if not 0 <= observed.current_player < 4:
        raise ValueError("current_player 必須在 0-3 之間")
    if not 0 <= observed.round_wind < 34:
        raise ValueError("round_wind 必須是有效牌編碼")
    if len(observed.seat_winds) != 4:
        raise ValueError("seat_winds 必須包含 4 個座風")
    for player in observed.response_players:
        if not 0 <= player < 4:
            raise ValueError("response_players 必須只包含 0-3")
    if observed.last_discard is not None:
        _validate_tile(observed.last_discard)
    if observed.discard_player is not None and not 0 <= observed.discard_player < 4:
        raise ValueError("discard_player 必須在 0-3 之間")

    players = []
    for player in observed.players:
        _validate_player(player)
        counts = [0] * NUM_TILE_TYPES
        for tile in player.hand:
            counts[tile] += 1
            if counts[tile] > 4:
                raise ValueError("同一玩家手牌中的牌不能超過 4 張")
        if player.unknown_melds < 0 or len(player.melds) + player.unknown_melds > 5:
            raise ValueError("副露組數必須在 0-5 之間")
        # 內容未知的副露用空牌組佔位：規則引擎只需要組數來判斷手牌張數與胡牌
        placeholders = [Meld(UNKNOWN_MELD, ()) for _ in range(player.unknown_melds)]
        players.append(PlayerState(
            hand=counts,
            flowers=[],
            melds=list(player.melds) + placeholders,
            discards=list(player.discards),
        ))

    return GameState(
        wall=[],
        players=players,
        current_player=observed.current_player,
        phase=observed.phase,
        last_discard=observed.last_discard,
        discard_player=observed.discard_player,
        response_player=observed.response_players[0] if observed.response_players else None,
        response_players=list(observed.response_players),
        round_wind=observed.round_wind,
        seat_winds=observed.seat_winds,
        gold_tiles=list(observed.gold_tiles),
        last_drawn=observed.last_drawn,
        claimed=observed.claimed,
        forbidden_discards=observed.forbidden_discards,
        declared=list(observed.declared),
    )