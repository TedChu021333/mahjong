"""台灣 16 張麻將的對局狀態與合法動作。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from random import Random

from .tiles import NUM_TILE_TYPES, build_wall, is_flower, is_suited
from .win import is_win


class Phase(StrEnum):
    DRAW = "draw"
    DISCARD = "discard"
    RESPONSE = "response"
    ENDED = "ended"


class ActionType(StrEnum):
    DRAW = "draw"
    DISCARD = "discard"
    CHOW = "chow"
    PUNG = "pung"
    KONG = "kong"
    WIN = "win"
    PASS = "pass"


@dataclass(frozen=True)
class Meld:
    kind: str
    tiles: tuple[int, ...]
    from_player: int | None = None


@dataclass(frozen=True)
class Action:
    kind: ActionType
    tile: int | None = None
    tiles: tuple[int, ...] = ()
    from_player: int | None = None


@dataclass
class PlayerState:
    """玩家的公開對局資料；hand 是長度 34 的計數陣列。"""

    hand: list[int] = field(default_factory=lambda: [0] * NUM_TILE_TYPES)
    flowers: list[int] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)

    @property
    def open_melds(self) -> int:
        return len(self.melds)

    @property
    def hand_size(self) -> int:
        return sum(self.hand)


@dataclass
class GameState:
    wall: list[int]
    players: list[PlayerState]
    current_player: int = 0
    phase: Phase = Phase.DRAW
    last_discard: int | None = None
    discard_player: int | None = None
    response_player: int | None = None
    response_players: list[int] = field(default_factory=list)
    round_wind: int = 27
    seat_winds: tuple[int, int, int, int] = (27, 28, 29, 30)

    def __post_init__(self) -> None:
        if len(self.players) != 4:
            raise ValueError("麻將對局必須有 4 位玩家")
        if not 0 <= self.current_player < 4:
            raise ValueError("current_player 必須在 0-3 之間")


def initial_state(rng: Random | None = None, dealer: int = 0) -> GameState:
    """建立洗好的初始牌局；莊家 17 張，其餘玩家 16 張。"""
    if not 0 <= dealer < 4:
        raise ValueError("dealer 必須在 0-3 之間")
    wall = build_wall(rng)
    players = [PlayerState() for _ in range(4)]
    for player in range(4):
        target = 17 if player == dealer else 16
        while players[player].hand_size < target:
            tile = wall.pop()
            if is_flower(tile):
                players[player].flowers.append(tile)
            else:
                players[player].hand[tile] += 1
    return GameState(wall=wall, players=players, current_player=dealer,
                     phase=Phase.DISCARD)


def _validate_tile(tile: int) -> None:
    if not 0 <= tile < NUM_TILE_TYPES:
        raise ValueError("動作中的牌必須是 0-33 的非花牌")


def _has_chow(hand: list[int], tile: int) -> list[tuple[int, ...]]:
    if not is_suited(tile):
        return []
    first = tile // 9 * 9
    candidates = []
    for start in (tile - 2, tile - 1, tile):
        if start < first or start % 9 > 6 or not start <= tile <= start + 2:
            continue
        needed = (start, start + 1, start + 2)
        if all(t == tile or hand[t] > 0 for t in needed):
            candidates.append(needed)
    return candidates


def _win_action(player: PlayerState, tile: int | None = None) -> Action | None:
    if tile is not None and tile in player.discards:
        return None
    counts = player.hand.copy()
    if tile is not None:
        counts[tile] += 1
    if is_win(counts, n_open_melds=player.open_melds):
        return Action(ActionType.WIN, tile=tile)
    return None


def legal_actions(state: GameState, player: int | None = None) -> list[Action]:
    """列出指定情境下的合法動作，不改變 state。"""
    actor = state.current_player if player is None else player
    if not 0 <= actor < 4:
        raise ValueError("player 必須在 0-3 之間")
    current = state.players[actor]

    if state.phase == Phase.DRAW:
        if actor != state.current_player or not state.wall:
            return []
        return [Action(ActionType.DRAW)]

    if state.phase == Phase.DISCARD:
        if actor != state.current_player:
            return []
        actions = [Action(ActionType.DISCARD, tile=t)
                   for t, count in enumerate(current.hand) if count]
        actions.extend(
            Action(ActionType.KONG, tile=t, tiles=(t,) * 4)
            for t, count in enumerate(current.hand) if count >= 4
        )
        win = _win_action(current)
        return ([win] if win else []) + actions

    if state.phase != Phase.RESPONSE or state.last_discard is None:
        return []
    pending = state.response_players or (
        [state.response_player] if state.response_player is not None else []
    )
    if actor not in pending or state.discard_player is None:
        return []

    tile = state.last_discard
    actions = [Action(ActionType.PASS, tile=tile,
                      from_player=state.discard_player)]
    win = _win_action(current, tile)
    if win:
        actions.append(Action(ActionType.WIN, tile=tile,
                              from_player=state.discard_player))
    if current.hand[tile] >= 2:
        actions.append(Action(ActionType.PUNG, tile=tile,
                              tiles=(tile,) * 3,
                              from_player=state.discard_player))
    if current.hand[tile] >= 3:
        actions.append(Action(ActionType.KONG, tile=tile,
                              tiles=(tile,) * 4,
                              from_player=state.discard_player))
    if actor == (state.discard_player + 1) % 4:
        for chow in _has_chow(current.hand, tile):
            actions.append(Action(ActionType.CHOW, tile=tile, tiles=chow,
                                  from_player=state.discard_player))
    return actions


def add_tile(player: PlayerState, tile: int) -> None:
    """把一張牌加入玩家手牌。"""
    _validate_tile(tile)
    player.hand[tile] += 1
    if player.hand[tile] > 4:
        player.hand[tile] -= 1
        raise ValueError("同一種牌不能超過 4 張")


def remove_tile(player: PlayerState, tile: int) -> None:
    _validate_tile(tile)
    if player.hand[tile] == 0:
        raise ValueError("玩家手上沒有這張牌")
    player.hand[tile] -= 1


def _clear_response(state: GameState) -> None:
    state.last_discard = None
    state.discard_player = None
    state.response_player = None
    state.response_players.clear()


def _draw(state: GameState, player: int) -> None:
    """摸到非花牌為止；花牌移到花牌區並自動補摸。"""
    while state.wall:
        tile = state.wall.pop()
        if is_flower(tile):
            state.players[player].flowers.append(tile)
            continue
        add_tile(state.players[player], tile)
        state.phase = Phase.DISCARD
        return
    state.phase = Phase.ENDED


def apply_action(state: GameState, action: Action, player: int | None = None) -> None:
    """執行一個合法動作並更新 state；非法動作會拋出 ValueError。"""
    if state.phase == Phase.RESPONSE:
        actor = state.response_player if player is None else player
    else:
        actor = state.current_player if player is None else player
    legal = legal_actions(state, actor) if actor is not None else []
    is_pass = action.kind == ActionType.PASS and any(
        candidate.kind == ActionType.PASS for candidate in legal
    )
    if actor is None or (action not in legal and not is_pass):
        raise ValueError("不是目前情境下的合法動作")

    current = state.players[actor]
    if action.kind == ActionType.DRAW:
        _draw(state, actor)
        return
    if action.kind == ActionType.DISCARD:
        assert action.tile is not None
        remove_tile(current, action.tile)
        current.discards.append(action.tile)
        state.last_discard = action.tile
        state.discard_player = actor
        state.response_players = [(actor + offset) % 4 for offset in (1, 2, 3)]
        state.response_player = state.response_players[0]
        state.phase = Phase.RESPONSE
        return
    if action.kind == ActionType.WIN:
        if action.tile is not None:
            add_tile(current, action.tile)
        state.phase = Phase.ENDED
        return
    if action.kind == ActionType.PASS:
        state.response_players.remove(actor)
        if state.response_players:
            state.response_player = state.response_players[0]
        else:
            state.current_player = (state.discard_player + 1) % 4
            _clear_response(state)
            state.phase = Phase.DRAW
        return
    if action.kind == ActionType.KONG and state.phase == Phase.DISCARD:
        assert action.tile is not None
        for _ in range(4):
            remove_tile(current, action.tile)
        current.melds.append(Meld("kong", (action.tile,) * 4))
        state.phase = Phase.DRAW
        return

    assert state.last_discard is not None
    assert state.discard_player is not None
    claimed = state.last_discard
    source_discards = state.players[state.discard_player].discards
    if not source_discards or source_discards[-1] != claimed:
        raise ValueError("牌河最後一張與待回應棄牌不一致")
    if action.kind == ActionType.CHOW:
        claimed_tiles = list(action.tiles)
        claimed_tiles.remove(claimed)
        for tile in claimed_tiles:
            remove_tile(current, tile)
        current.melds.append(Meld("chow", action.tiles, state.discard_player))
    elif action.kind == ActionType.PUNG:
        for _ in range(2):
            remove_tile(current, claimed)
        current.melds.append(Meld("pung", (claimed,) * 3, state.discard_player))
    elif action.kind == ActionType.KONG:
        for _ in range(3):
            remove_tile(current, claimed)
        current.melds.append(Meld("kong", (claimed,) * 4, state.discard_player))
        source_discards.pop()
        state.current_player = actor
        _clear_response(state)
        state.phase = Phase.DRAW
        return
    else:
        raise ValueError("未知的動作")
    source_discards.pop()
    state.current_player = actor
    _clear_response(state)
    state.phase = Phase.DISCARD