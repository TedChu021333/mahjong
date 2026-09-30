"""台灣 16 張麻將的對局狀態與合法動作。

台灣規則重點：
- 沒有振聽：打過的牌別人打出來仍然可以胡。
- 留牌：牌牆剩 DEAD_WALL_SIZE 張時不再摸牌，流局。
- 吃碰後只能打牌，不能直接胡；槓牌後補摸一張。
- 加槓（碰出的刻子補第 4 張）時，其他家可以搶槓胡。
- 花牌自動補花；湊滿 8 張為八仙過海，別家摸到第 8 張時持有 7 張者七搶一。

明星三缺一的額外規則：
- 金牌：每局隨機 GOLD_TILES_AT_START 種牌，每次槓成立再加 1 種。
- 聽：打牌時可同時宣告聽牌（打完必須是聽牌），之後手牌鎖住，只能打掉
  摸進的牌或胡，也不能吃碰槓。
- 換三張（可選）：開局每人可選 0～MAX_SWAP 張與牌牆交換。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from random import Random

from .tiles import NUM_FLOWERS, NUM_TILE_TYPES, build_wall, is_flower, is_suited
from .win import is_win, winning_tiles

DEAD_WALL_SIZE = 16
"""台灣麻將留 8 墩（16 張）不摸。"""
GOLD_TILES_AT_START = 2
MAX_SWAP = 3


class Phase(StrEnum):
    SWAP = "swap"
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
    DECLARE = "declare"
    """打出 tile 並宣告聽牌。"""
    SWAP = "swap"
    """換三張：tiles 為要換掉的牌（可為空，代表不換）。"""


@dataclass(frozen=True)
class Meld:
    kind: str
    tiles: tuple[int, ...]
    from_player: int | None = None
    """None 代表暗槓。"""


@dataclass(frozen=True)
class Action:
    kind: ActionType
    tile: int | None = None
    tiles: tuple[int, ...] = ()
    from_player: int | None = None


@dataclass(frozen=True)
class FlowerWin:
    winner: int
    kind: str
    """"八仙過海" 或 "七搶一"。"""
    from_player: int | None = None
    """七搶一時被搶走第 8 張花的玩家。"""


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
    dealer: int = 0
    dealer_streak: int = 0
    """莊家連莊次數（連 n 拉 n：莊家有參與的胡牌加 2n 台）。"""
    reserve: int = 0
    """牌牆保留不摸的張數；initial_state 設為 DEAD_WALL_SIZE。"""
    last_drawn: int | None = None
    """目前玩家最後摸進的牌，自摸時就是胡的那張。"""
    claimed: bool = False
    """剛吃碰，這一手只能打牌。"""
    forbidden_discards: tuple[int, ...] = ()
    """剛吃碰後這一手不能打的牌（見 forbidden_after_claim）。"""
    after_kong: bool = False
    """剛槓完補牌，此時自摸為槓上開花。"""
    robbing_kong: bool = False
    """RESPONSE 是否為加槓的搶槓回應（只能胡或過）。"""
    flower_win: FlowerWin | None = None
    gold_tiles: list[int] = field(default_factory=list)
    """目前的金牌牌種。"""
    gold_pool: list[int] = field(default_factory=list)
    """槓牌時依序加入的金牌候選；開局就洗好，讓同一 seed 的對局可重現。"""
    declared: list[bool] = field(default_factory=lambda: [False] * 4)
    """各家是否已宣告聽牌。"""
    early_declared: list[bool] = field(default_factory=lambda: [False] * 4)
    """是否在第一次打牌時就宣告聽牌（地聽）。"""
    swap_players: list[int] = field(default_factory=list)
    """換三張階段還沒決定的玩家（依序）。"""

    def __post_init__(self) -> None:
        if len(self.players) != 4:
            raise ValueError("麻將對局必須有 4 位玩家")
        if not 0 <= self.current_player < 4:
            raise ValueError("current_player 必須在 0-3 之間")
        if not 0 <= self.dealer < 4:
            raise ValueError("dealer 必須在 0-3 之間")

    @property
    def drawable(self) -> int:
        """還能摸的張數。"""
        return max(0, len(self.wall) - self.reserve)


def seat_winds_for_dealer(dealer: int) -> tuple[int, int, int, int]:
    """莊家為東，逆時針（座位順序）依序南西北。"""
    return tuple(27 + (player - dealer) % 4 for player in range(4))  # type: ignore[return-value]


def _collect_flower(state: GameState, player: int, tile: int) -> None:
    state.players[player].flowers.append(tile)
    counts = [len(p.flowers) for p in state.players]
    if counts[player] == NUM_FLOWERS:
        state.flower_win = FlowerWin(player, "八仙過海")
    elif sum(counts) == NUM_FLOWERS:
        holder = next((p for p, count in enumerate(counts) if count == NUM_FLOWERS - 1), None)
        if holder is not None:
            state.flower_win = FlowerWin(holder, "七搶一", from_player=player)
    if state.flower_win is not None:
        state.phase = Phase.ENDED


def initial_state(rng: Random | None = None, dealer: int = 0,
                  round_wind: int = 27, swap: bool = False,
                  dealer_streak: int = 0) -> GameState:
    """建立洗好的初始牌局；莊家 17 張，其餘玩家 16 張，花牌自動補。

    swap=True 時先進入換三張階段，從莊家開始每人決定一次。
    """
    if not 0 <= dealer < 4:
        raise ValueError("dealer 必須在 0-3 之間")
    rng = rng or Random()
    gold_order = list(range(NUM_TILE_TYPES))
    rng.shuffle(gold_order)
    state = GameState(
        wall=build_wall(rng), players=[PlayerState() for _ in range(4)],
        current_player=dealer, phase=Phase.DISCARD, round_wind=round_wind,
        seat_winds=seat_winds_for_dealer(dealer), dealer=dealer, dealer_streak=dealer_streak,
        reserve=DEAD_WALL_SIZE,
        gold_tiles=gold_order[:GOLD_TILES_AT_START],
        gold_pool=gold_order[GOLD_TILES_AT_START:],
    )
    for offset in range(4):
        player = (dealer + offset) % 4
        target = 17 if player == dealer else 16
        while state.players[player].hand_size < target:
            tile = state.wall.pop()
            if is_flower(tile):
                _collect_flower(state, player, tile)
                if state.flower_win is not None:
                    return state
            else:
                state.players[player].hand[tile] += 1
    if swap:
        state.phase = Phase.SWAP
        state.swap_players = [(dealer + offset) % 4 for offset in range(4)]
    return state


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
    counts = player.hand.copy()
    if tile is not None:
        counts[tile] += 1
    if is_win(counts, n_open_melds=player.open_melds):
        return Action(ActionType.WIN, tile=tile)
    return None


def _declarable_discards(player: PlayerState) -> tuple[int, ...]:
    """打出後會聽牌的牌。"""
    return _declarable_discards_cached(tuple(player.hand), player.open_melds)


@lru_cache(maxsize=200_000)
def _declarable_discards_cached(hand: tuple[int, ...], n_open: int) -> tuple[int, ...]:
    # 向聽數只依賴 game.tiles，沒有循環引用；先用它快速排除，
    # 只有向聽 0 的候選才做精確的聽牌檢查（排除聽的牌已被自己握滿 4 張）。
    from agent.shanten import shanten

    if sum(hand) != 17 - 3 * n_open or shanten(hand, n_open) > 0:
        return ()
    counts = list(hand)
    options = []
    for tile, count in enumerate(hand):
        if not count:
            continue
        counts[tile] -= 1
        if shanten(counts, n_open) == 0 and winning_tiles(counts, n_open):
            options.append(tile)
        counts[tile] += 1
    return tuple(options)


def _add_gold_tile(state: GameState) -> None:
    """槓成立時多一張金牌。"""
    if state.gold_pool:
        state.gold_tiles.append(state.gold_pool.pop(0))


def _added_kong_tiles(player: PlayerState) -> list[int]:
    return [meld.tiles[0] for meld in player.melds
            if meld.kind == "pung" and player.hand[meld.tiles[0]]]


def forbidden_after_claim(action: Action, claimed: int) -> tuple[int, ...]:
    """吃碰後不能馬上打的牌：碰的那張；吃的那張，以及吃在順子邊張時另一頭的牌
    （例如用 67 吃 5，不能打 5 和 8；用 46 吃 5 只不能打 5）。"""
    if action.kind == ActionType.PUNG:
        return (claimed,)
    if action.kind != ActionType.CHOW:
        return ()
    low = min(action.tiles)
    rank = low % 9
    if claimed == low and rank <= 5:
        return (claimed, low + 3)
    if claimed == low + 2 and rank >= 1:
        return (claimed, low - 1)
    return (claimed,)


def allowed_discards(hand: list[int], forbidden: tuple[int, ...] = ()) -> list[int]:
    """可以打的牌種；手上只剩禁打的牌時全部都能打。"""
    tiles = [tile for tile, count in enumerate(hand) if count]
    allowed = [tile for tile in tiles if tile not in forbidden]
    return allowed or tiles


def legal_actions(state: GameState, player: int | None = None) -> list[Action]:
    """列出指定情境下的合法動作，不改變 state。"""
    actor = state.current_player if player is None else player
    if not 0 <= actor < 4:
        raise ValueError("player 必須在 0-3 之間")
    current = state.players[actor]

    if state.phase == Phase.SWAP:
        # 可換的組合太多，只列出「不換」；apply_action 另外驗證要換的牌
        return [Action(ActionType.SWAP)] if actor == state.current_player else []

    if state.phase == Phase.DRAW:
        if actor != state.current_player or not state.drawable:
            return []
        return [Action(ActionType.DRAW)]

    if state.phase == Phase.DISCARD:
        if actor != state.current_player:
            return []
        win = _win_action(current) if not state.claimed else None
        if state.declared[actor]:
            # 聽牌後手牌鎖住：只能胡或打掉剛摸進的牌
            assert state.last_drawn is not None
            return ([win] if win else []) + [Action(ActionType.DISCARD, tile=state.last_drawn)]
        allowed = allowed_discards(current.hand, state.forbidden_discards if state.claimed else ())
        actions = [Action(ActionType.DISCARD, tile=t) for t in allowed]
        actions.extend(Action(ActionType.DECLARE, tile=t)
                       for t in _declarable_discards(current) if t in allowed)
        if state.claimed:
            return actions
        if state.drawable:  # 槓完要補牌
            actions.extend(
                Action(ActionType.KONG, tile=t, tiles=(t,) * 4)
                for t, count in enumerate(current.hand) if count >= 4
            )
            actions.extend(Action(ActionType.KONG, tile=t, tiles=(t,) * 4)
                           for t in _added_kong_tiles(current))
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
    if state.robbing_kong or state.declared[actor]:
        return actions
    if current.hand[tile] >= 2:
        actions.append(Action(ActionType.PUNG, tile=tile,
                              tiles=(tile,) * 3,
                              from_player=state.discard_player))
    # 明星三缺一：上家打的牌不能明槓（遊戲顯示「上家出牌，不可明槓」），只能碰
    from_upper = actor == (state.discard_player + 1) % 4
    if current.hand[tile] >= 3 and state.drawable and not from_upper:
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
    state.robbing_kong = False


def _draw(state: GameState, player: int) -> None:
    """摸到非花牌為止；花牌移到花牌區並自動補摸。摸到留牌就流局。"""
    while state.drawable:
        tile = state.wall.pop()
        if is_flower(tile):
            _collect_flower(state, player, tile)
            if state.flower_win is not None:
                return
            continue
        add_tile(state.players[player], tile)
        state.last_drawn = tile
        state.phase = Phase.DISCARD
        return
    state.phase = Phase.ENDED


def _swap(state: GameState, player: int, tiles: tuple[int, ...]) -> None:
    """換三張：先摸新牌（花牌照常補花），換掉的牌再放回牌牆可摸的位置。"""
    current = state.players[player]
    if len(tiles) > MAX_SWAP:
        raise ValueError(f"最多換 {MAX_SWAP} 張")
    for tile in tiles:
        remove_tile(current, tile)
    received = 0
    while received < len(tiles):
        if not state.drawable:
            raise ValueError("牌牆不足以換牌")
        tile = state.wall.pop()
        if is_flower(tile):
            _collect_flower(state, player, tile)
            if state.flower_win is not None:
                return
            continue
        add_tile(current, tile)
        received += 1
    # 放回的位置由牌局內容決定，同一 seed 的對局可重現；留牌區不放
    placer = Random(hash((len(state.wall), player, tiles)))
    for tile in tiles:
        state.wall.insert(placer.randint(state.reserve, len(state.wall)), tile)
    state.swap_players.remove(player)
    if state.swap_players:
        state.current_player = state.swap_players[0]
    else:
        state.current_player = state.dealer
        state.phase = Phase.DISCARD


def _upgrade_to_kong(player: PlayerState, tile: int) -> None:
    index = next(i for i, meld in enumerate(player.melds)
                 if meld.kind == "pung" and meld.tiles[0] == tile)
    player.melds[index] = Meld("kong", (tile,) * 4, player.melds[index].from_player)


def _downgrade_to_pung(player: PlayerState, tile: int) -> None:
    index = next(i for i, meld in enumerate(player.melds)
                 if meld.kind == "kong" and meld.tiles[0] == tile)
    player.melds[index] = Meld("pung", (tile,) * 3, player.melds[index].from_player)


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
    is_swap = action.kind == ActionType.SWAP and bool(legal)
    if actor is None or (action not in legal and not is_pass and not is_swap):
        raise ValueError("不是目前情境下的合法動作")

    current = state.players[actor]
    if action.kind == ActionType.SWAP:
        _swap(state, actor, action.tiles)
        return
    if action.kind == ActionType.DRAW:
        _draw(state, actor)
        return
    if action.kind in {ActionType.DISCARD, ActionType.DECLARE}:
        assert action.tile is not None
        if action.kind == ActionType.DECLARE:
            state.declared[actor] = True
            state.early_declared[actor] = not current.discards
        remove_tile(current, action.tile)
        current.discards.append(action.tile)
        state.claimed = False
        state.forbidden_discards = ()
        state.after_kong = False
        state.last_discard = action.tile
        state.discard_player = actor
        state.response_players = [(actor + offset) % 4 for offset in (1, 2, 3)]
        state.response_player = state.response_players[0]
        state.phase = Phase.RESPONSE
        return
    if action.kind == ActionType.WIN:
        if action.tile is not None:
            if state.robbing_kong:
                assert state.discard_player is not None
                _downgrade_to_pung(state.players[state.discard_player], action.tile)
            add_tile(current, action.tile)
        state.phase = Phase.ENDED
        return
    if action.kind == ActionType.PASS:
        state.response_players.remove(actor)
        if state.response_players:
            state.response_player = state.response_players[0]
            return
        assert state.discard_player is not None
        if state.robbing_kong:
            # 沒人搶槓：加槓成立，由槓的人補牌
            state.current_player = state.discard_player
            state.after_kong = True
            _add_gold_tile(state)
        else:
            state.current_player = (state.discard_player + 1) % 4
        _clear_response(state)
        state.phase = Phase.DRAW
        return
    if action.kind == ActionType.KONG and state.phase == Phase.DISCARD:
        assert action.tile is not None
        if current.hand[action.tile] == 4:
            for _ in range(4):
                remove_tile(current, action.tile)
            current.melds.append(Meld("kong", (action.tile,) * 4))
            state.after_kong = True
            _add_gold_tile(state)
            state.phase = Phase.DRAW
            return
        # 加槓：先讓其他家決定是否搶槓
        remove_tile(current, action.tile)
        _upgrade_to_kong(current, action.tile)
        state.last_discard = action.tile
        state.discard_player = actor
        state.response_players = [(actor + offset) % 4 for offset in (1, 2, 3)]
        state.response_player = state.response_players[0]
        state.robbing_kong = True
        state.phase = Phase.RESPONSE
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
        state.after_kong = True
        _add_gold_tile(state)
        state.phase = Phase.DRAW
        return
    else:
        raise ValueError("未知的動作")
    source_discards.pop()
    state.current_player = actor
    _clear_response(state)
    state.claimed = True
    state.forbidden_discards = forbidden_after_claim(action, claimed)
    state.phase = Phase.DISCARD
