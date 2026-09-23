"""以向聽數與有效進張為核心的簡單規則式 AI。"""
from __future__ import annotations

from random import Random
from typing import Sequence

from .shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions


def choose_discard(hand: list[int], n_open_melds: int = 0,
                   rng: Random | None = None,
                   visible_counts: Sequence[int] | None = None,
                   safe_tiles: Sequence[int] = (),
                   suji_tiles: Sequence[int] = (),
                   defensive: bool = False) -> int:
    """選一張牌打出：最低向聽優先，再取有效進張最多。"""
    if sum(hand) not in (17 - 3 * n_open_melds, 16 - 3 * n_open_melds):
        raise ValueError("選擇打牌時手牌張數不正確")
    if visible_counts is not None and (
        len(visible_counts) != 34 or any(count < 0 or count > 4 for count in visible_counts)
    ):
        raise ValueError("可見牌計數必須是長度 34 且每格介於 0-4")
    candidates = []
    for tile, count in enumerate(hand):
        if not count:
            continue
        after_discard = hand.copy()
        after_discard[tile] -= 1
        next_shanten = shanten(after_discard, n_open_melds)
        candidates.append((next_shanten, tile, after_discard))
    if not candidates:
        raise ValueError("沒有可打出的牌")
    minimum_shanten = min(candidate[0] for candidate in candidates)
    safe_set = set(safe_tiles)
    if any(tile < 0 or tile >= 34 for tile in safe_set):
        raise ValueError("安全牌編碼必須在 0-33 之間")
    suji_set = set(suji_tiles)
    if any(tile < 0 or tile >= 34 for tile in suji_set):
        raise ValueError("筋牌編碼必須在 0-33 之間")
    best = []
    for next_shanten, tile, after_discard in candidates:
        if next_shanten != minimum_shanten:
            continue
        effective = effective_tiles(after_discard, n_open_melds)
        if visible_counts is None:
            outs = len(effective)
        else:
            outs = sum(max(0, 4 - visible_counts[tile]) for tile in effective)
        risk = 0 if tile in safe_set else 1 if tile in suji_set else 2
        if defensive:
            best.append((risk, -outs, next_shanten, tile))
        else:
            best.append((-outs, risk, next_shanten, tile))
    best_score = min(score[:3] for score in best)
    best_tiles = [tile for *score, tile in best if tuple(score) == best_score]
    return (rng or Random()).choice(best_tiles)


def visible_tile_counts(state: GameState, player: int) -> list[int]:
    """以個人視角統計自己的手牌、牌河與公開副露。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    visible = [0] * 34
    own = state.players[player]
    for tile, count in enumerate(own.hand):
        visible[tile] += count
    for known_player in state.players:
        for tile in known_player.discards:
            visible[tile] += 1
        for meld in known_player.melds:
            for tile in meld.tiles:
                visible[tile] += 1
    if any(count > 4 for count in visible):
        raise ValueError("牌局狀態出現超過 4 張同種牌")
    return visible


def safe_tiles(state: GameState, player: int) -> set[int]:
    """回傳對可觀測威脅對手都已打過的現物牌。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    opponents = [
        opponent_state for opponent, opponent_state in enumerate(state.players)
        if opponent != player and len(opponent_state.melds) >= 2
    ]
    if not opponents:
        opponents = [
            opponent_state for opponent, opponent_state in enumerate(state.players)
            if opponent != player
        ]
    safe: set[int] | None = None
    for opponent_state in opponents:
        discards = set(opponent_state.discards)
        safe = discards if safe is None else safe & discards
    return safe or set()


def suji_tiles(state: GameState, player: int) -> set[int]:
    """回傳筋牌候選；這不是絕對安全牌，僅降低兩面聽風險。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    discarded = set()
    for opponent, opponent_state in enumerate(state.players):
        if opponent != player:
            discarded.update(opponent_state.discards)
    suji = set()
    for suit_start in (0, 9, 18):
        for rank in range(1, 10):
            tile = suit_start + rank - 1
            partners = []
            if rank > 3:
                partners.append(suit_start + rank - 4)
            if rank <= 6:
                partners.append(suit_start + rank + 2)
            if any(partner in discarded for partner in partners):
                suji.add(tile)
    return suji


def opponent_threat_score(state: GameState, observer: int, opponent: int) -> int:
    """以公開資訊估計對手威脅，不讀取對手手牌。"""
    if not 0 <= observer < 4 or not 0 <= opponent < 4:
        raise ValueError("observer 與 opponent 必須在 0-3 之間")
    if observer == opponent:
        raise ValueError("observer 與 opponent 不可相同")
    opponent_state = state.players[opponent]
    meld_count = len(opponent_state.melds)
    if meld_count >= 3:
        score = 4
    elif meld_count == 2:
        score = 3
    elif meld_count == 1:
        score = 2
    else:
        score = 0
    if meld_count and len(opponent_state.discards) >= 6:
        score += 1
    return min(score, 5)


def should_fold(state: GameState, player: int) -> bool:
    """判斷是否值得放棄進攻，進入完全防守。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    own = state.players[player]
    own_shanten = shanten(own.hand, own.open_melds)
    highest_threat = max(
        opponent_threat_score(state, player, opponent)
        for opponent in range(4) if opponent != player
    )
    return own_shanten >= 2 and highest_threat >= 3


def choose_rule_action_for_player(
    state: GameState, player: int, rng: Random | None = None
) -> Action | None:
    """只替指定玩家做決策，不讀取對手隱藏手牌。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    actions = legal_actions(state, player)
    win = next((action for action in actions if action.kind == ActionType.WIN), None)
    if win is not None:
        return win
    if state.phase == Phase.RESPONSE:
        current = state.players[player]
        baseline = shanten(current.hand, current.open_melds)
        for action in actions:
            if action.kind not in {ActionType.CHOW, ActionType.PUNG, ActionType.KONG}:
                continue
            concealed = current.hand.copy()
            claimed = state.last_discard
            assert claimed is not None
            skipped_claimed = False
            for tile in action.tiles:
                if tile == claimed and not skipped_claimed:
                    skipped_claimed = True
                    continue
                concealed[tile] -= 1
            if shanten(concealed, current.open_melds + 1) <= baseline:
                return action
        return next((action for action in actions if action.kind == ActionType.PASS), None)
    if state.phase == Phase.DISCARD:
        defensive = should_fold(state, player)
        tile = choose_discard(state.players[player].hand,
                              state.players[player].open_melds, rng,
                              visible_tile_counts(state, player), safe_tiles(state, player),
                              suji_tiles(state, player), defensive=defensive)
        return Action(ActionType.DISCARD, tile=tile)
    action = next((action for action in actions if action.kind == ActionType.DRAW), None)
    return action


def choose_rule_action(state: GameState, rng: Random | None = None) -> tuple[int, Action] | None:
    """模擬器相容的聚合器；實際玩家決策請使用指定座位 API。"""
    players = state.response_players if state.phase == Phase.RESPONSE else [state.current_player]
    for player in players:
        action = choose_rule_action_for_player(state, player, rng)
        if action is not None and action.kind == ActionType.WIN:
            return player, action
    for player in players:
        action = choose_rule_action_for_player(state, player, rng)
        if action is not None and action.kind != ActionType.PASS:
            return player, action
    if players:
        action = choose_rule_action_for_player(state, players[0], rng)
        return (players[0], action) if action is not None else None
    return None