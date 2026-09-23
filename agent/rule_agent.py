"""以向聽數與有效進張為核心的簡單規則式 AI。"""
from __future__ import annotations

from random import Random
from typing import Sequence

from .shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions


def choose_discard(hand: list[int], n_open_melds: int = 0,
                   rng: Random | None = None,
                   visible_counts: Sequence[int] | None = None,
                   safe_tiles: Sequence[int] = ()) -> int:
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
    best = []
    for next_shanten, tile, after_discard in candidates:
        if next_shanten != minimum_shanten:
            continue
        effective = effective_tiles(after_discard, n_open_melds)
        if visible_counts is None:
            outs = len(effective)
        else:
            outs = sum(max(0, 4 - visible_counts[tile]) for tile in effective)
        best.append((-outs, 0 if tile in safe_set else 1, tile))
    best_score = min((outs, risk) for outs, risk, _ in best)
    best_tiles = [tile for outs, risk, tile in best if (outs, risk) == best_score]
    return (rng or Random()).choice(best_tiles)


def visible_tile_counts(state: GameState) -> list[int]:
    """統計自己手牌、所有牌河與副露中已知的 34 種牌。"""
    visible = [0] * 34
    for player in state.players:
        for tile, count in enumerate(player.hand):
            visible[tile] += count
        for tile in player.discards:
            visible[tile] += 1
        for meld in player.melds:
            for tile in meld.tiles:
                visible[tile] += 1
    if any(count > 4 for count in visible):
        raise ValueError("牌局狀態出現超過 4 張同種牌")
    return visible


def safe_tiles(state: GameState, player: int) -> set[int]:
    """回傳至少一位對手已打過的現物牌。"""
    if not 0 <= player < 4:
        raise ValueError("player 必須在 0-3 之間")
    safe = set()
    for opponent, opponent_state in enumerate(state.players):
        if opponent != player:
            safe.update(opponent_state.discards)
    return safe


def choose_rule_action(state: GameState, rng: Random | None = None) -> tuple[int, Action] | None:
    """選擇規則式 AI 動作；胡牌優先，副露不應惡化向聽數。"""
    if state.phase == Phase.RESPONSE:
        players = state.response_players or (
            [state.response_player] if state.response_player is not None else []
        )
        for player in players:
            actions = legal_actions(state, player)
            win = next((action for action in actions if action.kind == ActionType.WIN), None)
            if win is not None:
                return player, win
        for player in players:
            actions = legal_actions(state, player)
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
                    return player, action
        if players:
            pass_action = next(action for action in legal_actions(state, players[0])
                                if action.kind == ActionType.PASS)
            return players[0], pass_action
        return None

    player = state.current_player
    actions = legal_actions(state, player)
    win = next((action for action in actions if action.kind == ActionType.WIN), None)
    if win is not None:
        return player, win
    if state.phase == Phase.DISCARD:
        tile = choose_discard(state.players[player].hand,
                              state.players[player].open_melds, rng,
                              visible_tile_counts(state), safe_tiles(state, player))
        return player, Action(ActionType.DISCARD, tile=tile)
    action = next((action for action in actions if action.kind == ActionType.DRAW), None)
    return (player, action) if action is not None else None