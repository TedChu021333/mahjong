"""以向聽數與有效進張為核心的簡單規則式 AI。"""
from __future__ import annotations

from random import Random

from .shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions


def choose_discard(hand: list[int], n_open_melds: int = 0,
                   rng: Random | None = None) -> int:
    """選一張牌打出：最低向聽優先，再取有效進張最多。"""
    if sum(hand) not in (17 - 3 * n_open_melds, 16 - 3 * n_open_melds):
        raise ValueError("選擇打牌時手牌張數不正確")
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
    best = []
    for next_shanten, tile, after_discard in candidates:
        if next_shanten != minimum_shanten:
            continue
        outs = len(effective_tiles(after_discard, n_open_melds))
        best.append((-outs, tile))
    best_outs = min(score[0] for score in best)
    best_tiles = [tile for outs, tile in best if outs == best_outs]
    return (rng or Random()).choice(best_tiles)


def choose_rule_action(state: GameState, rng: Random | None = None) -> tuple[int, Action] | None:
    """選擇規則式 AI 動作；目前碰吃槓先保守過牌。"""
    if state.phase == Phase.RESPONSE:
        players = state.response_players or (
            [state.response_player] if state.response_player is not None else []
        )
        for player in players:
            actions = legal_actions(state, player)
            win = next((action for action in actions if action.kind == ActionType.WIN), None)
            if win is not None:
                return player, win
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
                              state.players[player].open_melds, rng)
        return player, Action(ActionType.DISCARD, tile=tile)
    action = next((action for action in actions if action.kind == ActionType.DRAW), None)
    return (player, action) if action is not None else None