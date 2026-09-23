"""用隨機策略執行對局，驗證規則引擎可供 self-play 使用。"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .rules import Action, ActionType, GameState, Phase, apply_action, initial_state, legal_actions


@dataclass(frozen=True)
class GameResult:
    state: GameState
    steps: int
    winner: int | None = None
    win_tile: int | None = None
    win_by_discard: bool = False


@dataclass(frozen=True)
class SimulationStats:
    games: int
    wins: int
    draws: int
    total_steps: int

    @property
    def average_steps(self) -> float:
        return self.total_steps / self.games if self.games else 0.0


def choose_random_action(state: GameState, rng: Random) -> tuple[int, Action] | None:
    """選一個合法動作；胡牌優先，其餘動作隨機。"""
    if state.phase == Phase.RESPONSE:
        players = state.response_players or (
            [state.response_player] if state.response_player is not None else []
        )
    else:
        players = [state.current_player]
    if not players:
        return None
    available = [(player, legal_actions(state, player)) for player in players]
    available = [(player, actions) for player, actions in available if actions]
    if not available:
        return None
    winning = [
        (player, action)
        for player, actions in available
        for action in actions
        if action.kind == ActionType.WIN
    ]
    if winning:
        return winning[0]
    player, actions = rng.choice(available)
    return player, rng.choice(actions)


def play_game(
    rng: Random | None = None,
    max_steps: int = 10_000,
) -> GameResult:
    """執行一局隨機對局；牌牆耗盡或達到步數上限都視為流局。"""
    if max_steps <= 0:
        raise ValueError("max_steps 必須為正數")
    rng = rng or Random()
    state = initial_state(rng)
    for steps in range(1, max_steps + 1):
        if state.phase == Phase.ENDED:
            return GameResult(state, steps - 1)
        selected = choose_random_action(state, rng)
        if selected is None:
            state.phase = Phase.ENDED
            return GameResult(state, steps - 1)
        player, action = selected
        win_by_discard = state.phase == Phase.RESPONSE and action.kind == ActionType.WIN
        apply_action(state, action, player)
        if action.kind == ActionType.WIN:
            return GameResult(
                state,
                steps,
                winner=player,
                win_tile=action.tile,
                win_by_discard=win_by_discard,
            )
    state.phase = Phase.ENDED
    return GameResult(state, max_steps)


def simulate_games(
    games: int,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> SimulationStats:
    """執行多局隨機對局，回傳胡牌與流局統計。"""
    if games < 0:
        raise ValueError("games 不可為負數")
    rng = Random(seed)
    wins = 0
    total_steps = 0
    for _ in range(games):
        result = play_game(rng, max_steps)
        wins += result.winner is not None
        total_steps += result.steps
    return SimulationStats(games, wins, games - wins, total_steps)