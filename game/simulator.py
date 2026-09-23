"""用隨機策略執行對局，驗證規則引擎可供 self-play 使用。"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

from .rules import Action, ActionType, GameState, Phase, apply_action, initial_state, legal_actions
from .scoring import Score, score_hand
from .tiles import tile_name

ActionChooser = Callable[[GameState, Random], tuple[int, Action] | None]


@dataclass(frozen=True)
class GameResult:
    state: GameState
    steps: int
    winner: int | None = None
    win_tile: int | None = None
    win_by_discard: bool = False
    history: tuple["GameEvent", ...] = ()
    score: Score | None = None


@dataclass(frozen=True)
class GameEvent:
    step: int
    player: int
    phase: Phase
    action: Action


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
    choose_action: ActionChooser | None = None,
    record_history: bool = False,
) -> GameResult:
    """執行一局隨機對局；牌牆耗盡或達到步數上限都視為流局。"""
    if max_steps <= 0:
        raise ValueError("max_steps 必須為正數")
    rng = rng or Random()
    choose_action = choose_action or choose_random_action
    state = initial_state(rng)
    history: list[GameEvent] = []
    for steps in range(1, max_steps + 1):
        if state.phase == Phase.ENDED:
            return GameResult(state, steps - 1, history=tuple(history))
        selected = choose_action(state, rng)
        if selected is None:
            state.phase = Phase.ENDED
            return GameResult(state, steps - 1, history=tuple(history))
        player, action = selected
        if record_history:
            history.append(GameEvent(steps, player, state.phase, action))
        win_by_discard = state.phase == Phase.RESPONSE and action.kind == ActionType.WIN
        apply_action(state, action, player)
        if action.kind == ActionType.WIN:
            winner_state = state.players[player]
            score = score_hand(
                winner_state.hand,
                winner_state.melds,
                flowers=len(winner_state.flowers),
                self_draw=not win_by_discard,
                seat_wind=state.seat_winds[player],
                round_wind=state.round_wind,
            )
            return GameResult(
                state,
                steps,
                winner=player,
                win_tile=action.tile,
                win_by_discard=win_by_discard,
                history=tuple(history),
                score=score,
            )
    state.phase = Phase.ENDED
    return GameResult(state, max_steps, history=tuple(history))


def format_game_trace(result: GameResult, limit: int | None = None) -> str:
    """把單局 history 格式化成適合終端閱讀的逐步紀錄。"""
    events = result.history if limit is None else result.history[:limit]
    lines = []
    for event in events:
        detail = event.action.kind.value
        if event.action.tile is not None:
            detail += f" {tile_name(event.action.tile)}"
        if event.action.tiles:
            detail += " [" + " ".join(tile_name(tile) for tile in event.action.tiles) + "]"
        lines.append(f"{event.step:04d} P{event.player + 1} {event.phase.value:8s} {detail}")
    if limit is not None and len(result.history) > limit:
        lines.append(f"...（省略 {len(result.history) - limit} 步）")
    outcome = "流局" if result.winner is None else f"P{result.winner + 1} 胡牌"
    lines.append(f"結果：{outcome}，總步數 {result.steps}")
    return "\n".join(lines)


def simulate_games(
    games: int,
    seed: int | None = None,
    max_steps: int = 10_000,
    choose_action: ActionChooser | None = None,
) -> SimulationStats:
    """執行多局隨機對局，回傳胡牌與流局統計。"""
    if games < 0:
        raise ValueError("games 不可為負數")
    rng = Random(seed)
    wins = 0
    total_steps = 0
    for _ in range(games):
        result = play_game(rng, max_steps, choose_action)
        wins += result.winner is not None
        total_steps += result.steps
    return SimulationStats(games, wins, games - wins, total_steps)