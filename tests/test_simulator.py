from random import Random

import pytest

from game.rules import Phase
from game.simulator import format_game_trace, play_game, simulate_games


def test_random_game_reaches_terminal_state():
    result = play_game(Random(7))
    assert result.state.phase == Phase.ENDED
    assert result.steps > 0
    assert result.steps <= 10_000
    assert result.winner is None or 0 <= result.winner < 4


def test_many_random_games_are_reproducible_and_do_not_crash():
    first = simulate_games(50, seed=11)
    second = simulate_games(50, seed=11)
    assert first == second
    assert first.games == 50
    assert first.wins + first.draws == 50
    assert first.total_steps > 0


def test_invalid_simulation_arguments_are_rejected():
    with pytest.raises(ValueError):
        simulate_games(-1)
    with pytest.raises(ValueError):
        play_game(max_steps=0)


def test_game_history_is_optional_and_formatable():
    without_history = play_game(Random(3))
    with_history = play_game(Random(3), record_history=True)
    assert without_history.history == ()
    assert with_history.history
    trace = format_game_trace(with_history, limit=3)
    assert "P" in trace
    assert "結果：" in trace


def test_game_result_includes_score_when_a_game_is_won():
    from agent.rule_agent import choose_rule_action

    result = play_game(Random(3), choose_action=choose_rule_action)
    assert result.winner is not None
    assert result.score is not None
    assert result.score.patterns
    assert "基本胡" in result.score.patterns or result.score.tai >= 1