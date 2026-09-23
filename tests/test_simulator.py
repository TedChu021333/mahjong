from random import Random

import pytest

from game.rules import Phase
from game.simulator import play_game, simulate_games


def test_random_game_reaches_terminal_state():
    result = play_game(Random(7))
    assert result.state.phase == Phase.ENDED
    assert result.steps > 0
    assert result.steps <= 10_000
    assert result.winner is None or 0 <= result.winner < 4


def test_many_random_games_are_reproducible_and_do_not_crash():
    first = simulate_games(100, seed=11)
    second = simulate_games(100, seed=11)
    assert first == second
    assert first.games == 100
    assert first.wins + first.draws == 100
    assert first.total_steps > 0


def test_invalid_simulation_arguments_are_rejected():
    with pytest.raises(ValueError):
        simulate_games(-1)
    with pytest.raises(ValueError):
        play_game(max_steps=0)