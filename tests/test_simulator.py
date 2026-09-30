from random import Random

import pytest

from game.rules import ActionType, GameState, Phase, PlayerState, legal_actions
from game.simulator import (
    _choose_with_player_policy,
    format_game_trace,
    make_uniform_policy,
    play_game,
    simulate_games,
)
from game.tiles import parse, to_counts


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
    assert first.discard_wins + first.self_draw_wins == first.wins
    assert first.total_tai >= 0
    assert first.total_discards > 0
    assert 0 <= first.deal_in_rate <= 1


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


def test_simulation_stats_report_win_type_and_average_tai():
    stats = simulate_games(10, seed=17)
    assert stats.discard_wins + stats.self_draw_wins == stats.wins
    assert stats.average_tai >= 0
    assert stats.total_discards > 0
    assert 0 <= stats.deal_in_game_rate <= 1
    assert 0 <= stats.discard_win_rate <= 1


def test_player_policy_can_control_one_seat_against_random_opponents():
    from agent.rule_agent import choose_rule_action_for_player
    from game.simulator import _choose_random_for_player

    def policy(state, player, rng):
        if player == 0:
            return choose_rule_action_for_player(state, player, rng)
        return _choose_random_for_player(state, player, rng)

    stats = simulate_games(5, seed=19, player_policy=policy)
    assert sum(stats.wins_by_player) == stats.wins
    assert stats.total_discards > 0


def test_uniform_policy_supports_same_strategy_self_play():
    from agent.rule_agent import choose_rule_action_for_player

    policy = make_uniform_policy(choose_rule_action_for_player)
    stats = simulate_games(2, seed=29, player_policy=policy)
    assert sum(stats.wins_by_player) == stats.wins
    assert sum(stats.discards_by_player) == stats.total_discards
    assert len(stats.deal_in_rate_by_player) == 4
    assert sum(stats.wins_by_player) == stats.wins
    assert sum(stats.self_draw_wins_by_player) == stats.self_draw_wins
    assert sum(stats.discard_wins_by_player) == stats.discard_wins
    assert len(stats.win_rate_by_player) == 4
    assert len(stats.deal_in_confidence_by_player) == 4
    assert len(stats.win_confidence_by_player) == 4
    assert all(0 <= low <= high <= 1 for low, high in stats.win_confidence_by_player)

def test_pung_by_later_seat_beats_chow_by_next_seat():
    players = [PlayerState() for _ in range(4)]
    players[0].discards = parse("5m")
    players[1].hand = to_counts(parse("46m1234567p1234s"))
    players[2].hand = to_counts(parse("55m1234567p1234s"))
    state = GameState(wall=[], players=players, phase=Phase.RESPONSE,
                      last_discard=parse("5m")[0], discard_player=0,
                      response_player=1, response_players=[1, 2, 3])

    def claim_anything(state, player, rng):
        actions = legal_actions(state, player)
        return next((a for a in actions if a.kind != ActionType.PASS), actions[0])

    player, action = _choose_with_player_policy(state, Random(0), claim_anything)
    assert (player, action.kind) == (2, ActionType.PUNG)


def test_full_round_rotates_dealer_and_keeps_it_on_dealer_win_or_draw():
    from random import Random

    from agent.rule_agent import choose_rule_action_for_player
    from game.simulator import play_match, payments

    match = play_match(Random(3), player_policy=choose_rule_action_for_player, first_dealer=2)
    dealers = [hand.state.dealer for hand in match.hands]
    streaks = [hand.state.dealer_streak for hand in match.hands]
    assert dealers[0] == 2 and streaks[0] == 0
    passes = 0
    for before, after, streak_after in zip(match.hands, match.hands[1:], streaks[1:]):
        kept = before.winner is None or before.winner == before.state.dealer
        if kept:
            assert after.state.dealer == before.state.dealer
            assert streak_after == before.state.dealer_streak + 1
        else:
            assert after.state.dealer == (before.state.dealer + 1) % 4 and streak_after == 0
            passes += 1
    last = match.hands[-1]
    assert passes == 3 and last.winner is not None and last.winner != last.state.dealer
    assert list(match.totals) == [sum(payments(h)[s] for h in match.hands) for s in range(4)]
    assert sum(match.totals) == 0


def test_dealer_streak_adds_tai_and_extra_payment():
    from game.rules import GameState, PlayerState
    from game.scoring import Score
    from game.simulator import GameResult, payments

    state = GameState([], [PlayerState() for _ in range(4)], dealer=1, dealer_streak=2)
    # 非莊家自摸：莊家多付莊家 1 台 + 連 2 拉 2 共 4 台
    tsumo = GameResult(state, 0, winner=0, score=Score(3, ()))
    pay = payments(tsumo)
    assert pay[2] == pay[3] == -(2 + 3) and pay[1] == -(2 + 3 + 5)
