from random import Random

from agent.mc_agent import Option, choose_mc_action_for_player, determinize, evaluate_options
from agent.rollout import BASE, Seat, heuristic_discard, play_out
from game.rules import ActionType, GameState, Phase, PlayerState
from game.tiles import parse, to_counts
from tests.test_opponent_model import calibration


def P(value):
    return parse(value)[0]


def C(value):
    return to_counts(parse(value))


def test_ron_ends_the_rollout_with_the_discarder_paying():
    junk = "1479m258p369s234567z"
    seats = [Seat(C("123456789m123456p"), 0), Seat(C("123456789m123456p1z"), 0),
             Seat(C(junk), 0), Seat(C(junk), 0)]
    result = play_out(seats, [], 0, P("1z"), 6.0, 5.0)
    assert result == [-(BASE + 5.0), BASE + 5.0, 0.0, 0.0]  # 下家單騎東胡：底 + 5 台


def test_declared_winner_gets_one_more_tai_and_tsumo_is_paid_by_all():
    junk = "1479m258p369s234567z"  # 16 張、離聽牌很遠
    seats = [Seat(C("1479m258p369s1234567z"), 0), Seat(C("123456789m123456p1z"), 0, declared=True),
             Seat(C(junk), 0), Seat(C(junk), 0)]
    seats[0].hand[P("9s")] -= 1
    result = play_out(seats, [P("1z")], 0, P("9s"), 6.0, 5.0)  # 下家摸到東自摸
    assert result[1] == 3 * (BASE + 6 + 1)
    assert result[0] == result[2] == result[3] == -(BASE + 6 + 1)


def test_heuristic_discard_throws_isolated_tiles():
    assert heuristic_discard(C("123456789m12456p99s1z")) == P("1z")


def table_state(**discards):
    players = [PlayerState() for _ in range(4)]
    players[0].hand = C("123456789m12345p9s17z")
    for seat, tiles in discards.items():
        players[int(seat[1])].discards = parse(tiles)
    return GameState(list(range(40)), players, phase=Phase.DISCARD)


def test_determinized_hands_use_only_unseen_tiles():
    state = table_state(p1="1z2z3z", p2="4z5z", p3="7z6z")
    state.declared[1] = True
    seats, wall = determinize(state, 0, Random(0), calibration(0.3), 0.4)
    total = [0] * 34
    for seat in seats:
        for tile, count in enumerate(seat.hand):
            total[tile] += count
    for tile in wall:
        total[tile] += 1
    for tile in range(34):
        visible = sum(p.discards.count(tile) for p in state.players)
        assert total[tile] + visible <= 4
    assert sum(seats[1].hand) == 16 and len(wall) == state.drawable


def test_declaring_and_discarding_options_are_compared():
    state = table_state()
    options = [Option(P("1z"), False), Option(P("7z"), False)]
    values = evaluate_options(state, 0, options, Random(0), calibration(0.0), 4, 0.0)
    assert set(values) == set(options)


def test_mc_agent_returns_a_legal_discard_and_never_skips_a_win():
    state = table_state(p1="1z2z3z")
    action = choose_mc_action_for_player(state, 0, Random(0), calibration(0.0), rollouts=3)
    assert action.kind in (ActionType.DISCARD, ActionType.DECLARE)
    assert state.players[0].hand[action.tile] > 0
    winning = table_state()
    winning.players[0].hand = C("123456789m123456p11z")
    winning.last_drawn = P("1z")
    assert choose_mc_action_for_player(winning, 0, Random(0), calibration(0.0),
                                       rollouts=3).kind == ActionType.WIN


def test_only_significant_differences_change_the_default():
    from agent.mc_agent import significantly_better

    default = [10, -10, 8, -6, 0, 4, -9, 7]
    small = [1, -3, 4, -2, 3, -1, 2, -2]   # 平均只多 0.25，每副牌有好有壞
    large = [5, 6, 4, 5, 7, 3, 5, 6]       # 每副牌都好約 5 台
    assert not significantly_better([d + x for d, x in zip(default, small)], default)
    assert significantly_better([d + x for d, x in zip(default, large)], default)
    assert not significantly_better([d - x for d, x in zip(default, large)], default)


def test_parallel_evaluation_keeps_options_paired():
    from multiprocessing import Pool

    from agent.mc_agent import evaluate_options_parallel

    state = table_state(p1="1z2z3z")
    options = [Option(P("1z"), False), Option(P("7z"), False)]
    with Pool(2) as pool:
        results = evaluate_options_parallel(pool, 2, state, 0, options, Random(0), 4, 0.0, None)
    assert len(results[options[0]]) == len(results[options[1]]) == 4


def test_dealer_involvement_costs_extra_in_rollouts_and_danger():
    from agent.opponent_model import estimate_danger
    from agent.rollout import dealer_bonus

    assert dealer_bonus(None, 0, 1, 2) == 0 and dealer_bonus(2, 0, 1, 3) == 0
    assert dealer_bonus(2, 0, 1, 2) == 1 and dealer_bonus(2, 3, 2, 0) == 7
    junk = "1479m258p369s234567z"
    seats = [Seat(C("123456789m123456p"), 0), Seat(C("123456789m123456p1z"), 0),
             Seat(C(junk), 0), Seat(C(junk), 0)]
    # 下家是連 2 的莊家：放槍多付 1 + 4 台
    assert play_out(seats, [], 0, P("1z"), 6.0, 5.0, dealer=1, streak=2)[0] == -(BASE + 5 + 5)

    state = table_state(p2="1z2z3z")
    state.declared[2] = True
    plain = estimate_danger(state, 0, [P("9s")], Random(0), calibration(0.0), 100)
    state.dealer, state.dealer_streak = 2, 1
    dealer = estimate_danger(state, 0, [P("9s")], Random(0), calibration(0.0), 100)
    assert dealer.expected_loss[P("9s")] > plain.expected_loss[P("9s")]
