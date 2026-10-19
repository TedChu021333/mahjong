from random import Random

from agent.opponent_model import Calibration, estimate_danger, sample_waits, unseen_counts
from game.rules import GameState, Meld, PlayerState
from game.tiles import parse, to_counts


def P(value):
    return parse(value)[0]


def calibration(tenpai=0.0):
    hits = int(tenpai * 1000)
    return Calibration(
        tenpai={(d, m): (hits, 1000) for d in range(16) for m in range(6)},
        # 向聽越低、進張越多，平均贏得越多
        offense={(s, o, w): (10, 100, max(0, 6 - s) * 40 + o * 10)
                 for s in range(-1, 8) for o in range(7) for w in range(15)},
        tai={(False, False): (4.0, 100), (True, False): (5.0, 100)},
        discard_buckets=16, wall_bucket=8, outs_buckets=(0, 4, 8, 12, 16, 24, 32),
    )


def table(**overrides):
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("123456789m123456p1z9s"))
    for seat, discards in overrides.items():
        players[int(seat[1])].discards = parse(discards)
    return GameState([], players)


def test_unseen_counts_subtract_everything_public():
    state = table(p1="1z1z")
    state.players[2].melds.append(Meld("pung", (P("5z"),) * 3, 1))
    pool = unseen_counts(state, 0)
    assert pool[P("1z")] == 1 and pool[P("5z")] == 1 and pool[P("1m")] == 3 and pool[P("7z")] == 4


def test_sampled_hands_are_tenpai_and_respect_forbidden_waits():
    rng = Random(0)
    pool = [4] * 34
    waits = [sample_waits(pool, 0, rng) for _ in range(200)]
    assert sum(w is not None for w in waits) > 150
    forbidden = set(range(9))  # 萬子全部不可能是他聽的
    for _ in range(100):
        result = sample_waits(pool, 0, rng, forbidden)
        assert result is None or not forbidden & set(result)


def test_declared_opponent_makes_his_discards_safe_and_others_dangerous():
    state = table(p2="1z2z3z")
    state.declared[2] = True
    danger = estimate_danger(state, 0, [P("1z"), P("9s")], Random(0), calibration(0.0), 200)
    assert danger.probability[P("1z")] == 0          # 宣告者打過的牌
    assert danger.probability[P("9s")] > 0.02
    assert danger.expected_loss[P("9s")] > 0


def test_no_threat_means_no_danger():
    state = table()
    danger = estimate_danger(state, 0, [P("9s")], Random(0), calibration(0.0), 50)
    assert danger.probability[P("9s")] == 0


def test_five_open_melds_leave_a_single_wait():
    rng = Random(1)
    for _ in range(20):
        waits = sample_waits([4] * 34, 5, rng)
        assert waits is not None and len(waits) == 1


def test_offense_value_is_monotone():
    cal = calibration()
    for level in range(0, 4):
        values = [cal.offense_value(level, outs, 60) for outs in (0, 4, 8, 16, 32)]
        assert values == sorted(values)
        assert cal.offense_value(level, 16, 60) >= cal.offense_value(level + 1, 16, 60)


def test_sparse_late_buckets_borrow_from_earlier_ones():
    cal = calibration(0.5)
    sparse = dict(cal.tenpai)
    sparse[(15, 0)] = (0, 3)  # 樣本太少
    cal = Calibration(sparse, cal.offense, cal.tai, cal.discard_buckets, cal.wall_bucket,
                      cal.outs_buckets)
    assert cal.tenpai_probability(20, 0) > 0.4
    assert 0 < cal.undeclared_tenpai_probability(20, 0, 0.4) < cal.tenpai_probability(20, 0)
    assert cal.undeclared_tenpai_probability(20, 0, 0.0) == 0


def test_missed_river_tiles_are_made_up_by_min_discards():
    # 實戰牌河只記到約八成：對手牌河記到 0 張，但自己已打 10 手，聽牌率要照 10 張查
    late = Calibration(
        tenpai={(d, m): (500 if d >= 8 else 0, 1000) for d in range(16) for m in range(6)},
        offense={}, tai={(False, False): (4.0, 100), (True, False): (5.0, 100)},
        discard_buckets=16, wall_bucket=8, outs_buckets=(0, 4, 8, 12, 16, 24, 32))
    state = table()
    assert estimate_danger(state, 0, [P("5p")], Random(0), late, 100,
                           undeclared_share=1.0).probability[P("5p")] == 0
    state.min_discards = 10
    assert estimate_danger(state, 0, [P("5p")], Random(0), late, 100,
                           undeclared_share=1.0).probability[P("5p")] > 0.02


def test_tiles_discarded_after_a_declaration_are_nearly_safe():
    from random import Random

    from agent.opponent_model import estimate_danger
    from game.rules import GameState, Phase, PlayerState
    from game.tiles import parse, to_counts

    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("1479m258p369s1234567z"))
    state = GameState(list(range(60)), players, phase=Phase.DISCARD)
    state.declared[1] = True
    five = parse("5p")[0]
    before = estimate_danger(state, 0, [five], Random(0), samples=40, min_valid=150)
    state.declared_safe[1] = {five}
    after = estimate_danger(state, 0, [five], Random(0), samples=40, min_valid=150)
    assert after.probability[five] < before.probability[five] * 0.3


def test_sampling_continues_until_enough_valid_hands():
    from random import Random

    import agent.opponent_model as om
    from game.rules import GameState, Phase, PlayerState
    from game.tiles import parse, to_counts

    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("1479m258p369s1234567z"))
    players[1].discards = parse("123456789m")  # 宣告者打過很多：大部分抽樣作廢
    state = GameState(list(range(60)), players, phase=Phase.DISCARD)
    state.declared[1] = True
    counted = []
    original = om.sample_waits

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        counted.append(result is not None)
        return result

    om.sample_waits = spy
    try:
        om.estimate_danger(state, 0, [0], Random(0), samples=10, min_valid=50)
    finally:
        om.sample_waits = original
    assert sum(counted) >= 50 or len(counted) >= 50 * om.MAX_ATTEMPTS_PER_VALID
