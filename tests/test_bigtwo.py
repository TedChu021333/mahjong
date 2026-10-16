from random import Random

import pytest

from bigtwo.agents import plan_hand, random_policy, rule_policy, smallest_policy
from bigtwo.cards import CLUB_THREE, card_code, parse_cards
from bigtwo.combos import Kind, all_combos, beats, classify, legal_moves
from bigtwo.game import View, continue_game, penalty, penalty_conditions, play_game, swap_cards
from bigtwo.mc_agent import make_mc_policy
from bigtwo.rules import Rules


def c(text):
    return classify(parse_cards(text))


def test_card_order_spade_two_is_biggest_club_three_smallest():
    cards = parse_cards("3c 3s 2c 2s Ah")
    assert min(cards) == CLUB_THREE
    assert card_code(max(cards)) == "2s"
    assert parse_cards("10h") == parse_cards("Th")


def test_singles_compare_rank_then_suit():
    assert beats(c("4c"), c("3s"))
    assert beats(c("8s"), c("8h"))
    assert not beats(c("Ah"), c("2c"))


def test_pairs_compare_by_higher_suit():
    assert c("8d8h").kind == Kind.PAIR
    assert beats(c("8c8s"), c("8d8h"))
    assert classify(parse_cards("8d9h")) is None


def test_straight_order_a2345_smallest_23456_biggest():
    a2345 = c("Ac 2c 3d 4h 5s")
    low = c("3c 4d 5h 6s 7c")
    top = c("Tc Js Qd Kc Ah")
    big = c("2c 3d 4h 5s 6c")
    assert a2345.kind == low.kind == top.kind == big.kind == Kind.STRAIGHT
    assert beats(low, a2345) and beats(top, low) and beats(big, top)
    assert classify(parse_cards("Jc Qd Kh Ac 2s")) is None


def test_same_straight_compares_top_card_suit():
    assert beats(c("9c Td Jh Qd Ks"), c("9s Ts Js Qs Kh"))


def test_five_card_kinds_order():
    straight = c("9c Td Jh Qd Kc")
    full = c("3c 3d 3h 4c 4d")
    four = c("5c 5d 5h 5s 3c")
    sflush = c("4h 5h 6h 7h 8h")
    assert full.kind == Kind.FULL_HOUSE and four.kind == Kind.FOUR
    assert sflush.kind == Kind.STRAIGHT_FLUSH
    assert beats(full, straight) and beats(four, full) and beats(sflush, four)
    assert beats(c("Qd Qh Qs 3d 3h"), c("Jc Jd Jh Ac As"))


def test_flush_not_a_kind_by_default():
    assert classify(parse_cards("3h 5h 7h 9h Jh")) is None
    assert classify(parse_cards("3h 5h 7h 9h Jh"), Rules(flush_enabled=True)).kind == Kind.FLUSH


def test_bombs_only_beat_singles_when_enabled():
    four = c("5c 5d 5h 5s 3c")
    assert beats(four, c("2s"))
    assert beats(c("4h 5h 6h 7h 8h"), c("2c2s"))
    assert not beats(four, c("2s"), Rules(bombs_beat_smaller=False))
    assert not beats(c("2s"), four)


def test_legal_moves_follow_and_first_lead():
    hand = parse_cards("3c 3d 4h 5s 6c 7d 9s 2h")
    moves = legal_moves(hand, None, must_include=CLUB_THREE)
    assert moves and all(CLUB_THREE in m.cards for m in moves)
    assert {m.kind for m in moves} == {Kind.SINGLE, Kind.PAIR, Kind.STRAIGHT}
    follow = legal_moves(hand, c("Ks"), Rules(bombs_beat_smaller=False))
    assert [card_code(m.cards[0]) for m in follow] == ["2h"]


def test_all_combos_counts_full_house_and_four():
    hand = parse_cards("5c 5d 5h 5s 9c 9d")
    kinds = [m.kind for m in all_combos(hand)]
    assert kinds.count(Kind.FULL_HOUSE) == 4      # 4 種三條 × 一對 9
    assert kinds.count(Kind.FOUR) == 2


def test_penalty_conditions_stack():
    hand = parse_cards("3d 4d 5d 6d 7d 8d 9d Td Jd Qd 2h")
    last = c("3c")
    cond = penalty_conditions(hand, last, False)
    assert cond == ["超過9張牌", "老2未打", "最後脫手3"]
    assert penalty(hand, cond) == 11 * 8
    assert penalty(hand, cond, Rules(rate=3)) == 11 * 27


def test_penalty_matches_recorded_settlement():
    # 錄影 18:24 結算：贏家梅花三脫手，本局 2 倍、每張 5 分
    winner_last = c("3c")
    me = parse_cards("4c 4d Ts")
    other = parse_cards("4s 8d 9h Td Js Qs 2h")
    assert 5 * penalty(me, penalty_conditions(me, winner_last, False)) == 30
    assert 5 * penalty(other, penalty_conditions(other, winner_last, False)) == 140


def test_plan_hand_uses_few_plays():
    hand = parse_cards("3c 4d 5h 6s 7c 9c 9d 9h Kc Kd As 2s 2h")
    plan = plan_hand(hand)
    assert sorted(c for combo in plan for c in combo.cards) == sorted(hand)
    assert len(plan) <= 5


@pytest.mark.parametrize("policy", [random_policy, smallest_policy, rule_policy])
def test_games_finish_and_scores_balance(policy):
    rng = Random(1)
    for _ in range(300):
        result = play_game([policy] * 4, rng)
        assert sum(result.scores) == 0
        assert not result.hands[result.winner]
        assert result.history[0].seat == result.starter


def test_rule_policy_beats_random():
    rng = Random(7)
    total = 0
    for i in range(400):
        seat = i % 4
        policies = [random_policy] * 4
        policies[seat] = rule_policy
        total += play_game(policies, rng).scores[seat]
    assert total > 0


def test_pass_locks_out_variant_runs():
    rng = Random(3)
    for _ in range(100):
        result = play_game([rule_policy] * 4, rng, Rules(pass_locks_out=True, first_must_include_club3=True))
        assert CLUB_THREE in result.history[0].combo.cards
        assert sum(result.scores) == 0


def test_continue_game_replays_the_same_game():
    hands = [sorted(parse_cards(h)) for h in (
        "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c",
        "3d 4d 5d 6d 7d 8d 9d Td Jd Qd Kd Ad 2d",
        "3h 4h 5h 6h 7h 8h 9h Th Jh Qh Kh Ah 2h",
        "3s 4s 5s 6s 7s 8s 9s Ts Js Qs Ks As 2s")]
    seen = []

    def recorder(view, rng):
        if len(view.history) == 5 and not seen:
            seen.append(view)
        return rule_policy(view, rng)

    full = play_game([recorder] * 4, Random(0), hands=[list(h) for h in hands])
    view = seen[0]
    played = view.played
    now = [[c for c in h if c not in played] for h in hands]
    move = rule_policy(view, Random(0))
    again = continue_game(view, now, move, [rule_policy] * 4, Random(0))
    assert again.scores == full.scores and again.winner == full.winner


def test_mc_policy_plays_legal_games():
    mc = make_mc_policy(samples=2, limit=3)
    result = play_game([mc, rule_policy, rule_policy, rule_policy], Random(5))
    assert sum(result.scores) == 0


def test_swap_returns_same_counts_from_pooled_cards():
    rng = Random(2)
    hands = [sorted(parse_cards(h)) for h in (
        "3c 4c 5c 6c 7c 8c 9c Tc Jc Qc Kc Ac 2c",
        "3d 4d 5d 6d 7d 8d 9d Td Jd Qd Kd Ad 2d",
        "3h 4h 5h 6h 7h 8h 9h Th Jh Qh Kh Ah 2h",
        "3s 4s 5s 6s 7s 8s 9s Ts Js Qs Ks As 2s")]
    picks = [lambda h, r: list(h[:3]), lambda h, r: list(h[:2]), lambda h, r: [], lambda h, r: list(h[:3])]
    new = swap_cards(hands, picks, rng)
    assert [len(h) for h in new] == [13] * 4
    assert sorted(c for h in new for c in h) == list(range(52))
    assert new[2] == hands[2]
    pool = set(hands[0][:3] + hands[1][:2] + hands[3][:3])
    assert set(new[1]) - set(hands[1]) <= pool
