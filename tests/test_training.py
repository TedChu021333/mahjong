from random import Random

import numpy as np
import pytest

from game.rules import Action, ActionType, GameState, Phase, PlayerState, legal_actions
from game.simulator import play_game
from game.tiles import parse, to_counts
from training.actions import (
    ACTION_COUNT, CHOW_HIGH, CHOW_LOW, CHOW_MID, DECLARE, DISCARD, KONG, PASS, PUNG, WIN,
    action_index, legal_mask,
)
from training.encoding import CHANNELS, OFFSET, Noise, encode


def P(value):
    return parse(value)[0]


def test_action_indices_are_distinct_and_cover_the_space():
    assert ACTION_COUNT == WIN + 1 == 108
    assert action_index(Action(ActionType.DISCARD, tile=P("5m"))) == DISCARD + 4
    assert action_index(Action(ActionType.DECLARE, tile=P("1z"))) == DECLARE + 27
    assert action_index(Action(ActionType.KONG, tile=P("9s"))) == KONG + 26
    assert action_index(Action(ActionType.PUNG, tile=P("1m"))) == PUNG
    chow = [Action(ActionType.CHOW, tile=P("4m"), tiles=tuple(parse(t)))
            for t in ("456m", "345m", "234m")]
    assert [action_index(a) for a in chow] == [CHOW_LOW, CHOW_MID, CHOW_HIGH]
    assert action_index(Action(ActionType.PASS)) == PASS


def response_state():
    players = [PlayerState() for _ in range(4)]
    players[1].hand = to_counts(parse("35m44m99p123456s1z"))
    players[0].discards = [P("4m")]
    return GameState([0] * 40, players, current_player=0, phase=Phase.RESPONSE,
                     last_discard=P("4m"), discard_player=0, response_players=[1])


def test_legal_mask_marks_every_network_action():
    state = response_state()
    mask, lookup = legal_mask(legal_actions(state, 1))
    assert set(np.nonzero(mask)[0]) == {PASS, PUNG, CHOW_MID} and set(lookup) == {PASS, PUNG, CHOW_MID}


def test_encoding_uses_relative_seats_and_only_public_information():
    state = response_state()
    state.players[2].hand = to_counts(parse("999m"))  # 對手手牌不能出現在輸入裡
    planes = encode(state, 1)
    assert planes.shape == (CHANNELS, 34) and planes.dtype == np.float32
    assert planes[OFFSET["hand"] + 1, P("4m")] == 1 and planes[OFFSET["hand"] + 2, P("4m")] == 0
    assert planes[OFFSET["claim_tile"], P("4m")] == 1
    assert planes[OFFSET["discarder"] + 3].all()       # 打牌的是上家（相對座位 3）
    assert planes[OFFSET["discards"] + 3, P("4m")] == 0.25
    assert planes[OFFSET["unseen"], P("9m")] == 1      # 對家手上的 999m 看不到
    assert planes[OFFSET["unseen"], P("4m")] == 0.25   # 自己兩張 + 牌河一張


def test_noise_hides_river_and_declarations():
    state = response_state()
    state.players[0].discards = parse("1m2m3m4m5m6m7m8m9m4m")
    state.declared[0] = True
    clean = encode(state, 1)
    noisy = encode(state, 1, Noise(drop_discard=1.0, miss_declare=1.0), Random(0))
    assert clean[OFFSET["discards"] + 3].sum() > 0 and noisy[OFFSET["discards"] + 3].sum() == 0
    assert clean[OFFSET["declared"] + 3].all() and not noisy[OFFSET["declared"] + 3].any()


def test_min_discards_raises_the_discard_count_plane():
    state = response_state()
    state.min_discards = 10
    assert encode(state, 1)[OFFSET["discard_count"] + 2, 0] == pytest.approx(0.5)


def test_teacher_recorder_records_decisions_with_choices():
    from agent.rule_agent import choose_rule_action_for_player
    from training.teacher_data import Recorder

    recorder = Recorder(choose_rule_action_for_player, Noise(), Random(0))
    play_game(Random(3), player_policy=recorder, swap=True)
    assert len(recorder.actions) > 10
    for mask, action in zip(recorder.masks, recorder.actions):
        assert mask[action] and mask.sum() >= 2


def test_network_policy_plays_a_whole_game():
    pytest.importorskip("torch")
    from agent.nn_agent import NetworkPolicy
    from training.model import MahjongNet

    policy = NetworkPolicy(MahjongNet(width=16, blocks=1))
    result = play_game(Random(5), player_policy=policy, swap=True)
    assert result.state.phase == Phase.ENDED or result.winner is not None


def test_self_play_collects_games_and_updates_the_network():
    torch = pytest.importorskip("torch")
    from training.model import MahjongNet
    from training.selfplay import Batcher, collect, ppo_update

    model = MahjongNet(width=16, blocks=1).eval()
    before = [p.detach().clone() for p in model.parameters()]
    batcher = Batcher(model, "cpu", max_batch=4)
    records = collect(batcher, 3, 0, 0.5, Noise())
    batcher.close()
    assert len(records) == 3 and any(record.steps[seat] for record in records for seat in record.learners)
    stats = ppo_update(model, torch.optim.Adam(model.parameters(), 1e-3), records, "cpu", epochs=1)
    assert set(stats) == {"policy", "value", "entropy", "kl"}
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_hint_mode_can_use_the_network(tmp_path):
    pytest.importorskip("torch")
    from hint import advisor
    from hint.advisor import Observation
    from training.model import MahjongNet, save

    path = tmp_path / "net.pt"
    save(MahjongNet(width=16, blocks=1), path)
    try:
        advisor.use_policy("nn")
        advisor.use_network(str(path))
        hand = tuple(parse("123456789m123456p1z9s"))
        action = advisor.decide(Observation(hand, None, None, ()))
        assert action.kind in (ActionType.DISCARD, ActionType.DECLARE) and action.tile in hand
    finally:
        advisor.use_policy("rule")
