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


def test_parallel_actors_collect_the_requested_games():
    pytest.importorskip("torch")
    import multiprocessing

    from training.model import MahjongNet
    from training.selfplay import _actor_init, collect_parallel

    model = MahjongNet(width=16, blocks=1)
    with multiprocessing.get_context("spawn").Pool(2, _actor_init, (model.config, "cpu")) as pool:
        records = collect_parallel(pool, model, 2, 3, 0, 0.5)
    assert len(records) == 3


def test_guard_stops_when_the_policy_drifts_random_or_weaker():
    pytest.importorskip("torch")
    from training.selfplay import Guard

    guard = Guard()
    assert all(guard.update(0.3, -0.02) is None for _ in range(2 * Guard.WINDOW))
    assert all(guard.update(0.6, -0.02) is None for _ in range(Guard.WINDOW - 1))
    assert "熵" in guard.update(0.6, -0.02)
    weaker = Guard()
    for _ in range(Guard.WINDOW):
        weaker.update(0.3, -0.02)
    assert [weaker.update(0.3, -0.08) for _ in range(Guard.WINDOW)][-1].startswith("胡牌率")


def test_league_opponents_take_seats_but_are_not_trained():
    pytest.importorskip("torch")
    from training.model import MahjongNet
    from training.selfplay import Batcher, collect

    learner = Batcher(MahjongNet(width=16, blocks=1).eval(), "cpu", max_batch=4)
    past = Batcher(MahjongNet(width=16, blocks=1).eval(), "cpu", max_batch=4)
    records = collect(learner, 4, 0, 0.0, Noise(), past, pool_share=0.6)
    learner.close()
    past.close()
    pool_seats = [seat for record in records for seat in range(4)
                  if seat not in record.learners and seat not in record.rule_seats]
    assert pool_seats and all(not record.rule_seats for record in records)
    assert all(set(record.steps) == set(record.learners) for record in records)


def test_efficiency_override_prefers_outs_early_and_keeps_network_choice_later():
    pytest.importorskip("torch")
    from agent.nn_agent import efficient_discard
    from game.rules import GameState, Phase, PlayerState, legal_actions
    from game.tiles import to_counts
    from training.actions import legal_mask

    # 10/15 02:52：一萬二二二萬 六萬 八九萬 一三筒 七八八八八九條 東中，網路打六萬（進張 20）、打東 24
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("1222689m13p788889s15z"))
    state = GameState([0] * 60, players, phase=Phase.DISCARD, last_drawn=parse("5z")[0])
    actions = legal_actions(state, 0)
    _, lookup = legal_mask(actions)
    choice = Action(ActionType.DISCARD, tile=parse("6m")[0])
    logits = np.zeros(108, dtype=np.float32)
    picked = efficient_discard(state, 0, choice, lookup, logits, turns=6)
    assert picked.tile in (parse("1z")[0], parse("5z")[0])   # 孤張字牌（進張最多）
    players[0].discards = parse("1s2s3s4s5s6s")              # 已打 6 手：照網路
    assert efficient_discard(state, 0, choice, lookup, logits, turns=6) == choice
    players[0].discards = []
    state.declared[2] = True                                   # 有人宣告：照網路
    assert efficient_discard(state, 0, choice, lookup, logits, turns=6) == choice


def test_efficiency_override_does_not_let_the_network_go_back_a_shanten():
    pytest.importorskip("torch")
    from agent.nn_agent import efficient_discard
    from game.rules import GameState, Phase, PlayerState, legal_actions
    from game.tiles import to_counts
    from training.actions import legal_mask

    # 10/21 21:34：五六萬 二三三四筒 一二二四五六八九條 發發 南，網路打六萬（3 向聽），打南是 2 向聽
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("56m2334p12245689s66z2z"))
    players[0].discards = parse("7z")
    state = GameState([0] * 60, players, phase=Phase.DISCARD, last_drawn=parse("2z")[0])
    _, lookup = legal_mask(legal_actions(state, 0))
    choice = Action(ActionType.DISCARD, tile=parse("6m")[0])
    picked = efficient_discard(state, 0, choice, lookup, np.zeros(108, dtype=np.float32), turns=6)
    assert picked.tile in (parse("2z")[0], parse("3p")[0])  # 都還是 2 向聽、進張 20；孤張南最孤立
    assert picked.tile == parse("2z")[0]


def test_guard_takes_a_claim_the_network_passed_when_it_widens_the_hand():
    pytest.importorskip("torch")
    from agent.nn_agent import widening_claim
    from game.rules import GameState, Phase, PlayerState, legal_actions

    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("2666m345789p122455s"))
    players[2].discards = parse("2s")
    state = GameState([0] * 60, players, phase=Phase.RESPONSE, last_discard=parse("2s")[0],
                      discard_player=2, response_players=[0])
    action, note = widening_claim(state, 0, legal_actions(state, 0))
    assert action.kind == ActionType.PUNG and "4 →" in note
    state.declared[0] = True  # 自己聽牌鎖手時不會
    assert widening_claim(state, 0, legal_actions(state, 0)) is None


def test_paired_games_share_the_deal_and_use_each_other_as_baseline():
    torch = pytest.importorskip("torch")
    from training.model import MahjongNet
    from training.selfplay import Batcher, collect, play_one, play_pair, ppo_update

    model = MahjongNet(width=16, blocks=1).eval()
    batcher = Batcher(model, "cpu", max_batch=4)
    try:
        # 同一副牌、同一個 copy：完全重現（牌牆、對手、網路抽樣都一樣）
        a = play_one(7, batcher, 0.5, Noise(), copy=0)
        b = play_one(7, batcher, 0.5, Noise(), copy=0)
        assert a.reward == b.reward and a.learners == b.learners
        assert [s.action for s in a.steps[a.learners[0]]] == [s.action for s in b.steps[b.learners[0]]]
        first, second = play_pair(7, batcher, 0.5, Noise())
        assert first.learners == second.learners and first.rule_seats == second.rule_seats
        assert first.baseline == second.reward and second.baseline == first.reward
        records = collect(batcher, 4, 0, 0.5, Noise(), paired=True)
    finally:
        batcher.close()
    assert len(records) == 4 and all(record.baseline is not None for record in records)
    stats = ppo_update(model, torch.optim.Adam(model.parameters(), 1e-3), records, "cpu", epochs=1)
    assert set(stats) == {"policy", "value", "entropy", "kl"}


def test_parallel_actors_collect_paired_games():
    pytest.importorskip("torch")
    import multiprocessing

    from training.model import MahjongNet
    from training.selfplay import _actor_init, collect_parallel

    model = MahjongNet(width=16, blocks=1)
    with multiprocessing.get_context("spawn").Pool(2, _actor_init, (model.config, "cpu")) as pool:
        records = collect_parallel(pool, model, 2, 6, 0, 0.5, paired=True)
    assert len(records) == 6 and all(record.baseline is not None for record in records)


def test_oracle_planes_show_hidden_hands_and_next_draws():
    from random import Random

    from game.rules import initial_state
    from training.encoding import CHANNELS
    from training.oracle_value import HIDDEN_CHANNELS, hidden_planes, play_and_record

    state = initial_state(Random(3))
    planes = hidden_planes(state, 0)
    assert planes.shape == (HIDDEN_CHANNELS, 34)
    # 下家（相對座位 1）的暗手：張數 ≥1 的平面和他的手牌一致
    assert (planes[0] == (np.asarray(state.players[1].hand) >= 1)).all()
    next_tile = state.wall[-1]
    if next_tile < 34:
        assert planes[16, next_tile] == 1
    public, hidden, target = play_and_record(11)
    assert public.shape[1:] == (CHANNELS, 34) and hidden.shape[1:] == (HIDDEN_CHANNELS, 34)
    assert len(public) == len(hidden) == len(target)


def test_attack_mode_claims_when_outs_grow_and_only_blocks_very_dangerous_tiles(monkeypatch):
    pytest.importorskip("torch")
    import agent.safety as safety
    from agent.nn_agent import NetworkPolicy
    from agent.opponent_model import Danger
    from training.model import MahjongNet

    policy = NetworkPolicy(MahjongNet(width=16, blocks=1), guard=True, attack=True)
    # 一二二條 四五五條（一向聽 4 張），對家打二條：碰了向聽不變、進張變多 → 極致進攻要碰
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("2666m345789p122455s"))
    players[2].discards = parse("2s")
    state = GameState([0] * 60, players, phase=Phase.RESPONSE, last_discard=parse("2s")[0],
                      discard_player=2, response_players=[0])
    actions = legal_actions(state, 0)
    passing = next(a for a in actions if a.kind == ActionType.PASS)
    assert policy._attack_claim(state, 0, passing, actions).kind == ActionType.PUNG
    assert "極致進攻" in policy.last_note

    # 有人宣告：放槍 20% 的照打，40% 的換成門檻內進攻最好的
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("12399m456p789s111z22z5s"))
    state = GameState([0] * 60, players, phase=Phase.DISCARD, last_drawn=parse("5s")[0])
    state.declared[2] = True
    risk = {t: 0.01 for t in range(34)}
    risk[parse("5s")[0]] = 0.2
    monkeypatch.setattr(safety, "estimate_danger",
                        lambda *args, **kwargs: Danger(dict(risk), {t: 0.0 for t in risk}))
    assert safety.attack_discard(state, 0, parse("5s")[0], Random(0)) == (parse("5s")[0], None)
    risk[parse("5s")[0]] = 0.4
    tile, note = safety.attack_discard(state, 0, parse("5s")[0], Random(0))
    assert tile != parse("5s")[0] and "太危險" in note


def test_attack_pauses_for_far_hands_and_prefers_safer_equal_tiles(monkeypatch):
    pytest.importorskip("torch")
    import agent.safety as safety
    from agent.nn_agent import NetworkPolicy
    from agent.opponent_model import Danger
    from training.model import MahjongNet

    policy = NetworkPolicy(MahjongNet(width=16, blocks=1), guard=True, attack=True)
    players = [PlayerState() for _ in range(4)]
    players[0].hand = to_counts(parse("12399m456p789s111z22z5s"))   # 聽牌
    state = GameState([0] * 60, players, phase=Phase.DISCARD, last_drawn=parse("5s")[0])
    assert policy._attacking(state, 0)
    state.declared[2] = True
    assert policy._attacking(state, 0)                               # 聽牌：照樣進攻
    players[0].hand = to_counts(parse("1479m258p369s1234567z"))      # 很遠的手
    assert not policy._attacking(state, 0)                           # 有人宣告又很遠：暫停
    state.declared[2] = False
    assert policy._attacking(state, 0)

    # 進攻一樣（向聽、進張相同）的兩張孤張字牌：改打比較安全的
    players[0].hand = to_counts(parse("123m456p789s11z22z56s67z"))
    state.declared[2] = True
    risk = {t: 0.01 for t in range(34)}
    risk[parse("7z")[0]] = 0.08
    monkeypatch.setattr(safety, "estimate_danger",
                        lambda *args, **kwargs: Danger(dict(risk), {t: 0.0 for t in risk}))
    tile, note = safety.attack_discard(state, 0, parse("7z")[0], Random(0))
    assert tile != parse("7z")[0] and "進攻一樣" in note
