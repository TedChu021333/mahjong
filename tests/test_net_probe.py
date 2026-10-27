import numpy as np
import pytest

torch = pytest.importorskip("torch")

import training.net_probe as net_probe  # noqa: E402
from training.model import MahjongNet  # noqa: E402
from training.net_probe import GreedyBatcher, NetPoint, evaluate, report, sample_points  # noqa: E402


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MahjongNet(width=16, blocks=1).eval()


def test_greedy_batcher_returns_the_best_legal_action(model):
    from training.encoding import CHANNELS

    batcher = GreedyBatcher(model, "cpu")
    obs = np.zeros((CHANNELS, 34), np.float32)
    mask = np.zeros(108, bool)
    mask[[3, 40]] = True
    with torch.no_grad():
        logits = model(torch.from_numpy(obs[None]))[0][0].numpy()
    try:
        assert batcher.ask(obs, mask) == (3 if logits[3] > logits[40] else 40)
    finally:
        batcher.close()


def test_points_rank_network_choices_and_evaluate_runs(model, monkeypatch):
    points = sample_points(model, "cpu", 2, seed=3, rate=0.3)
    assert len(points) == 2
    for point in points:
        assert point.kinds[0].startswith("net1") and len(point.candidates) == len(point.kinds) >= 2
        assert len(set(point.candidates)) == len(point.candidates)
    monkeypatch.setattr(net_probe, "_model", model)
    monkeypatch.setattr(net_probe, "THREADS", 4)
    same = NetPoint(points[0].state, points[0].player, [points[0].candidates[0]] * 2, ["net1", "net2"])
    matrices = evaluate(([same], 3, 1, True))
    # 同一張牌打兩次：同一副牌、同一個種子，結果完全一樣
    assert len(matrices[0]) == 3 and all(row[0] == row[1] for row in matrices[0])
    assert "net2" in report([same], matrices)
