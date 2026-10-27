import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.expert_train import labels, predict, teacher_gain, train  # noqa: E402
from training.model import MahjongNet  # noqa: E402


def data_with(means, ses, candidates):
    return {"means": np.array(means, np.float32), "ses": np.array(ses, np.float32),
            "candidates": np.array(candidates, np.int16)}


def test_labels_switch_only_when_clearly_better():
    data = data_with([[0.0, 0.5, 0.2], [0.0, 0.3, np.nan], [0.0, -1.0, 0.9]],
                     [[0.0, 0.4, 0.15], [0.0, 0.4, np.nan], [0.0, 0.1, 0.5]],
                     [[1, 2, 3], [4, 5, -1], [6, 7, 8]])
    # 第一個：第 2 名 0.5−0.4=0.1 勝過第 3 名 0.2−0.15=0.05；第二個：0.3 < 0.4 不換；第三個：0.9−0.5 換第 3 名
    assert labels(data).tolist() == [1, 0, 2]
    assert labels(data, margin=0.0).tolist() == [1, 1, 2]


def test_teacher_gain_counts_only_choices_among_candidates():
    data = data_with([[0.0, 1.0], [0.0, -2.0], [0.0, 3.0]], [[0, 0]] * 3, [[1, 2], [3, 4], [5, 6]])
    gain, _, inside = teacher_gain(np.array([2, 3, 9]), data)
    assert gain == pytest.approx(0.5) and inside == pytest.approx(2 / 3)


def test_training_moves_the_network_toward_the_labels():
    from training.encoding import CHANNELS

    torch.manual_seed(0)
    model = MahjongNet(width=16, blocks=1)
    rng = np.random.default_rng(0)
    obs = rng.random((64, CHANNELS, 34)).astype(np.float16)
    mask = np.zeros((64, 108), bool)
    mask[:, [3, 7]] = True
    data = {"obs": obs, "mask": mask, "candidates": np.tile(np.array([[3, 7]], np.int16), (64, 1))}
    target = np.ones(64, int)          # 一律學第 2 個候選（動作 7）
    train(model, data, target, "cpu", epochs=30, lr=3e-3, batch=32, kl_weight=0.0)
    assert (predict(model, obs, mask, "cpu") == 7).mean() > 0.9
