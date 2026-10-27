import numpy as np
import pytest

torch = pytest.importorskip("torch")

import training.net_probe as net_probe  # noqa: E402
from training.expert_data import load_shards, run_task  # noqa: E402
from training.model import MahjongNet  # noqa: E402


def test_task_saves_candidates_and_skips_finished_work(tmp_path, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(net_probe, "_model", MahjongNet(width=16, blocks=1).eval())
    monkeypatch.setattr(net_probe, "THREADS", 4)
    path = tmp_path / "ex_00000.npz"
    assert run_task((3, 2, 2, str(path))) == 2
    assert run_task((3, 2, 2, str(path))) == 0          # 已完成的跳過
    data = load_shards([path])
    assert data["obs"].shape[0] == 2 and data["obs"].dtype == np.float16
    for mask, candidates, ses in zip(data["mask"], data["candidates"], data["ses"]):
        real = candidates[candidates >= 0]
        assert len(real) >= 2 and mask[real].all()       # 候選都是合法動作
        assert ses[0] == 0.0
