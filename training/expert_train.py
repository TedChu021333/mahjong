"""專家迭代：用 training.expert_data 的教材微調網路。

標籤：老師（網路自己打完的模擬）說某張候選比網路第一名好超過 MARGIN 倍標準誤，就學那張；否則學網路第一名。
損失 = 對標籤的交叉熵 + KL_WEIGHT × KL(原本網路 ‖ 新網路)，後者讓其他偏好不要跑掉。只動策略，價值頭不訓練。
留 HOLDOUT 的局面不訓練：看新網路在這些局面挑的牌，照老師的模擬數字平均比網路第一名多幾台（挑牌沒用到
這些局面的模擬，不會被雜訊灌水）。

    .venv-train/Scripts/python -m training.expert_train --data ex1 --init models/rl3_0960.pt --out models/ex1.pt
"""
from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from training.expert_data import DATA_DIR, load_shards
from training.model import load, masked_logits, save

MARGIN = 1.0
KL_WEIGHT = 1.0
HOLDOUT = 0.1


def labels(data: dict[str, np.ndarray], margin: float = MARGIN) -> np.ndarray:
    """每個局面要學的候選位置（0 = 網路第一名）。"""
    means, ses = data["means"], data["ses"]
    edge = means - means[:, :1] - margin * np.nan_to_num(ses, nan=0.0)
    edge = np.where(np.isnan(means), -np.inf, edge)
    edge[:, 0] = 0.0
    return edge.argmax(1)


def teacher_gain(choice: np.ndarray, data: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """網路挑的動作照老師的平均比第一名多幾台（挑到候選以外的動作不算），回傳（平均、2 倍標準誤、落在候選內的比例）。"""
    gains = []
    for row, action in enumerate(choice):
        where = np.nonzero(data["candidates"][row] == action)[0]
        if len(where):
            gains.append(data["means"][row, where[0]] - data["means"][row, 0])
    gains = np.array(gains)
    return float(gains.mean()), float(2 * gains.std() / np.sqrt(len(gains))), len(gains) / len(choice)


def predict(model, obs: np.ndarray, mask: np.ndarray, device: str, batch: int = 1024) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(obs), batch):
            o = torch.from_numpy(obs[start:start + batch].astype(np.float32)).to(device)
            m = torch.from_numpy(mask[start:start + batch]).to(device)
            out.append(masked_logits(model(o)[0], m).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train(model, data: dict[str, np.ndarray], target: np.ndarray, device: str, epochs: int = 4,
          lr: float = 3e-5, batch: int = 256, kl_weight: float = KL_WEIGHT, seed: int = 0) -> None:
    reference = deepcopy(model).eval()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    actions = data["candidates"][np.arange(len(target)), target].astype(np.int64)
    for epoch in range(epochs):
        order = rng.permutation(len(target))
        total = 0.0
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            obs = torch.from_numpy(data["obs"][index].astype(np.float32)).to(device)
            mask = torch.from_numpy(data["mask"][index]).to(device)
            logits = masked_logits(model(obs)[0], mask)
            with torch.no_grad():
                ref = F.softmax(masked_logits(reference(obs)[0], mask), 1)
            log_probs = F.log_softmax(logits, 1)
            ce = F.nll_loss(log_probs, torch.from_numpy(actions[index]).to(device))
            kl = (ref * (torch.log(ref.clamp_min(1e-12)) - log_probs)).sum(1).mean()
            loss = ce + kl_weight * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        print(f"第 {epoch + 1} 輪 損失 {total / len(order):.4f}", flush=True)
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="ex1", help="training/data/expert/<名稱>_*.npz")
    parser.add_argument("--init", default="models/rl3_0960.pt")
    parser.add_argument("--out", default="models/ex1.pt")
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--kl", type=float, default=KL_WEIGHT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(DATA_DIR.glob(f"{args.data}_*.npz"))
    paths = [path for path in paths if not path.name.endswith(".tmp.npz")]
    data = load_shards(paths)
    target = labels(data, args.margin)
    n = len(target)
    split = np.random.default_rng(0).random(n) < HOLDOUT
    train_data = {key: value[~split] for key, value in data.items()}
    test_data = {key: value[split] for key, value in data.items()}
    print(f"{n} 個局面（{len(paths)} 檔），老師要換牌 {np.mean(target != 0):.1%}；驗證 {split.sum()} 個", flush=True)
    model = load(args.init, device)
    before = predict(model, test_data["obs"], test_data["mask"], device)
    train(model, train_data, target[~split], device, args.epochs, args.lr, kl_weight=args.kl)
    after = predict(model, test_data["obs"], test_data["mask"], device)
    test_target = test_data["candidates"][np.arange(split.sum()), target[split]]
    for name, choice in (("原本", before), ("微調後", after)):
        gain, margin, inside = teacher_gain(choice, test_data)
        print(f"{name}：和老師一致 {np.mean(choice == test_target):.1%}、老師估計比第一名多 {gain:+.3f} ±{margin:.3f} 台"
              f"（{inside:.0%} 落在候選內）", flush=True)
    save(model, args.out)
    print(f"存到 {args.out}")


if __name__ == "__main__":
    main()
