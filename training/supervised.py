"""監督學習：網路模仿老師 AI 的選擇（策略頭），同時學預測這局輸贏（價值頭）。

    .venv-train/Scripts/python -m training.supervised --data training/data/rule --out models/sl_rule.pt

資料檔逐批讀入（全部載入要十幾 GB），每次混 BUFFER_FILES 個檔打散；最後 VALIDATION_FILES 個檔
不訓練，用來看「和老師選一樣」的比例（top-1 一致率）與價值誤差。
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.model import VALUE_SCALE, MahjongNet, masked_logits, save

BUFFER_FILES = 6
VALIDATION_FILES = 2
VALUE_WEIGHT = 0.5


def load_files(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [np.load(path) for path in paths]
    return {key: np.concatenate([part[key] for part in parts]) for key in ("obs", "mask", "action", "value")}


def batches(data: dict[str, np.ndarray], size: int, shuffle: bool, device: str):
    order = np.random.permutation(len(data["action"])) if shuffle else np.arange(len(data["action"]))
    for start in range(0, len(order), size):
        index = order[start:start + size]
        yield (torch.from_numpy(data["obs"][index].astype(np.float32)).to(device),
               torch.from_numpy(data["mask"][index]).to(device),
               torch.from_numpy(data["action"][index].astype(np.int64)).to(device),
               torch.from_numpy(data["value"][index]).to(device) / VALUE_SCALE)


def evaluate(model: MahjongNet, data: dict[str, np.ndarray], device: str) -> tuple[float, float]:
    model.eval()
    correct, total, value_error = 0, 0, 0.0
    with torch.no_grad():
        for obs, mask, action, value in batches(data, 2048, False, device):
            logits, predicted = model(obs)
            correct += (masked_logits(logits, mask).argmax(1) == action).sum().item()
            value_error += ((predicted - value) ** 2).sum().item()
            total += len(action)
    model.train()
    return correct / total, value_error / total


def train(files: list[Path], out: Path, epochs: int, batch_size: int, lr: float,
          width: int, blocks: int, device: str) -> None:
    validation, training = files[-VALIDATION_FILES:], files[:-VALIDATION_FILES]
    held_out = load_files(validation)
    model = MahjongNet(width, blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = sum(len(np.load(path)["action"]) for path in training) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=epochs * steps_per_epoch + 10)
    policy_loss, value_loss = nn.CrossEntropyLoss(), nn.MSELoss()
    started = time.time()
    print(f"訓練 {len(training)} 檔、驗證 {len(validation)} 檔（{len(held_out['action'])} 筆），"
          f"每輪約 {steps_per_epoch} 步", flush=True)
    for epoch in range(epochs):
        random.shuffle(training)
        seen, running = 0, 0.0
        for group in range(0, len(training), BUFFER_FILES):
            data = load_files(training[group:group + BUFFER_FILES])
            for obs, mask, action, value in batches(data, batch_size, True, device):
                logits, predicted = model(obs)
                loss = policy_loss(masked_logits(logits, mask), action) \
                    + VALUE_WEIGHT * value_loss(predicted, value)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()
                running += loss.item() * len(action)
                seen += len(action)
        accuracy, value_mse = evaluate(model, held_out, device)
        print(f"第 {epoch + 1} 輪：訓練損失 {running / seen:.3f}，驗證一致率 {accuracy:.1%}，"
              f"價值均方誤差 {value_mse:.3f}（{time.time() - started:.0f} 秒）", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        save(model, out, epoch=epoch + 1, accuracy=accuracy, value_mse=value_mse)


def main() -> None:
    parser = argparse.ArgumentParser(description="監督學習：模仿老師 AI")
    parser.add_argument("--data", default="training/data/rule", help="資料檔前綴")
    parser.add_argument("--out", type=Path, default=Path("models/sl_rule.pt"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="只用前幾個檔（測試用）")
    args = parser.parse_args()
    prefix = Path(args.data)
    files = sorted(prefix.parent.glob(prefix.name + "_*.npz"))
    if args.limit:
        files = files[:args.limit]
    if len(files) <= VALIDATION_FILES:
        raise SystemExit(f"資料檔太少：{len(files)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"裝置 {device}", flush=True)
    train(files, args.out, args.epochs, args.batch, args.lr, args.width, args.blocks, device)


if __name__ == "__main__":
    main()
