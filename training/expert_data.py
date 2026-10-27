"""專家迭代教材：網路自己打完的模擬當老師，標出每個打牌局面「網路前 3 名裡哪張最好」。

training.net_probe（10/28）：網路第 2、3 名比第一名好超過 1 倍標準誤才換，換 28% 的局面，換一批模擬驗證每個決定
多 +0.058±0.044 台。這裡大量產生這種教材給 training.expert_train 微調網路。

每個程序：自己用網路四家互打抽局面（training.net_probe.sample_points）、補完後每張候選用同一副牌模擬 rollouts 次
（四家網路、取最高分），存每張候選的平均與相對第一名的標準誤。網路的輸入照自我對打加上實戰辨識雜訊（LIVE_NOISE）。
結果每個工作包存一個檔（training/data/expert/<名稱>_<序號>.npz），中斷後重跑會跳過已完成的。

    .venv-train/Scripts/python -m training.expert_data --points 25000 --rollouts 200 --actors 12 --name ex1
"""
from __future__ import annotations

import argparse
import time
from math import sqrt
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import mean, pstdev

import numpy as np
import torch

import training.net_probe as net_probe
from training.encoding import LIVE_NOISE, encode
from training.actions import legal_mask
from game.rules import ActionType, legal_actions

DATA_DIR = Path("training/data/expert")


def summarize(point, rows: list[list[float]], rng: Random) -> dict:
    """一個局面的教材：網路輸入、合法動作、候選動作編號、各候選平均輸贏與相對第一名的標準誤。"""
    n = len(rows)
    means = [mean(row[c] for row in rows) for c in range(len(point.candidates))]
    ses = [0.0] + [pstdev([row[c] - row[0] for row in rows]) / sqrt(n) for c in range(1, len(point.candidates))]
    actions = legal_actions(point.state, point.player)
    mask, lookup = legal_mask(actions)
    index_of = {action.tile: index for index, action in lookup.items() if action.kind == ActionType.DISCARD}
    return {"obs": encode(point.state, point.player, LIVE_NOISE, rng).astype(np.float16), "mask": mask,
            "candidates": np.array([index_of[tile] for tile in point.candidates], np.int16),
            "rule": np.array([kind.endswith("rule") for kind in point.kinds]),
            "means": np.array(means, np.float32), "ses": np.array(ses, np.float32)}


def run_task(task: tuple[int, int, int, str]) -> int:
    """一個工作包：抽 points 個局面、模擬、存檔；回傳局面數。"""
    seed, points, rollouts, path = task
    if Path(path).exists():
        return 0
    sampled = net_probe.sample_points(net_probe._model, net_probe._device, points, seed)
    matrices = net_probe.evaluate((sampled, rollouts, seed * 7919, True))
    rng = Random(seed)
    items = [summarize(point, rows, rng) for point, rows in zip(sampled, matrices)]
    width = max(len(item["candidates"]) for item in items)

    def pad(key, fill, dtype):
        return np.array([np.pad(item[key], (0, width - len(item[key])), constant_values=fill)
                         for item in items], dtype)

    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, obs=np.stack([item["obs"] for item in items]),
                        mask=np.stack([item["mask"] for item in items]),
                        candidates=pad("candidates", -1, np.int16), rule=pad("rule", False, bool),
                        means=pad("means", np.nan, np.float32), ses=pad("ses", np.nan, np.float32))
    tmp.replace(path)
    return len(items)


def load_shards(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [dict(np.load(path)) for path in paths]
    width = max(part["candidates"].shape[1] for part in parts)
    fills = {"candidates": -1, "rule": False, "means": np.nan, "ses": np.nan}
    out = {}
    for key in ("obs", "mask", "candidates", "rule", "means", "ses"):
        arrays = []
        for part in parts:
            array = part[key]
            if key in fills and array.shape[1] < width:
                array = np.pad(array, ((0, 0), (0, width - array.shape[1])), constant_values=fills[key])
            arrays.append(array)
        out[key] = np.concatenate(arrays)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/rl3_0960.pt")
    parser.add_argument("--name", default="ex1")
    parser.add_argument("--points", type=int, default=25000)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--actors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = [(args.seed + i, args.per_task, args.rollouts, str(DATA_DIR / f"{args.name}_{i:05d}.npz"))
             for i in range(args.points // args.per_task)]
    started, done = time.time(), 0
    with get_context("spawn").Pool(args.actors, initializer=net_probe._init,
                                   initargs=(args.model, device)) as pool:
        for count in pool.imap_unordered(run_task, tasks):
            done += count
            if count:
                rate = done / (time.time() - started)
                left = (len(tasks) * args.per_task - done) / max(rate, 1e-9)
                print(f"{done} 個局面，{time.time() - started:.0f} 秒，每小時 {rate * 3600:.0f} 個，"
                      f"約剩 {left / 3600:.1f} 小時", flush=True)


if __name__ == "__main__":
    main()
