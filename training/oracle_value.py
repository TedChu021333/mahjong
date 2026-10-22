"""實驗：看得到全部資訊的價值頭（Suphx 的 oracle 做法）能不能把輸贏的運氣扣得更乾淨。

    .venv-train/Scripts/python -m training.oracle_value --games 6000 --workers 3

背景（10/21）：PPO 的優勢 = 輸贏 − 基準。用價值頭（只看自己看得到的）當基準，殘差約 7 台；
同一副牌打兩次（--paired）兩次相差的標準差也有 7.4～9 台——牌局一分岔，之後的摸牌完全不同，抵銷不了。
這裡用規則式 AI 四家互打收集決定點，訓練兩個一樣大的價值網路：
  公開：只看 encode()（出牌網路看到的）；
  全知：再加上三家暗手、牌牆剩什麼、接下來要摸的牌（hidden_planes）。
比較兩者在沒看過的牌局上的殘差（台）。全知的殘差小很多，才值得把它接進訓練當基準（出牌網路仍只看公開資訊）。
"""
from __future__ import annotations

import argparse
import time
from random import Random

import numpy as np
import torch
from torch import nn

from agent.rule_agent import choose_rule_action_for_player
from game.rules import GameState, legal_actions
from game.simulator import after_rake, payments, play_game
from game.tiles import NUM_TILE_TYPES
from training.encoding import CHANNELS, encode, relative
from training.model import VALUE_SCALE

NEXT_DRAWS = 16
"""全知平面列出接下來幾張要摸的牌（牌牆從尾端摸）。"""
HIDDEN_CHANNELS = 3 * 4 + 4 + NEXT_DRAWS
RECORD_SHARE = 0.2
"""每個決定點記下來的機率（同一局前後的決定高度相關，不必全記）。"""


def hidden_planes(state: GameState, player: int) -> np.ndarray:
    """player 看不到的資訊：三家暗手（依相對座位、張數 ≥1..4）、牌牆剩下的牌（張數 ≥1..4）、接下來要摸的牌。"""
    planes = np.zeros((HIDDEN_CHANNELS, NUM_TILE_TYPES), dtype=np.float32)
    for seat in range(4):
        rel = relative(seat, player)
        if rel == 0:
            continue
        hand = np.asarray(state.players[seat].hand)
        for level in range(4):
            planes[(rel - 1) * 4 + level] = hand >= level + 1
    drawable = state.wall[state.reserve:] if state.reserve else list(state.wall)
    wall = np.zeros(NUM_TILE_TYPES)
    for tile in drawable:
        if tile < NUM_TILE_TYPES:
            wall[tile] += 1
    for level in range(4):
        planes[12 + level] = wall >= level + 1
    for index, tile in enumerate(reversed(drawable[-NEXT_DRAWS:])):
        if tile < NUM_TILE_TYPES:  # 花牌摸到會補，先不標
            planes[16 + index, tile] = 1
    return planes


def play_and_record(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """規則式四家打一局（含換三張），回傳 (公開平面, 全知平面, 輸贏 / VALUE_SCALE)。"""
    rng = Random(seed)
    record_rng = Random(seed + 7)
    rows: list[tuple[np.ndarray, np.ndarray, int]] = []

    def policy(state, player, game_rng):
        if len(legal_actions(state, player)) > 1 and record_rng.random() < RECORD_SHARE:
            rows.append((encode(state, player).astype(np.float16),
                         hidden_planes(state, player).astype(np.float16), player))
        return choose_rule_action_for_player(state, player, game_rng)

    result = play_game(rng, player_policy=policy, swap=True, dealer=rng.randrange(4))
    net = payments(result)
    if not rows:
        empty = np.zeros((0, CHANNELS, NUM_TILE_TYPES), np.float16)
        return empty, np.zeros((0, HIDDEN_CHANNELS, NUM_TILE_TYPES), np.float16), np.zeros(0, np.float32)
    public = np.stack([row[0] for row in rows])
    hidden = np.stack([row[1] for row in rows])
    target = np.array([after_rake(net[row[2]]) / VALUE_SCALE for row in rows], np.float32)
    return public, hidden, target


def _play_many(seeds: list[int]) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    return [(seed, *play_and_record(seed)) for seed in seeds]


def collect(games: int, workers: int, first_seed: int = 5_000_000):
    seeds = list(range(first_seed, first_seed + games))
    chunks = [seeds[i::workers * 8] for i in range(workers * 8)]
    if workers > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_play_many, chunks)
    else:
        parts = [_play_many(chunk) for chunk in chunks]
    played = sorted((item for part in parts for item in part), key=lambda item: item[0])
    game_ids = np.concatenate([np.full(len(item[3]), item[0]) for item in played])
    public = np.concatenate([item[1] for item in played])
    hidden = np.concatenate([item[2] for item in played])
    target = np.concatenate([item[3] for item in played])
    return game_ids, public, hidden, target


class ValueNet(nn.Module):
    """和 MahjongNet 同樣的 1D 殘差卷積，只有價值頭。"""

    def __init__(self, channels: int, width: int = 64, blocks: int = 6) -> None:
        super().__init__()
        from training.model import Block

        self.stem = nn.Sequential(nn.Conv1d(channels, width, 3, padding=1, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU())
        self.body = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * 34, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x))).squeeze(1)


def rmse(model, x, y, rows, device) -> float:
    errors = []
    model.eval()
    with torch.no_grad():
        rows_t = torch.from_numpy(rows)
        for start in range(0, len(rows_t), 4096):
            index = rows_t[start:start + 4096]
            errors.append((model(x[index].float().to(device)) - y[index].to(device)).cpu().numpy())
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)) * VALUE_SCALE)


def fit(inputs: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray, device: str,
        epochs: int, batch: int = 512, seed: int = 0, lr: float = 3e-4, label: str = "") -> float:
    """訓練價值網路，回傳沒看過的牌局上的殘差（台，RMSE）；每輪印訓練集與測試集的殘差。"""
    torch.manual_seed(seed)
    model = ValueNet(inputs.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    probe = train[:: max(1, len(train) // len(test))][: len(test)]
    x = torch.from_numpy(inputs)
    y = torch.from_numpy(target)
    train_t = torch.from_numpy(train)
    for epoch in range(epochs):
        model.train()
        order = train_t[torch.randperm(len(train_t))]
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            loss = ((model(x[index].float().to(device)) - y[index].to(device)) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"  {label} 第 {epoch + 1} 輪：訓練集殘差 {rmse(model, x, y, probe, device):.2f} 台、"
              f"測試集 {rmse(model, x, y, test, device):.2f} 台", flush=True)
    return rmse(model, x, y, test, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="全知價值頭能扣掉多少運氣")
    parser.add_argument("--games", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cache", type=str, default="training/data/oracle_data.npz",
                        help="收集的資料存在這裡；檔案已存在就直接讀")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    from pathlib import Path

    cache = Path(args.cache)
    if cache.exists():
        data = np.load(cache)
        game_ids, public, hidden, target = data["game_ids"], data["public"], data["hidden"], data["target"]
        print(f"讀取 {cache}：{len(target)} 個決定點", flush=True)
    else:
        game_ids, public, hidden, target = collect(args.games, args.workers)
        np.savez(cache, game_ids=game_ids, public=public, hidden=hidden, target=target)
        print(f"收集 {args.games} 局、{len(target)} 個決定點（{time.time() - started:.0f} 秒）", flush=True)
    games = np.unique(game_ids)
    test_games = set(games[: len(games) // 10].tolist())
    is_test = np.array([gid in test_games for gid in game_ids])
    train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    spread = float(np.sqrt(np.mean((target[test] - target[train].mean()) ** 2)) * VALUE_SCALE)
    print(f"不預測（用平均當基準）：殘差 {spread:.2f} 台", flush=True)
    public_error = fit(public, target, train, test, device, args.epochs, lr=args.lr, label="公開")
    print(f"公開資訊的價值網路：殘差 {public_error:.2f} 台", flush=True)
    full = np.concatenate([public, hidden], axis=1)
    oracle_error = fit(full, target, train, test, device, args.epochs, lr=args.lr, label="全知")
    print(f"全知的價值網路：殘差 {oracle_error:.2f} 台", flush=True)
    explained = 1 - (oracle_error / spread) ** 2
    print(f"全知能解釋 {explained:.0%} 的輸贏變異（公開 {1 - (public_error / spread) ** 2:.0%}）", flush=True)


if __name__ == "__main__":
    main()
