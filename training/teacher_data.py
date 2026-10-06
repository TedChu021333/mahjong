"""產生監督學習資料：讓老師 AI 在模擬器四家互打，記下每個要做判斷的決定。

每筆：輸入平面（training.encoding，含模仿實戰辨識缺陷的雜訊）、合法動作遮罩、老師選的動作、
這位玩家這局扣抽成後的輸贏（台，價值頭的目標）。只有兩個以上合法動作時才記（只能打摸進的牌、
只能摸牌不需要判斷）。換三張與摸牌不交給網路。

    python -m training.teacher_data --games 20000 --workers 2 --teacher rule --out training/data/rule
資料分成多個 .npz 檔（每個工作程序每 CHUNK_GAMES 局一檔），np.savez_compressed 壓縮。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from random import Random

import numpy as np

from game.rules import legal_actions
from game.simulator import after_rake, payments, play_game
from training.actions import ACTION_COUNT, action_index, handled_by_network, legal_mask
from training.encoding import CHANNELS, LIVE_NOISE, Noise, encode

CHUNK_GAMES = 500


def teacher_policy(name: str):
    if name == "rule":
        from agent.rule_agent import choose_rule_action_for_player
        return choose_rule_action_for_player
    if name == "ev":
        from agent.ev_agent import choose_ev_action_for_player
        return choose_ev_action_for_player
    raise ValueError(f"未知的老師：{name}")


@dataclass
class Recorder:
    """包住老師的 policy，記下每個要判斷的決定（之後補上這局的輸贏）。"""
    policy: object
    noise: Noise
    rng: Random
    obs: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    players: list = field(default_factory=list)

    def __call__(self, state, player, rng):
        action = self.policy(state, player, rng)
        if action is None or not handled_by_network(action):
            return action
        mask, _ = legal_mask(legal_actions(state, player))
        if mask.sum() >= 2:
            self.obs.append(encode(state, player, self.noise, self.rng).astype(np.float16))
            self.masks.append(mask)
            self.actions.append(action_index(action))
            self.players.append(player)
        return action


def play_chunk(args: tuple[str, int, int, str, Noise]) -> tuple[str, int, int]:
    teacher, seed, games, out, noise = args
    recorder = Recorder(teacher_policy(teacher), noise, Random(seed + 1))
    values = []
    for index in range(games):
        start = len(recorder.actions)
        rng = Random(seed * 1_000_003 + index)
        result = play_game(rng, player_policy=recorder, swap=True, dealer=rng.randrange(4))
        net = [after_rake(amount) for amount in payments(result)]
        values += [net[player] for player in recorder.players[start:]]
    path = Path(f"{out}_{seed:05d}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.stack(recorder.obs) if recorder.obs else np.zeros((0, CHANNELS, 34), np.float16),
        mask=np.stack(recorder.masks) if recorder.masks else np.zeros((0, ACTION_COUNT), bool),
        action=np.asarray(recorder.actions, dtype=np.int16),
        value=np.asarray(values, dtype=np.float32))
    return str(path), games, len(recorder.actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="產生監督學習資料")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--teacher", default="rule", choices=("rule", "ev"))
    parser.add_argument("--out", default="training/data/rule")
    parser.add_argument("--seed", type=int, default=0, help="第一個檔的編號（續產生時接著編）")
    parser.add_argument("--no-noise", action="store_true", help="不模仿實戰辨識缺陷")
    args = parser.parse_args()
    noise = Noise() if args.no_noise else LIVE_NOISE
    chunks = [(args.teacher, args.seed + index, min(CHUNK_GAMES, args.games - index * CHUNK_GAMES),
               args.out, noise)
              for index in range((args.games + CHUNK_GAMES - 1) // CHUNK_GAMES)]
    started, total_games, total_samples = time.time(), 0, 0
    with Pool(args.workers) as pool:
        for path, games, samples in pool.imap_unordered(play_chunk, chunks):
            total_games += games
            total_samples += samples
            print(f"{path}：{games} 局 {samples} 筆（累計 {total_games} 局 {total_samples} 筆，"
                  f"{time.time() - started:.0f} 秒）", flush=True)


if __name__ == "__main__":
    main()
