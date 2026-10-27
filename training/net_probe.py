"""專家迭代可行性（第二版）：模擬裡四家都用網路打完，看這種老師有沒有用。

第 0 階段（training.teacher_probe）用規則式打完的模擬分不出網路比規則式好，那種老師最多教回規則式。
這裡改成網路自己打完（每次取分數最高的動作，和實戰一樣），檢查兩件事：
1. 認不認得出網路比規則式好：網路第一名 vs 規則式那張（不同時），應該是網路那張較好。
2. 找不找得到改進：網路第 2、3 名顯著比第一名好的局面比例，要明顯高於雜訊（單邊 2 倍標準誤約 2.3%）。

局面從網路四家互打抽（沒宣告、要打牌）；補完與分家牌牆沿用 training.teacher_probe。每個程序一個 GPU 批次執行緒、
THREADS 條執行緒各跑一個模擬（Python 規則引擎受 GIL 限制，靠多程序撐速度）。

    .venv-train/Scripts/python -m training.net_probe --points 400 --rollouts 400 --actors 12 --out result/net_probe.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from multiprocessing import get_context
from random import Random
from statistics import mean, pstdev

import numpy as np
import torch

from agent.rule_agent import choose_rule_action_for_player
from game.rules import Action, ActionType, GameState, Phase, legal_actions
from game.simulator import after_rake, payments, play_game
from game.tiles import tile_name
from training.actions import legal_mask
from training.encoding import encode
from training.model import load, masked_logits
from training.teacher_probe import determinize

THREADS = 64
BATCH_WAIT = 0.002
TOP = 3
"""網路分數前幾名當候選。"""


@dataclass
class NetPoint:
    state: GameState
    player: int
    candidates: list[int]
    kinds: list[str]
    """每個候選是什麼：net1、net2、net3、rule。第一個一定是 net1。"""


class GreedyBatcher:
    """收集各執行緒的推論請求湊成一批送網路，回傳分數最高的合法動作編號。"""

    def __init__(self, model, device: str, max_batch: int = THREADS) -> None:
        self.model, self.device, self.max_batch = model, device, max_batch
        self.requests: queue.Queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def ask(self, obs: np.ndarray, mask: np.ndarray) -> int:
        slot = [obs, mask, threading.Event(), None]
        self.requests.put(slot)
        slot[2].wait()
        return slot[3]

    def _loop(self) -> None:
        while self.running:
            try:
                batch = [self.requests.get(timeout=0.1)]
            except queue.Empty:
                continue
            deadline = time.perf_counter() + BATCH_WAIT
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.requests.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            obs = torch.from_numpy(np.stack([item[0] for item in batch])).to(self.device)
            mask = torch.from_numpy(np.stack([item[1] for item in batch])).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(obs)
                choices = masked_logits(logits, mask).argmax(1).cpu().numpy()
            for item, choice in zip(batch, choices):
                item[3] = int(choice)
                item[2].set()

    def close(self) -> None:
        self.running = False
        self.thread.join()


def network_policy(ask):
    """ask(obs, mask) → 動作編號；摸牌、換三張與只有一個選擇時不問網路。"""
    def policy(state: GameState, player: int, rng: Random) -> Action | None:
        actions = legal_actions(state, player)
        if not actions:
            return None
        mask, lookup = legal_mask(actions)
        if not lookup:
            return choose_rule_action_for_player(state, player, rng)
        if len(lookup) == 1:
            return next(iter(lookup.values()))
        return lookup[ask(encode(state, player), mask)]
    return policy


def sample_points(model, device: str, count: int, seed: int, rate: float = 0.05) -> list[NetPoint]:
    """網路四家互打，符合條件的打牌局面以 rate 機率記下：候選是網路前 TOP 名的打牌，加上規則式那張。"""
    rng = Random(seed)
    points: list[NetPoint] = []

    def scores(state, player):
        actions = legal_actions(state, player)
        mask, lookup = legal_mask(actions)
        obs = torch.from_numpy(encode(state, player)[None]).to(device)
        with torch.no_grad():
            logits = masked_logits(model(obs)[0], torch.from_numpy(mask[None]).to(device))[0].cpu().numpy()
        return logits, lookup

    def ask(obs, mask):
        with torch.no_grad():
            logits = model(torch.from_numpy(obs[None]).to(device))[0]
            return int(masked_logits(logits, torch.from_numpy(mask[None]).to(device))[0].argmax())

    base = network_policy(ask)

    def policy(state: GameState, player: int, game_rng: Random) -> Action | None:
        action = base(state, player, game_rng)
        if (len(points) < count and action is not None and action.kind == ActionType.DISCARD
                and state.phase == Phase.DISCARD and not state.declared[player] and rng.random() < rate):
            logits, lookup = scores(state, player)
            ranked = [index for index in np.argsort(-logits)
                      if index in lookup and lookup[index].kind == ActionType.DISCARD][:TOP]
            candidates = [lookup[index].tile for index in ranked]
            kinds = [f"net{rank + 1}" for rank in range(len(candidates))]
            rule = choose_rule_action_for_player(state, player, Random(0))
            if rule is not None and rule.kind == ActionType.DISCARD and rule.tile not in candidates:
                candidates.append(rule.tile)
                kinds.append("rule")
            elif rule is not None and rule.kind == ActionType.DISCARD:
                kinds[candidates.index(rule.tile)] += "+rule"
            if len(candidates) >= 2 and candidates[0] == action.tile:
                points.append(NetPoint(deepcopy(state), player, candidates, kinds))
        return action

    while len(points) < count:
        play_game(Random(rng.getrandbits(32)), player_policy=policy, dealer=rng.randrange(4), swap=True)
    return points


_model = None
_device = "cpu"


def _init(path: str, device: str) -> None:
    global _model, _device
    torch.set_num_threads(1)
    _device = device
    _model = load(path, device)


def evaluate(task: tuple[list[NetPoint], int, int, bool]) -> list[list[list[float]]]:
    """每個局面一個矩陣：每次模擬一列、各候選這家的輸贏（同一副補完的牌、同一個種子）。"""
    points, rollouts, seed, streams = task
    batcher = GreedyBatcher(_model, _device)
    jobs = []
    results = [[[0.0] * len(p.candidates) for _ in range(rollouts)] for p in points]
    for p_index, point in enumerate(points):
        rng = Random(seed + p_index)
        for r in range(rollouts):
            world = determinize(point.state, point.player, rng, streams)
            game_seed = rng.getrandbits(32)
            for c_index in range(len(point.candidates)):
                jobs.append((p_index, r, c_index, world, game_seed))
    lock = threading.Lock()
    pending = iter(jobs)
    continue_with = network_policy(batcher.ask)

    def worker() -> None:
        while True:
            with lock:
                job = next(pending, None)
            if job is None:
                return
            p_index, r, c_index, world, game_seed = job
            point = points[p_index]
            tile = point.candidates[c_index]
            first = [True]

            def policy(state, player, rng):
                if first[0] and player == point.player and state.phase == Phase.DISCARD:
                    first[0] = False
                    return Action(ActionType.DISCARD, tile)
                return continue_with(state, player, rng)

            result = play_game(Random(game_seed), player_policy=policy, start=deepcopy(world))
            results[p_index][r][c_index] = after_rake(payments(result)[point.player])

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    batcher.close()
    return results


def report(points: list[NetPoint], matrices: list[list[list[float]]]) -> str:
    by_kind: dict[str, list[tuple[float, float]]] = {}
    paired = []
    for point, rows in zip(points, matrices):
        n = len(rows)
        for c in range(1, len(point.candidates)):
            diff = [row[c] - row[0] for row in rows]
            sd = pstdev(diff)
            paired.append(sd)
            by_kind.setdefault(point.kinds[c].split("+")[0], []).append((mean(diff), sd / sqrt(n)))
    lines = [f"局面 {len(points)}、每張候選模擬 {len(matrices[0])} 次；逐次相減標準差 "
             f"{sqrt(mean([s * s for s in paired])):.2f} 台",
             "候選 − 網路第一名（台）："]
    for kind in ("rule", "net2", "net3"):
        pairs = by_kind.get(kind, [])
        if not pairs:
            continue
        means = [m for m, _ in pairs]
        avg = mean(means)
        spread = sqrt(sum((m - avg) ** 2 for m in means) / max(1, len(means) - 1) / len(means))
        better = mean([m > 2 * se for m, se in pairs if se > 0] or [0])
        worse = mean([m < -2 * se for m, se in pairs if se > 0] or [0])
        true_sd = sqrt(max(0.0, pstdev(means) ** 2 - mean([se * se for _, se in pairs])))
        lines.append(f"  {kind:5} {len(pairs):4} 個局面：平均 {avg:+.3f} ±{2 * spread:.3f}；"
                     f"顯著較好 {better:.1%}、顯著較差 {worse:.1%}（雜訊約各 2.3%）；真實差距標準差約 {true_sd:.2f}")
    lines.append("前一半模擬挑牌、後一半驗證（只換網路自己的第 2、3 名）：")
    for margin in (0.0, 1.0, 2.0):
        gain, switched = split_gain(points, matrices, margin)
        lines.append(f"  比第一名好超過 {margin:.0f} 倍標準誤才換：換了 {switched:.1%} 的局面、"
                     f"每個決定平均多 {mean(gain):+.3f} ±{2 * pstdev(gain) / sqrt(len(gain)):.3f} 台")
    return "\n".join(lines)


def split_gain(points: list[NetPoint], matrices: list[list[list[float]]],
               margin: float) -> tuple[list[float], float]:
    """每個局面用前一半模擬挑候選（網路第一名或第 2、3 名，比第一名好超過 margin 倍標準誤才換），
    用後一半算挑中的那張比第一名多幾台——不受挑選時的雜訊灌水。"""
    gains, switched = [], 0
    for point, rows in zip(points, matrices):
        half = len(rows) // 2
        first, second = rows[:half], rows[half:]
        choice, best = 0, 0.0
        for c in range(1, len(point.candidates)):
            if point.kinds[c].startswith("rule"):
                continue
            diff = [row[c] - row[0] for row in first]
            edge = mean(diff) - margin * pstdev(diff) / sqrt(half)
            if edge > best:
                choice, best = c, edge
        switched += choice != 0
        gains.append(mean(row[choice] - row[0] for row in second))
    return gains, switched / len(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/rl3_0960.pt")
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--rollouts", type=int, default=400)
    parser.add_argument("--actors", type=int, default=12)
    parser.add_argument("--per-task", type=int, default=5, help="每個工作包幾個局面")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--streams", action="store_true", default=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    points = sample_points(load(args.model, device), device, args.points, args.seed)
    print(f"抽到 {len(points)} 個局面（{time.time() - started:.0f} 秒）", flush=True)
    tasks = [(points[i:i + args.per_task], args.rollouts, args.seed * 1_000_003 + i, args.streams)
             for i in range(0, len(points), args.per_task)]
    matrices = []
    with get_context("spawn").Pool(args.actors, initializer=_init, initargs=(args.model, device)) as pool:
        for done, part in enumerate(pool.imap(evaluate, tasks), 1):
            matrices += part
            print(f"{len(matrices)}/{len(points)} 局面，{time.time() - started:.0f} 秒", flush=True)
            if done % 10 == 0:
                print(report(points[:len(matrices)], matrices), flush=True)
    print(report(points, matrices))
    if args.out:
        with open(args.out.replace(".csv", ".pkl"), "wb") as f:
            pickle.dump((points, matrices), f)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["player", "candidates", "kinds", "means"])
            for point, rows in zip(points, matrices):
                means = [mean(row[c] for row in rows) for c in range(len(point.candidates))]
                writer.writerow([point.player, " ".join(tile_name(t) for t in point.candidates),
                                 " ".join(point.kinds), " ".join(f"{m:.3f}" for m in means)])


if __name__ == "__main__":
    main()
