"""強化學習（PPO）：網路自己打，以每局扣抽成後的實際輸贏為獎勵。

    .venv-train/Scripts/python -m training.selfplay --init models/sl_rule.pt --out models/rl.pt

每一輪：同時開 GAMES_IN_FLIGHT 局（每局一條執行緒跑原本的 game.simulator.play_game，規則與
模擬器完全共用），輪到網路做決定時把請求丟進佇列，由批次執行緒湊成一批一起送 GPU。
每局每個座位有 rule_share 的機率換成規則式 AI（不訓練），避免只學會打贏自己。
打完 games_per_iteration 局後，用這些決定做幾回 PPO 更新：
  優勢 = 這局實際輸贏 − 價值頭預測（麻將運氣成分大，用預測當基準扣掉大部分運氣）。
--paired（10/21 起）：同一副牌打兩次（牌牆、對手、對手的隨機數都一樣，只有網路的抽樣不同），
  優勢 = 這次的輸贏 − 另一次的輸贏：發牌好壞兩次一樣、相減抵銷，剩下的差別來自網路自己的選擇。
  10/20 rl7 用價值頭當基準練 840 輪沒有進步（只學到更保守），判斷是輸贏雜訊蓋過了打法的差別。
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

import numpy as np
import torch

from agent.rule_agent import choose_rule_action_for_player
from game.rules import legal_actions
from game.simulator import after_rake, payments, play_game
from training.actions import legal_mask
from training.encoding import LIVE_NOISE, Noise, encode
from training.model import VALUE_SCALE, MahjongNet, load, masked_logits, save

LEAGUE_SIZE = 6
GAMES_IN_FLIGHT = 32
"""同時開幾局（每局一條執行緒）：實測 8 → 7 局/秒、16 → 18、32 → 22（GIL 限制，再多沒幫助）。"""
BATCH_WAIT = 0.002
"""湊一批最多等幾秒。"""


@dataclass
class Step:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    logprob: float
    value: float


class Batcher:
    """收集各局的推論請求，湊成一批送網路；依 softmax 抽樣動作（探索）。"""

    def __init__(self, model: MahjongNet, device: str, max_batch: int = GAMES_IN_FLIGHT) -> None:
        self.model, self.device, self.max_batch = model, device, max_batch
        self.requests: queue.Queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def ask(self, obs: np.ndarray, mask: np.ndarray, rng: Random) -> tuple[int, float, float]:
        done = threading.Event()
        slot: list = [obs, mask, rng.random(), done, None]
        self.requests.put(slot)
        done.wait()
        return slot[4]

    def _loop(self) -> None:
        while self.running:
            try:
                first = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.perf_counter() + BATCH_WAIT
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.requests.get(timeout=max(0.0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            obs = torch.from_numpy(np.stack([item[0] for item in batch])).to(self.device)
            mask = torch.from_numpy(np.stack([item[1] for item in batch])).to(self.device)
            with torch.no_grad():
                logits, values = self.model(obs)
                probs = torch.softmax(masked_logits(logits, mask), 1).cpu().numpy()
            values = values.cpu().numpy()
            for row, item in enumerate(batch):
                cumulative = np.cumsum(probs[row])
                action = int(min(np.searchsorted(cumulative, item[2] * cumulative[-1]),
                                 len(cumulative) - 1))
                while not item[1][action]:  # 浮點誤差落在不合法的動作上
                    action -= 1
                item[4] = (action, float(np.log(max(probs[row, action], 1e-12))), float(values[row]))
                item[3].set()

    def close(self) -> None:
        self.running = False
        self.thread.join()


@dataclass
class GameRecord:
    learners: list[int]
    steps: dict[int, list[Step]] = field(default_factory=dict)
    reward: dict[int, float] = field(default_factory=dict)
    winner: int | None = None
    discarder: int | None = None
    rule_seats: list[int] = field(default_factory=list)
    """規則式 AI 的座位（其餘非網路座位是對手池裡過去的自己）。"""
    baseline: dict[int, float] | None = None
    """同一副牌另一次的輸贏（--paired）；有的話優勢以它為基準，不用價值頭預測。"""


def play_one(seed: int, batcher: Batcher, rule_share: float, noise: Noise,
             opponent: Batcher | None = None, pool_share: float = 0.0, copy: int = 0) -> GameRecord:
    """每個座位依機率當規則式（rule_share）、對手池裡過去的自己（pool_share，需要 opponent）或
    正在訓練的網路；至少一個座位是網路，只有它的決定拿來訓練。
    同一個 seed 的牌牆、座位、對手用的隨機數都一樣；網路的抽樣另用 (seed, copy) 的隨機數，
    所以 copy 不同就是「同一副牌、網路重打一次」。"""
    rng = Random(seed)
    learner_rng = Random(f"{seed}:{copy}")
    roles = []
    for _ in range(4):
        roll = rng.random()
        if roll < rule_share:
            roles.append("rule")
        elif opponent is not None and roll < rule_share + pool_share:
            roles.append("pool")
        else:
            roles.append("learner")
    if "learner" not in roles:
        roles[rng.randrange(4)] = "learner"
    learners = [seat for seat, role in enumerate(roles) if role == "learner"]
    record = GameRecord(learners, {seat: [] for seat in learners},
                        rule_seats=[seat for seat, role in enumerate(roles) if role == "rule"])
    noise_rng = Random(seed + 1)

    def policy(state, player, game_rng):
        if roles[player] == "rule":
            return choose_rule_action_for_player(state, player, game_rng)
        actions = legal_actions(state, player)
        if not actions:
            return None
        mask, lookup = legal_mask(actions)
        if not lookup:
            return choose_rule_action_for_player(state, player, game_rng)
        if len(lookup) == 1:
            return next(iter(lookup.values()))
        obs = encode(state, player, noise, noise_rng)
        if roles[player] == "pool":
            return lookup[opponent.ask(obs, mask, game_rng)[0]]
        action, logprob, value = batcher.ask(obs, mask, learner_rng)
        record.steps[player].append(Step(obs.astype(np.float16), mask, action, logprob, value))
        return lookup[action]

    result = play_game(rng, player_policy=policy, swap=True, dealer=rng.randrange(4))
    net = payments(result)
    record.reward = {seat: after_rake(net[seat]) / VALUE_SCALE for seat in learners}
    record.winner, record.discarder = result.winner, result.discarder
    return record


def play_pair(seed: int, batcher: Batcher, rule_share: float, noise: Noise,
              opponent: Batcher | None = None, pool_share: float = 0.0) -> list[GameRecord]:
    """同一副牌打兩次，各自以另一次的輸贏為基準。"""
    first = play_one(seed, batcher, rule_share, noise, opponent, pool_share, copy=0)
    second = play_one(seed, batcher, rule_share, noise, opponent, pool_share, copy=1)
    first.baseline, second.baseline = dict(second.reward), dict(first.reward)
    return [first, second]


def collect(batcher: Batcher, games: int, first_seed: int, rule_share: float,
            noise: Noise, opponent: Batcher | None = None,
            pool_share: float = 0.0, paired: bool = False) -> list[GameRecord]:
    """打 games 局（paired 時是 games // 2 副牌、每副打兩次）。"""
    records: list[GameRecord] = []
    lock = threading.Lock()
    deals = max(1, games // 2) if paired else games
    seeds = iter(range(first_seed, first_seed + deals))

    def worker() -> None:
        while True:
            with lock:
                seed = next(seeds, None)
            if seed is None:
                return
            if paired:
                played = play_pair(seed, batcher, rule_share, noise, opponent, pool_share)
            else:
                played = [play_one(seed, batcher, rule_share, noise, opponent, pool_share)]
            with lock:
                records.extend(played)

    threads = [threading.Thread(target=worker) for _ in range(min(GAMES_IN_FLIGHT, deals))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records


def ppo_update(model: MahjongNet, optimizer, records: list[GameRecord], device: str,
               epochs: int = 3, batch_size: int = 1024, clip: float = 0.2,
               value_weight: float = 0.5, entropy_weight: float = 0.0) -> dict[str, float]:
    obs, masks, actions, old_logprobs, returns, advantages = [], [], [], [], [], []
    for record in records:
        for seat, steps in record.steps.items():
            for step in steps:
                obs.append(step.obs)
                masks.append(step.mask)
                actions.append(step.action)
                old_logprobs.append(step.logprob)
                returns.append(record.reward[seat])
                base = record.baseline[seat] if record.baseline is not None else step.value
                advantages.append(record.reward[seat] - base)
    if not actions:
        return {}
    obs_t = torch.from_numpy(np.stack(obs).astype(np.float32))
    mask_t = torch.from_numpy(np.stack(masks))
    action_t = torch.tensor(actions)
    old_t = torch.tensor(old_logprobs, dtype=torch.float32)
    return_t = torch.tensor(returns, dtype=torch.float32)
    adv_t = torch.tensor(advantages, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
    model.train()
    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "batches": 0}
    for _ in range(epochs):
        order = torch.randperm(len(action_t))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            logits, value = model(obs_t[index].to(device))
            logits = masked_logits(logits, mask_t[index].to(device))
            log_probs = torch.log_softmax(logits, 1)
            new = log_probs.gather(1, action_t[index].to(device)[:, None]).squeeze(1)
            ratio = torch.exp(new - old_t[index].to(device))
            adv = adv_t[index].to(device)
            policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            value_loss = ((value - return_t[index].to(device)) ** 2).mean()
            probs = log_probs.exp()
            entropy = -(probs * log_probs.clamp(min=-30)).sum(1).mean()
            loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            stats["policy"] += policy_loss.item()
            stats["value"] += value_loss.item()
            stats["entropy"] += entropy.item()
            # 新舊策略差多少（近似 KL）：太大代表一次更新走太遠
            stats["kl"] += (old_t[index].to(device) - new).mean().item()
            stats["batches"] += 1
    model.eval()
    batches = max(1, stats.pop("batches"))
    return {key: value / batches for key, value in stats.items()}


def win_gap(records: list[GameRecord]) -> float:
    """網路座位胡牌率 − 規則式座位胡牌率（沒有規則式座位時為網路胡牌率）。"""
    seats = sum(len(record.learners) for record in records)
    rule_seats = sum(len(record.rule_seats) for record in records)
    wins = sum(record.winner in record.learners for record in records) / max(1, seats)
    rule_wins = sum(record.winner in record.rule_seats for record in records) / max(1, rule_seats)
    return wins - rule_wins if rule_seats else wins


class Guard:
    """訓練變差就停：10/7 熵權重 0.01 讓網路越練越隨機（熵 0.3 → 1.0、胡牌率 23% → 17%）。
    每 WINDOW 輪比較一次：熵比第一個區間高 ENTROPY_DRIFT 以上，或網路對規則式的胡牌率差距
    比第一個區間掉 GAP_DROP 以上，就停止。"""
    WINDOW = 50
    ENTROPY_DRIFT = 0.15
    GAP_DROP = 0.02

    def __init__(self) -> None:
        self.entropy: list[float] = []
        self.gap: list[float] = []
        self.baseline: tuple[float, float] | None = None

    def update(self, entropy: float, gap: float) -> str | None:
        self.entropy.append(entropy)
        self.gap.append(gap)
        if len(self.entropy) < self.WINDOW:
            return None
        window = (float(np.mean(self.entropy)), float(np.mean(self.gap)))
        self.entropy.clear()
        self.gap.clear()
        if self.baseline is None:
            self.baseline = window
            return None
        if window[0] - self.baseline[0] > self.ENTROPY_DRIFT:
            return f"熵 {self.baseline[0]:.2f} → {window[0]:.2f}，網路越來越隨機"
        if self.baseline[1] - window[1] > self.GAP_DROP:
            return f"胡牌率差距 {self.baseline[1]:+.1%} → {window[1]:+.1%}，網路變弱"
        return None


def summarize(records: list[GameRecord]) -> str:
    rewards = [record.reward[seat] * VALUE_SCALE for record in records for seat in record.learners]
    seats = sum(len(record.learners) for record in records)
    wins = sum(record.winner in record.learners for record in records)
    deal_ins = sum(record.discarder in record.learners for record in records)
    rule_seats = sum(len(record.rule_seats) for record in records)
    rule_wins = sum(record.winner in record.rule_seats for record in records)
    rule_deal_ins = sum(record.discarder in record.rule_seats for record in records)
    text = (f"網路每局平均 {np.mean(rewards):+.2f} 台（扣抽成），胡牌率 {wins / seats:.1%}、"
            f"放槍率 {deal_ins / seats:.1%}")
    if rule_seats:
        text += f"；規則式 胡牌率 {rule_wins / rule_seats:.1%}、放槍率 {rule_deal_ins / rule_seats:.1%}"
    gaps = [(record.reward[seat] - record.baseline[seat]) * VALUE_SCALE
            for record in records if record.baseline is not None for seat in record.learners]
    if gaps:
        # 同一副牌兩次的差距比單局輸贏的波動小多少＝雜訊降了多少
        text += f"；同牌兩次相差標準差 {np.std(gaps):.2f} 台（單局輸贏 {np.std(rewards):.2f}）"
    return text


_actor_model: MahjongNet | None = None
_actor_opponent: MahjongNet | None = None
_actor_device = "cpu"


def _actor_init(config: dict, device: str) -> None:
    """對打程序啟動時建立自己的網路（每個程序各自用 GPU 推論）。"""
    global _actor_model, _actor_opponent, _actor_device
    torch.set_num_threads(1)
    _actor_device = device
    _actor_model = MahjongNet(**config).to(device).eval()
    _actor_opponent = MahjongNet(**config).to(device).eval()


def _actor_collect(task: tuple) -> list[GameRecord]:
    state, opponent_state, first_seed, games, rule_share, pool_share, paired = task
    _actor_model.load_state_dict(state)
    batcher = Batcher(_actor_model, _actor_device)
    opponent = None
    if opponent_state is not None:
        _actor_opponent.load_state_dict(opponent_state)
        opponent = Batcher(_actor_opponent, _actor_device)
    try:
        return collect(batcher, games, first_seed, rule_share, LIVE_NOISE, opponent, pool_share, paired)
    finally:
        batcher.close()
        if opponent is not None:
            opponent.close()


def cpu_state(model: MahjongNet) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def collect_parallel(pool, model: MahjongNet, actors: int, games: int, first_seed: int,
                     rule_share: float, opponent_state: dict | None = None,
                     pool_share: float = 0.0, paired: bool = False) -> list[GameRecord]:
    """把這輪的局數平均分給各對打程序，權重以 CPU 張量傳過去（paired 時以副為單位分）。"""
    state = cpu_state(model)
    unit = 2 if paired else 1
    units = max(1, games // unit)
    share = [units // actors + (1 if index < units % actors else 0) for index in range(actors)]
    tasks, seed = [], first_seed
    for count in share:
        if count:
            tasks.append((state, opponent_state, seed, count * unit, rule_share, pool_share, paired))
        seed += count
    return [record for part in pool.map(_actor_collect, tasks) for record in part]


def keep_awake() -> None:
    """訓練期間不讓 Windows 進入睡眠（10/7 凌晨 03:16 睡著、07:57 才醒，4 小時只跑了 72 輪）。
    只對這個程序有效，程式結束就恢復，不改電源設定。"""
    try:
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    except (AttributeError, OSError):
        pass  # 非 Windows


def main() -> None:
    parser = argparse.ArgumentParser(description="強化學習自我對打（PPO）")
    parser.add_argument("--init", type=Path, help="起始權重（監督學習的結果）")
    parser.add_argument("--out", type=Path, default=Path("models/rl.pt"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games", type=int, default=512, help="每輪打幾局")
    parser.add_argument("--rule-share", type=float, default=0.5, help="每個座位換成規則式 AI 的機率")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--entropy", type=float, default=0.0,
                        help="熵獎勵權重（10/7 用 0.01 讓網路越練越隨機，預設不加）")
    parser.add_argument("--hours", type=float, default=0, help="最多跑幾小時（0 = 不限）")
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--actors", type=int, default=1,
                        help="對打程序數（每個約佔一顆 CPU 核心；1 = 在主程序裡打）")
    parser.add_argument("--pool-share", type=float, default=0.0,
                        help="每個座位換成對手池裡過去的自己的機率（對手池從 --init 開始）")
    parser.add_argument("--pool-every", type=int, default=50, help="每幾輪把目前的網路加進對手池")
    parser.add_argument("--paired", action="store_true",
                        help="同一副牌打兩次、以另一次的輸贏為基準（降雜訊，10/21 起）")
    parser.add_argument("--threads", type=int, default=0,
                        help="主程序 PyTorch CPU 執行緒數（0 = 預設）")
    args = parser.parse_args()
    keep_awake()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.init, device) if args.init else MahjongNet().to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    started = time.time()
    seed = args.seed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pool = None
    if args.actors > 1:
        import multiprocessing

        pool = multiprocessing.get_context("spawn").Pool(
            args.actors, initializer=_actor_init, initargs=(model.config, device))
    guard = Guard()
    league = [cpu_state(model)] if args.pool_share > 0 else []
    """對手池：過去的自己（最多 LEAGUE_SIZE 個，起始權重一直保留）。"""
    league_rng = Random(args.seed)
    opponent_model = None
    if league and pool is None:
        opponent_model = MahjongNet(**model.config).to(device).eval()
    for iteration in range(1, args.iterations + 1):
        tick = time.time()
        opponent_state = league_rng.choice(league) if league else None
        if pool is not None:
            records = collect_parallel(pool, model, args.actors, args.games, seed, args.rule_share,
                                       opponent_state, args.pool_share, args.paired)
        else:
            batcher = Batcher(model, device)
            opponent = None
            if opponent_state is not None:
                opponent_model.load_state_dict(opponent_state)
                opponent = Batcher(opponent_model, device)
            records = collect(batcher, args.games, seed, args.rule_share, LIVE_NOISE,
                              opponent, args.pool_share, args.paired)
            batcher.close()
            if opponent is not None:
                opponent.close()
        seed += args.games
        played = time.time() - tick
        stats = ppo_update(model, optimizer, records, device, entropy_weight=args.entropy)
        save(model, args.out, iteration=iteration, seed=seed)
        if iteration % 10 == 0:
            save(model, args.out.with_name(f"{args.out.stem}_{iteration:04d}.pt"), iteration=iteration)
        if league and iteration % args.pool_every == 0:
            league.append(cpu_state(model))
            if len(league) > LEAGUE_SIZE:
                league.pop(1)  # 留著起始權重，丟掉最舊的中途版本
        print(f"第 {iteration} 輪：{summarize(records)}；對打 {played:.0f} 秒"
              f"（{args.games / played:.1f} 局/秒）、更新 {time.time() - tick - played:.0f} 秒；"
              + "、".join(f"{k} {v:.3f}" for k, v in stats.items()), flush=True)
        reason = guard.update(stats.get("entropy", 0.0), win_gap(records))
        if reason:
            print(f"自動停止：{reason}（最後幾個存檔可能已變差，用較早的 {args.out.stem}_XXXX.pt）",
                  flush=True)
            break
        if args.hours and time.time() - started > args.hours * 3600:
            break


if __name__ == "__main__":
    main()
