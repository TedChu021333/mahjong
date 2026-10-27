"""專家迭代第 0 階段：搜尋產生教材可不可行——同一個局面的幾張候選牌，要模擬幾次才分得出 0.1～0.3 台的差距。

做法：規則式四家互打，隨機抽輪到某家打牌的局面（沒宣告、至少兩張打完向聽一樣好的牌）。每個局面：
1. 補完：對手暗手與牌牆（含花牌）從這家看不到的牌重新洗、照原本張數發回，金牌候選也重洗。
2. 每張候選牌都用同一副補完的牌、同一個亂數種子，規則式四家打到這局結束，記這家的輸贏（扣抽成，台）。
3. 候選與規則式那張逐次相減：同一副牌打法只差一張，相減能扣掉多少運氣，決定一個局面要模擬幾次。
`--other nn:models/rl3_0960.pt`：只比另一個 AI 和規則式選得不一樣的局面（候選就是這兩張），看規則式模擬
認不認得出比較強的 AI 的選擇——認不出就不能拿規則式模擬當老師。
`--streams`：牌牆分成四疊、各家只摸自己那疊（`StreamWall`）。打出不同的牌後吃碰會改變輪次，一般牌牆之後
每家摸到的牌全部錯開；分疊時每家摸到的牌不變，相減應該能扣掉更多運氣。

`python -m training.teacher_probe --points 200 --rollouts 200 --workers 2 --out result/teacher_probe.csv`
"""
from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from multiprocessing import Pool
from random import Random
from statistics import mean, pstdev

from agent.rule_agent import choose_rule_action_for_player
from agent.shanten import shanten
from game.rules import Action, ActionType, GameState, Phase, legal_actions
from game.simulator import after_rake, payments, play_game
from game.tiles import is_flower, tile_name

CANDIDATES = 4
"""每個局面最多比幾張牌（規則式那張 + 打完向聽一樣好的其他牌）。"""


@dataclass
class Point:
    state: GameState
    player: int
    candidates: list[int]
    """第一張是規則式選的牌。"""


def rule_policy(state: GameState, player: int, rng: Random) -> Action | None:
    return choose_rule_action_for_player(state, player, rng)


def discard_candidates(state: GameState, player: int, chosen: int, rng: Random,
                       limit: int = CANDIDATES) -> list[int]:
    """規則式那張排第一，再加打完向聽數和它一樣的其他牌（隨機挑）。"""
    me = state.players[player]
    tiles = sorted({action.tile for action in legal_actions(state, player)
                    if action.kind == ActionType.DISCARD and action.tile is not None})

    def after(tile: int) -> int:
        hand = list(me.hand)
        hand[tile] -= 1
        return shanten(hand, len(me.melds))

    best = after(chosen)
    others = [tile for tile in tiles if tile != chosen and after(tile) == best]
    rng.shuffle(others)
    return [chosen] + others[:limit - 1]


def sample_points(count: int, seed: int, rate: float = 0.05, other=None) -> list[Point]:
    """規則式互打，每個符合條件的打牌局面以 rate 機率記下來。給 other（同 PlayerPolicy 介面）時，
    只記 other 打另一張牌的局面，候選是 [規則式那張, other 那張]。"""
    rng = Random(seed)
    points: list[Point] = []

    def policy(state: GameState, player: int, game_rng: Random) -> Action | None:
        action = rule_policy(state, player, game_rng)
        if (len(points) < count and action is not None and action.kind == ActionType.DISCARD
                and state.phase == Phase.DISCARD and not state.declared[player]
                and rng.random() < rate):
            if other is None:
                candidates = discard_candidates(state, player, action.tile, rng)
            else:
                choice = other(state, player, rng)
                candidates = [action.tile]
                if (choice is not None and choice.kind == ActionType.DISCARD
                        and choice.tile != action.tile):
                    candidates.append(choice.tile)
            if len(candidates) >= 2:
                points.append(Point(deepcopy(state), player, candidates))
        return action

    while len(points) < count:
        play_game(Random(rng.getrandbits(32)), player_policy=policy, dealer=rng.randrange(4))
    return points


class StreamWall:
    """各家各一疊的牌牆：誰摸（state.current_player）就從誰那疊拿，那疊摸完才拿別疊最多的。"""

    def __init__(self, state: GameState, tiles: list[int]) -> None:
        self.state = state
        self.streams = [tiles[seat::4] for seat in range(4)]

    def __len__(self) -> int:
        return sum(len(stream) for stream in self.streams)

    def __iter__(self):
        for stream in self.streams:
            yield from stream

    def pop(self) -> int:
        stream = self.streams[self.state.current_player]
        if not stream:
            stream = max(self.streams, key=len)
        return stream.pop()


def determinize(state: GameState, player: int, rng: Random, streams: bool = False) -> GameState:
    """對手暗手與牌牆從這家看不到的牌重發（各家暗手張數不變，花牌只會在牌牆）。"""
    world = deepcopy(state)
    hidden = list(world.wall)
    sizes = {}
    for opponent in range(4):
        if opponent == player:
            continue
        hand = world.players[opponent].hand
        sizes[opponent] = sum(hand)
        hidden += [tile for tile, n in enumerate(hand) for _ in range(n)]
        world.players[opponent].hand = [0] * len(hand)
    rng.shuffle(hidden)
    for opponent, size in sizes.items():
        hand = world.players[opponent].hand
        while size:
            index = next(i for i, tile in enumerate(hidden) if not is_flower(tile))
            hand[hidden.pop(index)] += 1
            size -= 1
    world.wall = StreamWall(world, hidden) if streams else hidden
    rng.shuffle(world.gold_pool)
    return world


def evaluate(point: Point, rollouts: int, seed: int, streams: bool = False) -> list[list[float]]:
    """每次模擬一列：各候選這家的輸贏（同一副補完的牌、同一個種子）。"""
    rng = Random(seed)
    rows = []
    for _ in range(rollouts):
        world = determinize(point.state, point.player, rng, streams)
        game_seed = rng.getrandbits(32)
        row = []
        for tile in point.candidates:
            start = deepcopy(world)
            # 打出候選牌的那一步交給 play_game：第一步固定打它，之後照規則式
            first = [True]

            def policy(state: GameState, player: int, game_rng: Random, tile=tile) -> Action | None:
                if first[0] and player == point.player and state.phase == Phase.DISCARD:
                    first[0] = False
                    return Action(ActionType.DISCARD, tile)
                return rule_policy(state, player, game_rng)

            result = play_game(Random(game_seed), player_policy=policy, start=start)
            row.append(after_rake(payments(result)[point.player]))
        rows.append(row)
    return rows


def _evaluate_job(job: tuple[Point, int, int, bool]) -> list[list[float]]:
    return evaluate(*job)


def summarize(point: Point, rows: list[list[float]]) -> dict:
    n = len(rows)
    base = [row[0] for row in rows]
    out = {"player": point.player, "candidates": " ".join(tile_name(t) for t in point.candidates),
           "rollouts": n, "rule_mean": mean(base), "sd_single": pstdev(base)}
    diffs = []
    for c in range(1, len(point.candidates)):
        d = [row[c] - row[0] for row in rows]
        sd = pstdev(d)
        diffs.append((mean(d), sd, sd / sqrt(n), pstdev([row[c] for row in rows])))
    out["diff_means"] = [d[0] for d in diffs]
    out["diff_sds"] = [d[1] for d in diffs]
    out["diff_ses"] = [d[2] for d in diffs]
    out["other_sds"] = [d[3] for d in diffs]
    return out


def report(summaries: list[dict]) -> str:
    sd_single = sqrt(mean([s["sd_single"] ** 2 for s in summaries]))
    pairs = [(m, sd, se, osd, s["sd_single"]) for s in summaries
             for m, sd, se, osd in zip(s["diff_means"], s["diff_sds"], s["diff_ses"], s["other_sds"])]
    sd_paired = sqrt(mean([p[1] ** 2 for p in pairs]))
    sd_indep = sqrt(mean([p[3] ** 2 + p[4] ** 2 for p in pairs]))
    identical = mean([p[1] == 0 for p in pairs])
    lines = [f"局面 {len(summaries)}、候選對 {len(pairs)}、每局面模擬 {summaries[0]['rollouts']} 次",
             f"單次輸贏標準差 {sd_single:.2f} 台",
             f"兩張候選逐次相減的標準差 {sd_paired:.2f} 台（分開模擬時約 {sd_indep:.2f}）；"
             f"{identical:.0%} 的候選對每次結果完全一樣",
             "要分出差距（2 倍標準誤差）需要的模擬次數："]
    for edge in (0.1, 0.2, 0.3, 0.5):
        lines.append(f"  {edge:.1f} 台：{(2 * sd_paired / edge) ** 2:,.0f} 次"
                     f"（分開模擬 {(2 * sd_indep / edge) ** 2:,.0f} 次）")
    gaps = sorted(abs(p[0]) for p in pairs)
    significant = mean([abs(p[0]) > 2 * p[2] for p in pairs if p[2] > 0] or [0])
    lines.append(f"候選與規則式那張的差距（絕對值）：中位數 {gaps[len(gaps) // 2]:.2f}、"
                 f"75% {gaps[3 * len(gaps) // 4]:.2f}、90% {gaps[9 * len(gaps) // 10]:.2f} 台；"
                 f"超過 2 倍標準誤差的 {significant:.0%}")
    better = [p for p in pairs if p[0] > 2 * p[2] and p[2] > 0]
    lines.append(f"候選顯著比規則式好的 {len(better)} 對、平均好 "
                 f"{mean([p[0] for p in better]) if better else 0:.2f} 台")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--other", default=None, help="和規則式比較的 AI（eval.arena 的代號，例如 nn:models/rl3_0960.pt）")
    parser.add_argument("--streams", action="store_true", help="牌牆分成各家一疊")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    started = time.time()
    other = None
    if args.other:
        from eval.arena import make_policy
        other = make_policy(args.other)
    points = sample_points(args.points, args.seed, other=other)
    jobs = [(point, args.rollouts, args.seed * 1_000_003 + i, args.streams) for i, point in enumerate(points)]
    summaries = []
    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap(_evaluate_job, jobs)):
            summaries.append(summarize(points[i], rows))
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(points)} 局面，{time.time() - started:.0f} 秒", flush=True)
    print(report(summaries))
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["player", "candidates", "rollouts", "rule_mean", "sd_single",
                             "diff_means", "diff_ses"])
            for s in summaries:
                writer.writerow([s["player"], s["candidates"], s["rollouts"], f"{s['rule_mean']:.3f}",
                                 f"{s['sd_single']:.3f}",
                                 " ".join(f"{m:.3f}" for m in s["diff_means"]),
                                 " ".join(f"{e:.3f}" for e in s["diff_ses"])])


if __name__ == "__main__":
    main()
