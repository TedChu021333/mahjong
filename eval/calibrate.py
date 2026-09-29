"""從模擬器對局蒐集策略模型要用的機率表，寫到 agent/calibration.json。

    python -m eval.calibrate --games 4000 --workers 14

每次有人要打牌時記錄：
- 其他三家是否已聽牌（對應他們打過幾張、幾組副露；宣告過的也算）→ 聽牌機率。
  模擬器裡聽牌一律宣告，實戰約四成聽牌不宣告，換算見 agent.opponent_model。
- 打牌者打完後的向聽數、有效進張、牌牆剩餘張數，以及這局最後胡牌贏了多少 → 進攻的期望收益
- 胡牌者的台數（依有沒有宣告、自摸或放槍）→ 預估的輸贏金額
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from random import Random

from agent.rule_agent import choose_rule_action_for_player
from agent.shanten import effective_tiles, shanten
from game.rules import ActionType, Phase
from eval.arena import payments
from game.simulator import play_game

OUTPUT = Path(__file__).resolve().parent.parent / "agent" / "calibration.json"
DISCARD_BUCKETS = 16
"""打過幾張的分組上限（之後都歸在最後一組）。"""
WALL_BUCKET = 8
"""牌牆剩餘張數每 8 張一組。"""
OUTS_BUCKETS = (0, 4, 8, 12, 16, 24, 32)


def outs_bucket(outs: int) -> int:
    return max(index for index, low in enumerate(OUTS_BUCKETS) if outs >= low)


def _play(seed: int) -> dict:
    tenpai = defaultdict(lambda: [0, 0])       # (打過幾張, 副露) → [聽牌次數, 總次數]
    offense = []                               # (玩家, 向聽, 進張組, 牌牆組)
    records = {"tenpai": tenpai, "offense": offense}

    def policy(state, player, rng):
        action = choose_rule_action_for_player(state, player, rng)
        if state.phase == Phase.DISCARD and action is not None \
                and action.kind in (ActionType.DISCARD, ActionType.DECLARE) \
                and not state.declared[player]:
            for opponent in range(4):
                other = state.players[opponent]
                if opponent == player:
                    continue
                key = (min(len(other.discards), DISCARD_BUCKETS - 1), len(other.melds))
                tenpai[key][0] += shanten(other.hand, other.open_melds) == 0
                tenpai[key][1] += 1
            me = state.players[player]
            after = list(me.hand)
            after[action.tile] -= 1
            outs = sum(4 - after[t] for t in effective_tiles(after, me.open_melds))
            offense.append((player, shanten(after, me.open_melds), outs_bucket(outs),
                            state.drawable // WALL_BUCKET))
        return action

    result = play_game(Random(seed), player_policy=policy, dealer=seed % 4, swap=True)
    records["winner"] = result.winner
    records["gain"] = payments(result)[result.winner] if result.winner is not None else 0
    records["win"] = None if result.score is None else (
        result.score.tai, bool(result.state.declared[result.winner]), not result.win_by_discard)
    records["tenpai"] = {f"{d},{m}": v for (d, m), v in tenpai.items()}
    return records


def calibrate(games: int, workers: int, start: int = 0) -> dict:
    tenpai = defaultdict(lambda: [0, 0])
    offense = defaultdict(lambda: [0, 0, 0])
    """(向聽, 進張組, 牌牆組) → [胡牌次數, 總次數, 胡牌時贏的總台數]"""
    tai = defaultdict(list)
    with Pool(workers) as pool:
        for done, record in enumerate(pool.imap_unordered(_play, range(start, start + games),
                                                          chunksize=4), 1):
            for key, (hit, total) in record["tenpai"].items():
                tenpai[key][0] += hit
                tenpai[key][1] += total
            for player, level, outs, wall in record["offense"]:
                key = f"{level},{outs},{wall}"
                won = record["winner"] == player
                offense[key][0] += won
                offense[key][1] += 1
                offense[key][2] += record["gain"] if won else 0
            if record["win"] is not None:
                points, declared, self_draw = record["win"]
                tai[f"{int(declared)},{int(self_draw)}"].append(points)
            if done % 500 == 0:
                print(f"{done} 局", flush=True)
    return {
        "games": games,
        "discard_buckets": DISCARD_BUCKETS,
        "wall_bucket": WALL_BUCKET,
        "outs_buckets": list(OUTS_BUCKETS),
        "tenpai": {key: list(value) for key, value in sorted(tenpai.items())},
        "offense": {key: list(value) for key, value in sorted(offense.items())},
        "tai": {key: [sum(values) / len(values), len(values)] for key, values in sorted(tai.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="蒐集策略模型的機率表")
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=100_000, help="第一個種子（避開 arena 用的）")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    table = calibrate(args.games, args.workers, args.start)
    args.output.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已寫入 {args.output}")


if __name__ == "__main__":
    main()
