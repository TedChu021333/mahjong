"""人機對戰的記錄（10/30 使用者要求，用來分析自己打得哪裡有問題）。

每圈一份 `result/play_log_<日期_時間>.jsonl`，一行一筆 JSON：
- start：開始時間、電腦 AI 種類。
- event：每個動作（誰、做了什麼），重看牌局用。
- decision：輪到自己有選擇時的局面（自己看得到的部分）、能做的動作、自己選的、同一局面 AI（rl3_0960＋護欄）會選的，
  打牌時另有每張牌打完的向聽、進張與放槍機率（`discard_options`）。
- hand_end：每局結果（誰胡、誰放槍、台數明細、四家的牌與輸贏）。
- round_end：整圈四家輸贏。
分析用 `python -m play.analyze`。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from random import Random

from agent.opponent_model import estimate_danger, unseen_counts
from agent.shanten import effective_tiles, shanten
from game.rules import Action, ActionType, GameState, Phase

RESULT_DIR = Path("result")
DANGER_SAMPLES = 60
DANGER_MIN_VALID = 100
"""放槍估計：和提示模式放槍表同一個函式，湊 100 副有效手牌（標準差約 2～3%）。"""


def action_json(action: Action | None) -> dict | None:
    if action is None:
        return None
    return {"kind": action.kind.name, "tile": action.tile, "tiles": list(action.tiles) if action.tiles else []}


def view(state: GameState, me: int) -> dict:
    """自己看得到的局面：自己的手牌，四家的副露、牌河、花牌、宣告，金牌、牌牆。"""
    return {
        "phase": state.phase.name,
        "hand": list(state.players[me].hand),
        "last_drawn": state.last_drawn if state.current_player == me and state.phase == Phase.DISCARD else None,
        "last_discard": state.last_discard if state.phase == Phase.RESPONSE else None,
        "discard_player": state.discard_player if state.phase == Phase.RESPONSE else None,
        "claimed": state.claimed,
        "melds": [[{"kind": m.kind, "tiles": list(m.tiles), "concealed": m.from_player is None}
                   for m in p.melds] for p in state.players],
        "discards": [list(p.discards) for p in state.players],
        "flowers": [list(p.flowers) for p in state.players],
        "declared": list(state.declared),
        "gold": list(state.gold_tiles),
        "drawable": state.drawable,
        "dealer": state.dealer,
        "streak": state.dealer_streak,
        "seat_winds": list(state.seat_winds),
    }


def discard_options(state: GameState, me: int, actions: list[Action], rng: Random) -> dict[int, dict]:
    """每張能打的牌：打完的向聽、進張（扣掉看得到的）、放槍機率（對手模型）。"""
    tiles = sorted({a.tile for a in actions if a.kind in (ActionType.DISCARD, ActionType.DECLARE)})
    if not tiles:
        return {}
    player = state.players[me]
    unseen = unseen_counts(state, me)
    danger = estimate_danger(state, me, tiles, rng, samples=DANGER_SAMPLES, min_valid=DANGER_MIN_VALID)
    options = {}
    for tile in tiles:
        counts = list(player.hand)
        counts[tile] -= 1
        options[tile] = {
            "shanten": shanten(counts, len(player.melds)),
            "outs": sum(unseen[t] for t in effective_tiles(counts, len(player.melds))),
            "danger": round(danger.probability[tile], 4),
        }
    return options


class Recorder:
    def __init__(self, bot, me: int, label: str, directory: Path = RESULT_DIR) -> None:
        self.bot = bot
        """和三家電腦同一個 AI，用來算「同一局面 AI 會怎麼打」。"""
        self.me = me
        self.label = label
        self.directory = directory
        self.path: Path | None = None
        self.lock = threading.Lock()
        self._pending: tuple[threading.Thread, dict] | None = None

    def new_round(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"play_log_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.write({"type": "start", "ai": self.label})

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        record = {"time": datetime.now().strftime("%H:%M:%S"), **record}
        with self.lock, self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def event(self, player: int, action: Action) -> None:
        if action.kind in (ActionType.DRAW, ActionType.PASS):
            return
        self.write({"type": "event", "player": player, "action": action_json(action)})

    # 使用者想的時候在背景算 AI 的選擇與每張牌的數字，回覆後等算完再寫
    def thinking(self, state: GameState, actions: list[Action]) -> None:
        result: dict = {}

        def work() -> None:
            started = time.monotonic()
            try:
                rng = Random(0)
                result["ai"] = action_json(self.bot(state, self.me, rng))
                result["ai_note"] = getattr(self.bot, "last_note", None)
                if state.phase == Phase.DISCARD:
                    result["options"] = discard_options(state, self.me, actions, Random(1))
            except Exception as error:  # 記錄失敗不能擋住牌局
                result["error"] = repr(error)
            result["seconds"] = round(time.monotonic() - started, 2)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self._pending = (thread, {"view": view(state, self.me),
                                  "actions": [action_json(a) for a in actions], "result": result})

    def decided(self, choice: Action) -> None:
        if self._pending is None:
            return
        thread, record = self._pending
        self._pending = None
        thread.join()
        result = record.pop("result")
        self.write({"type": "decision", **record, "choice": action_json(choice), **result})

    def hand_end(self, summary) -> None:
        result = summary.result
        state = result.state
        self.write({
            "type": "hand_end", "number": summary.number, "dealer": summary.dealer, "streak": summary.streak,
            "winner": result.winner, "discarder": result.discarder, "self_draw": result.winner is not None
            and not result.win_by_discard, "win_tile": result.win_tile,
            "tai": result.score.tai if result.score is not None else 0,
            "patterns": list(result.score.patterns) if result.score is not None else [],
            "pay": list(summary.pay), "totals": list(summary.totals),
            "declared": list(state.declared),
            "hands": [list(p.hand) for p in state.players],
            "melds": [[{"kind": m.kind, "tiles": list(m.tiles)} for m in p.melds] for p in state.players],
        })

    def round_end(self, totals: list[int]) -> None:
        self.write({"type": "round_end", "totals": list(totals)})
