"""即時模式的牌河追蹤：記錄各家打過的牌，順便自動收集牌河模板。

牌河模板一開始很少，辨識不出的牌記為未知。每次出牌都有可靠的標註來源：
- 自己打的牌：比對打牌前後的暗手牌就知道。
- 別家打的牌：能吃碰時會放大顯示（辨識很準），就是那家牌河接著出現的那張。
辨識器認不出或認錯時，把這張牌存成新模板（train/river_templates/<牌>/auto_*.png），
並以標註修正記錄。
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from game.tiles import tile_code
from hint.advisor import RIVER_SEAT, Observation
from perception.capture import write_image
from perception.config import RIVER_TEMPLATE_DIR
from perception.river_reader import SEATS, RiverReader, crop_face
from perception.river_tracker import RiverEvent, RiverTracker

LABEL_FRAMES = 15
"""標註在這麼多幀內等不到那家牌河多一張就作廢。"""


class RiverWatch:
    def __init__(self, reader: RiverReader, template_dir: Path = RIVER_TEMPLATE_DIR,
                 save=write_image) -> None:
        self.reader = reader
        self.tracker = RiverTracker()
        self.template_dir = template_dir
        self.save = save
        self.saved = 0
        self._frame = 0
        self._labels: dict[str, tuple[int, int]] = {}
        """座位 → (牌, 取得標註時的幀數)。"""
        self._turn_hand: tuple[int, ...] | None = None

    def new_hand(self) -> None:
        self.tracker.reset()
        self._labels.clear()
        self._turn_hand = None

    def update(self, frame: np.ndarray, observation: Observation) -> list[RiverEvent]:
        self._frame += 1
        self._collect_labels(observation)
        events = self.tracker.update(self.reader.read(frame))
        for event in events:
            if event.kind == "discard":
                self._apply_label(frame, event)
        return events

    def discards(self) -> tuple[tuple[int, ...], ...]:
        """依 advisor 的座位編號排列，未知的牌略過。"""
        result = [()] * 4
        for seat, tiles in self.tracker.discards.items():
            result[RIVER_SEAT[seat]] = tuple(tile for tile in tiles if tile is not None)
        return tuple(result)

    def _collect_labels(self, observation: Observation) -> None:
        claim = observation.claim
        if claim is not None and claim.tile is not None:
            self._labels[claim.source] = (claim.tile, self._frame)
        if observation.my_turn:
            self._turn_hand = observation.hand
        elif self._turn_hand is not None and len(observation.hand) == len(self._turn_hand) - 1:
            gone = Counter(self._turn_hand) - Counter(observation.hand)
            if sum(gone.values()) == 1:
                self._labels["me"] = (next(iter(gone)), self._frame)
            self._turn_hand = None

    def _apply_label(self, frame: np.ndarray, event: RiverEvent) -> None:
        label = self._labels.pop(event.seat, None)
        if label is None or self._frame - label[1] > LABEL_FRAMES:
            return
        tile = label[0]
        entry = self.tracker.rivers[event.seat][-1]
        if entry.tile == tile:
            return  # 已經認得，不必再存
        path = self.template_dir / tile_code(tile) / \
            f"auto_{time.strftime('%Y%m%d_%H%M%S')}_{self._frame}_{event.seat}.png"
        self.save(path, crop_face(frame, event.box))
        self.saved += 1
        entry.tile, entry.score = tile, 1.0


def summary(watch: RiverWatch) -> str:
    from game.tiles import tile_name

    parts = []
    for seat in SEATS:
        tiles = watch.tracker.discards[seat]
        parts.append(f"{seat} " + "".join(tile_name(t) if t is not None else "?" for t in tiles))
    return "；".join(parts)
