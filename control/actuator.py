"""把 AI 的動作轉成滑鼠點擊（只在遊戲的訓練場使用）。

座標與螢幕擷取相同（1920x1080、顯示縮放 100%）。pyautogui 的 FAILSAFE 保持開啟：
把滑鼠甩到螢幕左上角會立刻中止程式。
"""
from __future__ import annotations

import time
from typing import Callable

from game.rules import Action, ActionType
from hint.advisor import Observation, chow_options
from perception.config import (
    ACTION_BUTTON_CENTERS,
    CANCEL_BUTTON_REGION,
    SWAP_CONFIRM_BUTTON,
    SWAP_KEEP_BUTTON,
)

Point = tuple[int, int]
BUTTON_FOR_ACTION = {
    ActionType.CHOW: "chow",
    ActionType.PUNG: "pung",
    ActionType.KONG: "kong",
    ActionType.WIN: "win",
}
STEP_DELAY = 0.4
"""按「聽」之後等選牌畫面出現的時間。"""


def _pyautogui_click(point: Point) -> None:
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.click(*point)


class Actuator:
    def __init__(self, click: Callable[[Point], None] = _pyautogui_click,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.click = click
        self.sleep = sleep
        self.pending_chow: int | None = None
        """按了「吃」之後，要在選項框中選第幾個。"""

    def follow_up(self, observation: Observation) -> bool:
        """吃法選項框出現時，點之前決定好的那一個。"""
        if self.pending_chow is None or not observation.chow_panels:
            return False
        index = self.pending_chow
        self.pending_chow = None
        if index >= len(observation.chow_panels):
            return False
        self.click(observation.chow_panels[index])
        return True

    def perform(self, action: Action, observation: Observation) -> bool:
        """執行動作；畫面上找不到對應的按鈕或牌時回傳 False，不亂點。"""
        buttons = observation.buttons or frozenset()
        if action.kind == ActionType.SWAP:
            return self._swap(action.tiles, observation)
        if action.kind == ActionType.DISCARD:
            return self._click_tile(action.tile, observation)
        if action.kind == ActionType.DECLARE:
            if "declare" not in buttons:
                return self._click_tile(action.tile, observation)
            self.click(ACTION_BUTTON_CENTERS["declare"])
            self.sleep(STEP_DELAY)
            return self._click_tile(action.tile, observation)
        if action.kind == ActionType.PASS:
            if observation.buttons is None:
                return False
            left, top, right, bottom = CANCEL_BUTTON_REGION
            self.click(((left + right) // 2, (top + bottom) // 2))
            return True
        name = BUTTON_FOR_ACTION.get(action.kind)
        if name is None or name not in buttons:
            return False
        if action.kind == ActionType.CHOW:
            options = chow_options(observation)
            if len(options) > 1 and action.tiles in options:
                self.pending_chow = options.index(action.tiles)
        self.click(ACTION_BUTTON_CENTERS[name])
        return True

    def _swap(self, tiles: tuple[int, ...], observation: Observation) -> bool:
        """一次只點一下：先把選起的牌調整成要換的牌，一致後再按確認。"""
        if observation.swap_prompt is None:
            return False
        wanted = set()
        for tile in tiles:
            index = max((i for i, t in enumerate(observation.hand)
                         if t == tile and i not in wanted), default=None)
            if index is None:
                return False
            wanted.add(index)
        raised = {i for i, up in enumerate(observation.raised) if up}
        wrong = sorted(wanted ^ raised)
        if wrong:
            left, top, right, bottom = observation.boxes[wrong[0]]
            self.click(((left + right) // 2, (top + bottom) // 2))
            return True
        if not wanted:
            self.click(SWAP_KEEP_BUTTON)
            return True
        if not observation.swap_prompt:
            return False  # 「換牌」還沒亮
        self.click(SWAP_CONFIRM_BUTTON)
        return True

    def _click_tile(self, tile: int | None, observation: Observation) -> bool:
        # 優先打摸進的那張（在最後），其次是最右邊的同種牌
        for hand_tile, box in reversed(list(zip(observation.hand, observation.boxes))):
            if hand_tile == tile:
                left, top, right, bottom = box
                self.click(((left + right) // 2, (top + bottom) // 2))
                return True
        return False
