"""把 AI 的動作轉成滑鼠點擊（只在遊戲的訓練場使用）。

座標與螢幕擷取相同（1920x1080、顯示縮放 100%）。pyautogui 的 FAILSAFE 保持開啟：
把滑鼠甩到螢幕左上角會立刻中止程式。
"""
from __future__ import annotations

import time
from typing import Callable

from game.rules import Action, ActionType
from hint.advisor import Observation
from perception.config import ACTION_BUTTON_CENTERS, CANCEL_BUTTON_REGION

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

    def perform(self, action: Action, observation: Observation) -> bool:
        """執行動作；畫面上找不到對應的按鈕或牌時回傳 False，不亂點。"""
        buttons = observation.buttons or frozenset()
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
        self.click(ACTION_BUTTON_CENTERS[name])
        return True

    def _click_tile(self, tile: int | None, observation: Observation) -> bool:
        # 優先打摸進的那張（在最後），其次是最右邊的同種牌
        for hand_tile, box in reversed(list(zip(observation.hand, observation.boxes))):
            if hand_tile == tile:
                left, top, right, bottom = box
                self.click(((left + right) // 2, (top + bottom) // 2))
                return True
        return False
