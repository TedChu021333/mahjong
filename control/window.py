"""遊戲視窗管理（Windows）：找出遊戲視窗、切到最前面，以及使用者多久沒操作。

實戰 10/3 03:03 VS Code 跑到遊戲前面，提示程式看不到遊戲就停了 8 小時。自動模式在看不到遊戲、
而且使用者沒有在操作電腦時，才把遊戲切回前面（使用者自己切去別的程式時不會搶回來）。
"""
from __future__ import annotations

import ctypes
import sys

GAME_TITLE = "明星3缺1"
"""遊戲視窗標題的一部分（Google Play Games 的視窗：「麻將 明星3缺1麻將 - …」）。"""
SW_RESTORE = 9
VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002


def _user32():
    if sys.platform != "win32":
        return None
    return ctypes.windll.user32


def find_game_window(title: str = GAME_TITLE) -> int | None:
    user32 = _user32()
    if user32 is None:
        return None
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        if length and user32.IsWindowVisible(hwnd):
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if title in buffer.value:
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(visit, 0)
    return found[0] if found else None


def game_is_foreground(title: str = GAME_TITLE) -> bool | None:
    """遊戲是不是最前面的視窗；找不到遊戲視窗（或不是 Windows）時為 None。"""
    user32 = _user32()
    hwnd = find_game_window(title)
    if user32 is None or hwnd is None:
        return None
    return user32.GetForegroundWindow() == hwnd


def focus_game(title: str = GAME_TITLE) -> bool:
    """把遊戲視窗切到最前面。Windows 只讓「剛收到輸入」的程式換前景視窗，所以先送一個 Alt。"""
    user32 = _user32()
    hwnd = find_game_window(title)
    if user32 is None or hwnd is None:
        return False
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.keybd_event(VK_MENU, 0, 0, 0)
    user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
    return bool(user32.SetForegroundWindow(hwnd))


class _LastInputInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def user_idle_seconds() -> float:
    """距離最後一次鍵盤、滑鼠輸入幾秒（自動點擊也算輸入，但看不到遊戲時程式不會點）。"""
    user32 = _user32()
    if user32 is None:
        return 0.0
    info = _LastInputInfo()
    info.cbSize = ctypes.sizeof(info)
    if not user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    return (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0


class GameWindow:
    """提示迴圈用的遊戲視窗操作（測試時換成假的）。"""

    def is_foreground(self) -> bool | None:
        return game_is_foreground()

    def focus(self) -> bool:
        return focus_game()

    def idle_seconds(self) -> float:
        return user_idle_seconds()
