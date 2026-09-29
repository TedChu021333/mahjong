"""提示模式入口。

    python -m hint                         # 即時擷取螢幕（遊戲最大化、1920x1080）
    python -m hint --image train/吃牌畫面.png
    python -m hint --video train/遊戲流程.mp4 --step 1
    python -m hint --no-overlay            # 建議只印在終端機，不顯示浮動小視窗
    python -m hint --auto                  # 自動點擊，只在對手全是電腦的訓練場使用

連續兩幀辨識結果相同才給建議（避開理牌、摸牌等動畫），建議改變時才印出。
自動模式：同一個畫面只操作一次；點完 RETRY_AFTER 秒畫面沒變（例如出牌要點兩下）
才補點，最多 MAX_RETRIES 次。讀不到手牌時檢查打完後的「繼續」類按鈕，
同樣連續兩幀一致才點；按下小結算「繼續」後等 NEXT_GAME_DELAY 秒再點下一場按鈕
（只在讀不到手牌時點，避免點到手牌）。每局結算畫面在按「繼續」前存到 result/。
滑鼠甩到螢幕左上角會立刻中止。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from game.rules import ActionType, forbidden_after_claim
from hint.advisor import Readers, decide, describe, hand_after_claim, observe
from perception.continue_button import ContinueButton
from perception.capture import grab_screen, read_image, write_image
from perception.config import CONTINUE_BUTTONS, NEXT_GAME_CLICKS, NEXT_GAME_DELAY, NEXT_GAME_REGION

RETRY_AFTER = 1.5
MAX_RETRIES = 2
RESULT_DIR = Path("result")
"""每局結算畫面的截圖。"""


def save_result(frame: np.ndarray) -> Path:
    path = RESULT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.png"
    write_image(path, frame)
    return path


def screen_frames(interval: float) -> Iterator[tuple[str, np.ndarray]]:
    while True:
        yield time.strftime("%H:%M:%S"), grab_screen()
        time.sleep(interval)


def video_frames(path: Path, step: float, start: float) -> Iterator[tuple[str, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"無法開啟影片：{path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    every = max(1, round(step * fps))
    index = 0
    while capture.grab():
        if index >= start * fps and index % every == 0:
            ok, frame = capture.retrieve()
            if ok:
                seconds = int(index / fps)
                yield f"{seconds // 60:02d}:{seconds % 60:02d}", frame
        index += 1


def run(frames: Iterator[tuple[str, np.ndarray]], readers: Readers,
        actuator=None, observe_fn=observe, clock=time.monotonic, show=None,
        find_continue=None, save_result=save_result) -> None:
    previous = None
    last_advice = None
    acted_for = None
    acted_at = 0.0
    retries = 0
    next_game_clicks = 0
    """按下小結算「繼續」後還要點幾次下一場按鈕。"""
    pending_claim = None
    """最近一次建議的吃碰：(吃碰後應有的手牌, 不能打的牌)。"""
    for stamp, frame in frames:
        observation = observe_fn(frame, readers)
        if observation is None and find_continue is not None:
            observation = find_continue(frame)
        if observation is None and next_game_clicks and clock() - acted_at >= NEXT_GAME_DELAY:
            left, top, right, bottom = NEXT_GAME_REGION
            print(f"[{stamp}] 按下一場", flush=True)
            actuator.click(((left + right) // 2, (top + bottom) // 2))
            next_game_clicks -= 1
            acted_at = clock()
        if observation is not None and not isinstance(observation, ContinueButton):
            next_game_clicks = 0  # 新的一局已經開始
        if observation is None or observation != previous:
            previous = observation
            continue
        if isinstance(observation, ContinueButton):
            if observation == acted_for:
                if clock() - acted_at < RETRY_AFTER or retries >= MAX_RETRIES:
                    continue
                retries += 1
            else:
                retries = 0
                print(f"[{stamp}] 結算截圖存到 {save_result(frame)}", flush=True)
                print(f"[{stamp}] 按「{observation.name}」", flush=True)
            actuator.click(observation.center)
            acted_for, acted_at = observation, clock()
            next_game_clicks = NEXT_GAME_CLICKS
            continue
        if actuator is not None and actuator.follow_up(observation):
            print(f"[{stamp}] 選擇吃法", flush=True)
            acted_for, acted_at, retries = observation, clock(), 0
            continue
        if observation.my_turn and pending_claim is not None:
            expected, forbidden = pending_claim
            if tuple(sorted(observation.hand)) == expected:
                observation = replace(observation, claimed=True, forbidden=forbidden)
            else:
                pending_claim = None
        action = decide(observation)
        if action is not None and action.kind in (ActionType.CHOW, ActionType.PUNG):
            pending_claim = (hand_after_claim(observation, action),
                             forbidden_after_claim(action, action.tile))
        advice = describe(action) if action is not None else None
        if advice != last_advice:
            if advice is not None:
                print(f"[{stamp}] {advice}", flush=True)
            if show is not None:
                show(advice)
        last_advice = advice
        if actuator is None or action is None:
            continue
        if observation == acted_for:
            if clock() - acted_at < RETRY_AFTER or retries >= MAX_RETRIES:
                continue
            retries += 1
        else:
            retries = 0
        if not actuator.perform(action, observation):
            print(f"[{stamp}] 畫面上找不到對應的按鈕或牌，略過", flush=True)
        acted_for = observation
        acted_at = clock()


def main() -> None:
    parser = argparse.ArgumentParser(description="麻將提示模式")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--step", type=float, default=0.5, help="影片每隔幾秒取一幀")
    parser.add_argument("--start", type=float, default=0.0, help="影片從第幾秒開始")
    parser.add_argument("--interval", type=float, default=0.3, help="即時模式擷取間隔（秒）")
    parser.add_argument("--auto", action="store_true",
                        help="自動點擊（只在對手全是電腦的訓練場使用；滑鼠甩到左上角中止）")
    parser.add_argument("--no-overlay", action="store_true", help="不顯示浮動小視窗")
    parser.add_argument("--demo-overlay", action="store_true",
                        help="只展示小視窗外觀（可拖曳調整位置），不辨識畫面")
    args = parser.parse_args()
    if args.demo_overlay:
        from hint.overlay import demo

        demo()
        return
    if args.image:
        observation = observe(read_image(args.image), Readers())
        action = decide(observation) if observation else None
        print(describe(action) if action else "畫面無法辨識或不需要做決定")
    elif args.video:
        run(video_frames(args.video, args.step, args.start), Readers())
    else:
        live(args.interval, overlay=not args.no_overlay, auto=args.auto)


def confirm_training_room(ask=input) -> bool:
    """經典版一般場是與真人配桌，自動點擊只能用在對手全是電腦的訓練場。"""
    print("自動模式只能在「訓練場」（三家對手都是電腦）使用；與真人配桌的場請用提示模式。")
    answer = ask("確認目前是對手全為電腦的訓練場？輸入 y 繼續：")
    return answer.strip().lower() == "y"


def live(interval: float, overlay: bool, auto: bool) -> None:
    actuator = None
    if auto:
        if not confirm_training_room():
            raise SystemExit("已取消自動模式。")
        from control.actuator import Actuator

        actuator = Actuator()
        print("自動模式 3 秒後開始，請切回遊戲視窗。滑鼠甩到螢幕左上角可立即中止。", flush=True)
        time.sleep(3)
    mode = "自動模式" if auto else "提示模式"
    readers = Readers()
    find_continue = None
    if auto:
        from perception.continue_button import ContinueButtons

        buttons = ContinueButtons.from_directory()
        missing = [name for name in CONTINUE_BUTTONS if name not in buttons.references]
        if missing:
            print(f"缺少按鈕參考圖，不會自動按：{'、'.join(missing)}", flush=True)
        find_continue = buttons.find
    if not overlay:
        print(f"{mode}啟動，Ctrl+C 結束", flush=True)
        run(screen_frames(interval), readers, actuator=actuator, find_continue=find_continue)
        return
    from hint.overlay import Overlay

    print(f"{mode}啟動（浮動小視窗），Ctrl+C 結束", flush=True)
    window = Overlay()
    window.run(lambda: run(screen_frames(interval), readers, actuator=actuator,
                           show=window.show, find_continue=find_continue))


if __name__ == "__main__":
    main()
