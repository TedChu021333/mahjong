"""提示模式入口。

    python -m hint                         # 即時擷取螢幕（遊戲最大化、1920x1080）
    python -m hint --image train/吃牌畫面.png
    python -m hint --video train/遊戲流程.mp4 --step 1
    python -m hint --auto                  # 自動點擊，只在遊戲的訓練場使用

連續兩幀辨識結果相同才給建議（避開理牌、摸牌等動畫），建議改變時才印出。
自動模式：同一個畫面只操作一次；點完 RETRY_AFTER 秒畫面沒變（例如出牌要點兩下）
才補點，最多 MAX_RETRIES 次。滑鼠甩到螢幕左上角會立刻中止。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from hint.advisor import Readers, chow_options, decide, describe, observe
from perception.capture import grab_screen, read_image

RETRY_AFTER = 1.5
MAX_RETRIES = 2


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
        actuator=None, observe_fn=observe, clock=time.monotonic) -> None:
    previous = None
    last_advice = None
    acted_for = None
    acted_at = 0.0
    retries = 0
    for stamp, frame in frames:
        observation = observe_fn(frame, readers)
        if observation is None or observation != previous:
            previous = observation
            continue
        action = decide(observation)
        advice = describe(action) if action is not None else None
        if advice is not None and advice != last_advice:
            print(f"[{stamp}] {advice}", flush=True)
        last_advice = advice
        if actuator is None or action is None:
            continue
        if observation == acted_for:
            if clock() - acted_at < RETRY_AFTER or retries >= MAX_RETRIES:
                continue
            retries += 1
        else:
            retries = 0
        if action.kind.value == "chow" and chow_options(observation) > 1:
            if observation != acted_for:
                print(f"[{stamp}] 有多種吃法，請手動操作", flush=True)
        elif not actuator.perform(action, observation):
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
                        help="自動點擊（只在遊戲的訓練場使用；滑鼠甩到左上角中止）")
    args = parser.parse_args()
    readers = Readers()
    if args.image:
        observation = observe(read_image(args.image), readers)
        action = decide(observation) if observation else None
        print(describe(action) if action else "畫面無法辨識或不需要做決定")
    elif args.video:
        run(video_frames(args.video, args.step, args.start), readers)
    elif args.auto:
        from control.actuator import Actuator

        print("自動模式啟動：只在訓練場使用。滑鼠甩到螢幕左上角可立即中止。", flush=True)
        time.sleep(3)  # 讓使用者切回遊戲視窗
        run(screen_frames(args.interval), readers, actuator=Actuator())
    else:
        print("提示模式啟動，Ctrl+C 結束", flush=True)
        run(screen_frames(args.interval), readers)


if __name__ == "__main__":
    main()
