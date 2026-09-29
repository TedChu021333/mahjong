"""提示模式入口。

    python -m hint                         # 即時擷取螢幕（遊戲最大化、1920x1080）
    python -m hint --image train/吃牌畫面.png
    python -m hint --video train/遊戲流程.mp4 --step 1
    python -m hint --no-overlay            # 建議只印在終端機，不顯示浮動小視窗

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

from hint.advisor import Readers, decide, describe, observe
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
        actuator=None, observe_fn=observe, clock=time.monotonic, show=None) -> None:
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
        if actuator is not None and actuator.follow_up(observation):
            print(f"[{stamp}] 選擇吃法", flush=True)
            acted_for, acted_at, retries = observation, clock(), 0
            continue
        action = decide(observation)
        advice = describe(action) if action is not None else None
        if advice is not None and advice != last_advice:
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
    parser.add_argument("--auto", action="store_true", help="自動點擊（目前停用，見說明）")
    parser.add_argument("--no-overlay", action="store_true", help="不顯示浮動小視窗")
    args = parser.parse_args()
    readers = Readers()
    if args.image:
        observation = observe(read_image(args.image), readers)
        action = decide(observation) if observation else None
        print(describe(action) if action else "畫面無法辨識或不需要做決定")
    elif args.video:
        run(video_frames(args.video, args.step, args.start), readers)
    elif args.auto:
        # 目前支援的「經典版」桌面是與真人配桌的場（畫面顯示真人即時配對、
        # 對手有玩家名稱與照片頭像），在這裡自動打牌等於對真人使用外掛，
        # 因此不提供。control.actuator 保留給確認沒有真人的場景或自己的模擬器。
        raise SystemExit("自動模式已停用：目前的桌面版型是與真人對戰的場，只提供提示模式。")
    elif args.no_overlay:
        print("提示模式啟動，Ctrl+C 結束", flush=True)
        run(screen_frames(args.interval), readers)
    else:
        from hint.overlay import Overlay

        print("提示模式啟動（浮動小視窗），Ctrl+C 結束", flush=True)
        overlay = Overlay()
        overlay.run(lambda: run(screen_frames(args.interval), readers, show=overlay.show))


if __name__ == "__main__":
    main()
