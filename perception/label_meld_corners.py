"""點選工具：在流局亮牌的截圖上，標上家、下家那一排牌的牌面四個角（給副露辨識做透視校正）。

    .venv-train/Scripts/python -m perception.label_meld_corners result/局末_20261022_031435.png

照視窗上方的提示依序點：每排標最上面、最下面兩張，每張依「左上 → 右上 → 右下 → 左下」點牌面（白色那面，
不含下面露出的厚度）的四個角。滑鼠旁邊有放大鏡（原圖 4 倍）方便點準。右鍵或 Backspace 取消上一點。
全部標完存到 train/meld_corners.json。
一排牌在桌上是一條直線（只是有透視），有頭尾兩張就能把整排拉直；副露與手牌之間、副露與副露之間有空隙
（使用者 10/22 指出），不能假設平均排列，拉直後由程式找空隙切開。
"""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

OUT = Path(__file__).resolve().parent.parent / "train" / "meld_corners.json"
SEATS = (("left", "上家"), ("right", "下家"))
TILES = (("first", "最上面（離你最遠）那張"), ("last", "最下面（離你最近）那張"))
CORNERS = ("左上", "右上", "右下", "左下")
ZOOM = 4
LENS = 40
"""放大鏡取原圖多少像素見方。"""


class CornerSession:
    """依序收集各排各張牌的四個角（原圖座標）。"""

    def __init__(self) -> None:
        self.steps = [(seat, tile, corner) for seat, _ in SEATS for tile, _ in TILES for corner in CORNERS]
        self.points: list[tuple[int, int]] = []

    @property
    def done(self) -> bool:
        return len(self.points) >= len(self.steps)

    def prompt(self) -> str:
        if self.done:
            return "全部標完了，已存檔，可以關閉視窗"
        seat, tile, corner = self.steps[len(self.points)]
        seat_name = dict(SEATS)[seat]
        tile_name = dict(TILES)[tile]
        return f"{seat_name}：{tile_name}的「{corner}」角（第 {len(self.points) + 1}/{len(self.steps)} 點）"

    def add(self, x: int, y: int) -> None:
        if not self.done:
            self.points.append((x, y))

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def to_json(self) -> dict:
        result: dict = {}
        for (seat, tile, corner), point in zip(self.steps, self.points):
            result.setdefault(seat, {}).setdefault(tile, []).append(list(point))
        return result


class App:
    def __init__(self, root: tk.Tk, image_path: Path, out: Path) -> None:
        self.root, self.out = root, out
        self.image = Image.open(image_path).convert("RGB")
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        self.scale = min(1.0, (screen_w - 40) / self.image.width, (screen_h - 160) / self.image.height)
        shown = self.image.resize((int(self.image.width * self.scale), int(self.image.height * self.scale)))
        self.photo = ImageTk.PhotoImage(shown)
        self.session = CornerSession()
        self.label = tk.Label(root, font=("Microsoft JhengHei", 16, "bold"), fg="#c62828")
        self.label.pack(fill="x")
        self.canvas = tk.Canvas(root, width=shown.width, height=shown.height, highlightthickness=0, cursor="tcross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.lens_photo = None
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", lambda event: self.undo())
        self.canvas.bind("<Motion>", self.motion)
        root.bind("<BackSpace>", lambda event: self.undo())
        self.refresh()

    def original(self, event) -> tuple[int, int]:
        return round(event.x / self.scale), round(event.y / self.scale)

    def click(self, event) -> None:
        x, y = self.original(event)
        self.session.add(x, y)
        if self.session.done:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.out.write_text(json.dumps(self.session.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        self.refresh()

    def undo(self) -> None:
        self.session.undo()
        self.refresh()

    def motion(self, event) -> None:
        x, y = self.original(event)
        box = (x - LENS // 2, y - LENS // 2, x + LENS // 2, y + LENS // 2)
        lens = self.image.crop(box).resize((LENS * ZOOM, LENS * ZOOM), Image.NEAREST)
        self.lens_photo = ImageTk.PhotoImage(lens)
        self.canvas.delete("lens")
        lx = event.x + 30 if event.x + 30 + LENS * ZOOM < self.canvas.winfo_width() else event.x - 30 - LENS * ZOOM
        ly = max(0, event.y - LENS * ZOOM // 2)
        self.canvas.create_image(lx, ly, image=self.lens_photo, anchor="nw", tags="lens")
        c = LENS * ZOOM // 2
        self.canvas.create_line(lx + c, ly, lx + c, ly + 2 * c, fill="red", tags="lens")
        self.canvas.create_line(lx, ly + c, lx + 2 * c, ly + c, fill="red", tags="lens")
        self.canvas.create_rectangle(lx, ly, lx + 2 * c, ly + 2 * c, outline="yellow", width=2, tags="lens")
        self.canvas.create_text(lx + 4, ly + 4, text=f"{x}, {y}", anchor="nw", fill="yellow",
                                font=("Consolas", 12, "bold"), tags="lens")

    def refresh(self) -> None:
        self.label.config(text=self.session.prompt())
        self.canvas.delete("mark")
        points = self.session.points
        for index, (x, y) in enumerate(points):
            sx, sy = x * self.scale, y * self.scale
            self.canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill="red", outline="white", tags="mark")
            if index % 4 == 3:  # 一張牌四個角都點了：連成框
                quad = [coord * self.scale for point in points[index - 3:index + 1] for coord in point]
                self.canvas.create_polygon(*quad, outline="#00e676", fill="", width=2, tags="mark")


def main() -> None:
    parser = argparse.ArgumentParser(description="標上家、下家那排牌的牌面四個角")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    root = tk.Tk()
    root.title("標副露位置（右鍵／Backspace 取消上一點）")
    App(root, args.image, args.out)
    root.mainloop()


if __name__ == "__main__":
    main()
