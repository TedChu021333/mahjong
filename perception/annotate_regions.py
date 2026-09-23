"""簡易圖片座標標註工具。

用法：
    python -m perception.annotate_regions train/完整正常畫面.png

操作：
    移動滑鼠：查看座標
    左鍵點擊兩次：標記矩形左上與右下角
    c：複製目前矩形座標
    r：重新開始
    q 或 Esc：離開
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


class RegionAnnotator:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.root = tk.Tk()
        self.root.title(f"座標標註：{image_path.name} ({self.image.width}x{self.image.height})")
        self.canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.status = tk.Label(self.root, anchor=tk.W)
        self.status.pack(fill=tk.X)
        self.points: list[tuple[int, int]] = []
        self.rectangle = None
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Key>", self.on_key)

    def on_motion(self, event: tk.Event) -> None:
        self.status.config(text=f"游標座標：x={event.x}, y={event.y}")

    def on_click(self, event: tk.Event) -> None:
        self.points.append((event.x, event.y))
        if len(self.points) == 2:
            (x1, y1), (x2, y2) = self.points
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            self.rectangle = (x1, y1, x2, y2)
            self.canvas.create_rectangle(*self.rectangle, outline="red", width=2)
            self.status.config(text=f"目前矩形：{self.rectangle}，按 c 複製、r 重設")

    def on_key(self, event: tk.Event) -> None:
        if event.keysym.lower() == "c" and self.rectangle:
            self.root.clipboard_clear()
            self.root.clipboard_append(",".join(map(str, self.rectangle)))
            messagebox.showinfo("已複製", f"座標：{self.rectangle}")
        elif event.keysym.lower() == "r":
            self.points.clear()
            self.rectangle = None
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        elif event.keysym.lower() in {"q", "escape"}:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    if not args.image.exists():
        parser.error(f"找不到圖片：{args.image}")
    RegionAnnotator(args.image).run()


if __name__ == "__main__":
    main()
