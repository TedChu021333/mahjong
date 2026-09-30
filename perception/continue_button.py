"""辨識打完後的「繼續」類按鈕（小結算的繼續、繼續下一場）。

每個按鈕用一張參考截圖做正規化相關比對；按鈕位置固定，只在它的區域附近搜尋。

    python -m perception.continue_button <截圖> <按鈕名稱>   # 從截圖裁出參考圖
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from perception.capture import read_image, write_image
from perception.config import (
    BUTTON_TEMPLATE_DIR,
    CONTINUE_BUTTONS,
    POPUP_CLOSERS,
    CONTINUE_MIN_SCORE,
    CONTINUE_SEARCH_MARGIN,
)


@dataclass(frozen=True)
class ContinueButton:
    name: str
    center: tuple[int, int]


def crop(image: np.ndarray, region: tuple[int, int, int, int], margin: int = 0) -> np.ndarray:
    left, top, right, bottom = region
    height, width = image.shape[:2]
    return image[max(0, top - margin):min(height, bottom + margin),
                 max(0, left - margin):min(width, right + margin)]


class ContinueButtons:
    def __init__(self, references: dict[str, np.ndarray]) -> None:
        self.references = {name: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                           for name, image in references.items()}

    @classmethod
    def from_directory(cls, directory: Path = BUTTON_TEMPLATE_DIR) -> "ContinueButtons":
        return cls({name: read_image(directory / f"{name}.png")
                    for name in CONTINUE_BUTTONS if (directory / f"{name}.png").exists()})

    def find(self, image: np.ndarray) -> ContinueButton | None:
        """同位置的按鈕（再贏一局、繼續戰鬥、破產後的再玩一局）字形相近，取分數最高的。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        best, best_score = None, CONTINUE_MIN_SCORE
        for name, reference in self.references.items():
            area = crop(gray, CONTINUE_BUTTONS[name], CONTINUE_SEARCH_MARGIN)
            if area.shape[0] < reference.shape[0] or area.shape[1] < reference.shape[1]:
                continue
            score = float(cv2.matchTemplate(area, reference, cv2.TM_CCOEFF_NORMED).max())
            if score >= best_score:
                best, best_score = name, score
        if best is None:
            return None
        left, top, right, bottom = CONTINUE_BUTTONS[best]
        return ContinueButton(best, ((left + right) // 2, (top + bottom) // 2))


class PopupClosers:
    """會蓋住牌桌的彈出面板：辨識到就回傳要點的位置（關閉鈕）。"""

    def __init__(self, references: dict[str, np.ndarray]) -> None:
        self.references = {name: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                           for name, image in references.items()}

    @classmethod
    def from_directory(cls, directory: Path = BUTTON_TEMPLATE_DIR) -> "PopupClosers":
        return cls({name: read_image(directory / f"{name}.png")
                    for name in POPUP_CLOSERS if (directory / f"{name}.png").exists()})

    def find(self, image: np.ndarray) -> ContinueButton | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for name, reference in self.references.items():
            region, click = POPUP_CLOSERS[name]
            area = crop(gray, region, CONTINUE_SEARCH_MARGIN)
            if area.shape[0] < reference.shape[0] or area.shape[1] < reference.shape[1]:
                continue
            if cv2.matchTemplate(area, reference, cv2.TM_CCOEFF_NORMED).max() >= CONTINUE_MIN_SCORE:
                return ContinueButton(name, click)
        return None


def main() -> None:
    regions = dict(CONTINUE_BUTTONS)
    regions.update({name: region for name, (region, _) in POPUP_CLOSERS.items()})
    if len(sys.argv) != 3 or sys.argv[2] not in regions:
        raise SystemExit(f"用法：python -m perception.continue_button <截圖> "
                         f"<{'|'.join(regions)}>")
    path = BUTTON_TEMPLATE_DIR / f"{sys.argv[2]}.png"
    write_image(path, crop(read_image(sys.argv[1]), regions[sys.argv[2]]))
    print(f"已存參考圖：{path}")


if __name__ == "__main__":
    main()
