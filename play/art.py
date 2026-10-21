"""人機對戰視窗的圖（PIL）：牌、牌背、按鈕、背景、面板。

牌面用 `train/templates/<牌>/` 裡遊戲本身的手牌模板（34 種都有、不進版控）；花牌沒有模板，
沒有模板的電腦也一樣，用字畫。所有圖都依參數快取，視窗重畫時不重算。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat

from game.tiles import FLOWER_NAMES, HONOR_NAMES, tile_code

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "train" / "templates"
FONT_FILES = ("C:/Windows/Fonts/msjhbd.ttc", "C:/Windows/Fonts/msjh.ttc")
RANKS = "一二三四五六七八九"
SUIT_NAMES = "萬筒條"
SUIT_COLORS = ((200, 30, 40), (25, 90, 180), (30, 125, 60))
HONOR_COLORS = ((30, 30, 30),) * 4 + ((200, 30, 40), (30, 125, 60), (25, 90, 180))
FLOWER_COLOR = (140, 40, 160)
FACE = (250, 248, 240)
BODY = (34, 128, 84)
"""牌底部露出的綠色牌身。"""
GOLD = (245, 190, 50)
SHADOW = 4
"""陰影多留的邊（右、下）。"""


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.ImageFont:
    for path in FONT_FILES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=None)
def template(tile: int) -> Image.Image | None:
    """遊戲的牌面圖（不含變暗的樣本）；沒有就 None。"""
    if tile >= 34:
        return None
    folder = TEMPLATE_DIR / tile_code(tile)
    files = sorted(path for path in folder.glob("*.png") if "變暗" not in path.name) if folder.is_dir() else []
    images = []
    for path in files:
        try:
            images.append(Image.open(path).convert("RGB"))
        except OSError:
            continue
    # 模板是不同時間的截圖，亮度不一（有些剛好在動畫中偏暗）：挑最亮的一張
    return max(images, key=lambda image: sum(ImageStat.Stat(image).mean), default=None)


def _mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=255)
    return mask


def _drawn_face(tile: int, w: int, h: int) -> Image.Image:
    """沒有模板時用字畫牌面。"""
    face = Image.new("RGB", (w, h), FACE)
    draw = ImageDraw.Draw(face)
    if tile < 27:
        suit, rank = divmod(tile, 9)
        color = SUIT_COLORS[suit]
        draw.text((w / 2, h * 0.3), RANKS[rank], fill=(25, 25, 25), font=font(int(h * 0.34)), anchor="mm")
        draw.text((w / 2, h * 0.7), SUIT_NAMES[suit], fill=color, font=font(int(h * 0.34)), anchor="mm")
    elif tile < 34:
        if tile == 33:
            draw.rectangle((w * 0.2, h * 0.2, w * 0.8, h * 0.8), outline=HONOR_COLORS[6], width=max(2, w // 18))
        else:
            draw.text((w / 2, h / 2), HONOR_NAMES[tile - 27], fill=HONOR_COLORS[tile - 27],
                      font=font(int(min(w * 0.8, h * 0.5))), anchor="mm")
    else:
        draw.text((w / 2, h * 0.4), FLOWER_NAMES[tile - 34], fill=FLOWER_COLOR,
                  font=font(int(min(w * 0.75, h * 0.42))), anchor="mm")
        draw.text((w / 2, h * 0.78), "花", fill=FLOWER_COLOR, font=font(int(h * 0.2)), anchor="mm")
    return face


def _with_shadow(body: Image.Image, radius: int) -> Image.Image:
    w, h = body.size
    out = Image.new("RGBA", (w + SHADOW, h + SHADOW), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 110))
    blurred = Image.new("RGBA", out.size, (0, 0, 0, 0))
    blurred.paste(shadow, (SHADOW, SHADOW), _mask((w, h), radius))
    out = Image.alpha_composite(out, blurred.filter(ImageFilter.GaussianBlur(2)))
    out.alpha_composite(body)
    return out


@lru_cache(maxsize=None)
def tile_image(tile: int, w: int, h: int, dim: bool = False, gold: bool = False,
               outline: tuple[int, int, int] | None = None) -> Image.Image:
    """一張直立的牌（含陰影，尺寸 w+SHADOW × h+SHADOW）。"""
    radius = max(3, w // 9)
    thickness = max(3, h // 14)
    body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    body.paste(Image.new("RGBA", (w, h), BODY + (255,)), (0, 0), _mask((w, h), radius))
    face_size = (w, h - thickness)
    source = template(tile)
    face = source.resize(face_size, Image.LANCZOS) if source is not None else _drawn_face(tile, *face_size)
    if gold:
        face = Image.blend(face, Image.new("RGB", face_size, (255, 214, 90)), 0.28)
    if dim:
        face = ImageEnhance.Brightness(face).enhance(0.6)
    body.paste(face, (0, 0), _mask(face_size, radius))
    draw = ImageDraw.Draw(body)
    if gold:
        draw.rounded_rectangle((1, 1, w - 2, h - thickness - 1), radius, outline=GOLD, width=2)
    if outline is not None:
        draw.rounded_rectangle((0, 0, w - 1, h - 1), radius, outline=outline, width=3)
    return _with_shadow(body, radius)


@lru_cache(maxsize=None)
def back_image(w: int, h: int) -> Image.Image:
    """牌背（綠色、有光澤）。"""
    radius = max(3, w // 6)
    body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gradient = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(gradient)
    for y in range(h):
        t = y / max(1, h - 1)
        draw.line((0, y, w, y), fill=(int(70 - 30 * t), int(170 - 60 * t), int(120 - 40 * t)))
    body.paste(gradient, (0, 0), _mask((w, h), radius))
    ImageDraw.Draw(body).rounded_rectangle((0, 0, w - 1, h - 1), radius, outline=(190, 235, 210), width=1)
    return _with_shadow(body, radius)


@lru_cache(maxsize=None)
def button_image(w: int, h: int, color: tuple[int, int, int]) -> Image.Image:
    """圓角、上亮下暗的按鈕底（字由視窗畫）。"""
    body = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(body)
    for y in range(h):
        t = y / max(1, h - 1)
        k = 1.25 - 0.45 * t
        draw.line((0, y, w, y), fill=tuple(min(255, int(c * k)) for c in color))
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(body, (0, 0), _mask((w, h), h // 2))
    ImageDraw.Draw(out).rounded_rectangle((0, 0, w - 1, h - 1), h // 2, outline=(255, 255, 255, 160), width=2)
    return _with_shadow(out, h // 2)


@lru_cache(maxsize=None)
def panel_image(w: int, h: int, alpha: int = 190, border: tuple[int, int, int] | None = None) -> Image.Image:
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), 18, fill=(8, 28, 20, alpha),
                           outline=(border + (255,)) if border else (255, 255, 255, 40), width=2)
    return out


@lru_cache(maxsize=None)
def felt_image(w: int, h: int) -> Image.Image:
    """牌桌絨布：中間亮、四周暗。"""
    small = Image.new("RGB", (w // 8, h // 8))
    sw, sh = small.size
    pixels = small.load()
    for y in range(sh):
        for x in range(sw):
            d = (((x - sw / 2) / (sw / 2)) ** 2 + ((y - sh / 2) / (sh / 2)) ** 2) ** 0.5
            k = max(0.0, 1 - 0.55 * d)
            pixels[x, y] = (int(12 + 30 * k), int(70 + 75 * k), int(45 + 45 * k))
    return small.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))


@lru_cache(maxsize=None)
def avatar_image(size: int, text: str, ring: tuple[int, int, int]) -> Image.Image:
    """玩家頭像：圓形、中間是門風字。"""
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.ellipse((2, 2, size - 3, size - 3), fill=(245, 236, 210), outline=ring, width=4)
    draw.text((size / 2, size / 2), text, fill=(60, 40, 20), font=font(int(size * 0.5)), anchor="mm")
    return out


@lru_cache(maxsize=None)
def badge_image(text: str, color: tuple[int, int, int], size: int = 30) -> Image.Image:
    """莊、聽等小方章。"""
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), 6, fill=color, outline=(255, 255, 255), width=2)
    draw.text((size / 2, size / 2), text, fill=(255, 255, 255), font=font(int(size * 0.62)), anchor="mm")
    return out
