"""視覺模組的固定環境設定。"""
from pathlib import Path


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
TRAIN_DIR = Path("train")
"""訓練截圖、標註與辨識資料的統一根目錄。"""
TEMPLATE_DIR = TRAIN_DIR / "templates"
"""牌面模板：train/templates/<牌代碼，如 5m、1z>/*.png。"""


# 以下為「經典版」綠色桌面（由 train/Google Play Games 2026.09.29 影片量測）。
# 暗手牌排在固定格線上（副露在左、暗手牌緊接其後），每張間距固定。
HAND_TILE_PITCH = 102.33
HAND_GRID_RIGHT = 1713
"""最右邊一格的右緣。"""
HAND_TILE_TOP = 826
HAND_TILE_BOTTOM = 999
HAND_FACE_PROBE_ROWS = (991, 997)
"""判斷每一格是否有暗手牌的橫條：暗手牌白色牌面延伸到 y≈998，直立的副露牌
牌面只到 y≈960，花牌展示區在手牌上方，都不會落在這條。"""
MAX_HAND_TILES = 16
"""格線上的格數（不含摸進的第 17 張）。"""
DRAWN_TILE_REGION = (1746, 836, 1848, 998)
"""摸進的第 17 張在畫面上的位置。"""
DRAWN_TILE_CROP_X = (1745, 1849)
"""第 17 張辨識用的裁切左右界，與格線上的牌對齊方式相同；y 沿用 HAND_TILE_TOP/BOTTOM。"""
TILE_RAISE = 49
"""被選取的牌（如換三張）上移的距離，由牌面下緣量出；副露牌下緣只差
約 21px，只接受接近此值的位移，避免把副露當成上移的暗手牌。"""

# 吃碰槓聽胡按鈕列：可按的按鈕是亮的，不能按的變暗。
ACTION_BUTTON_CENTERS = {
    "chow": (380, 716),
    "pung": (614, 716),
    "kong": (846, 716),
    "declare": (1076, 716),
    "win": (1306, 716),
}
CANCEL_BUTTON_REGION = (1491, 671, 1581, 761)
"""「取消」紅色按鈕；出現代表按鈕列正在顯示。"""

# 別家打出、可以吃碰胡的牌會放大顯示，位置依打牌的人而定（自己打的牌在 (910,580)，不需要）。
CLAIM_TILE_POSITIONS = {
    "left": (612, 392),    # 上家
    "top": (912, 262),     # 對家
    "right": (1212, 392),  # 下家
}
CLAIM_TILE_SIZE = (106, 153)
CLAIM_TILE_MARGINS = (0, 0, 0, 0)
CLAIM_TEMPLATE_BOX = (1, 16, 103, 173)
"""放大牌只有牌面，比對時手牌模板只取這塊（網格搜尋：最低 0.874、差距 0.417）。"""

# 金牌面板：一排 4 格，超過換第二排（第二排位置尚未在影片中出現，沿用推估）。
GOLD_SLOT_ORIGIN = (421, 39)
GOLD_SLOT_PITCH = (57.7, 82)
GOLD_SLOT_SIZE = (55, 78)
GOLD_MAX_SLOTS = 8
GOLD_TEMPLATE_BOX = (8, 33, 99, 158)
"""金牌小圖的圖案佔滿牌面，手牌模板要往內裁（網格搜尋：最低 0.774、差距 0.353）。"""
GOLD_MIN_SCORE = 0.6

# 換三張畫面：「不換」出現代表正在選牌；「換牌」在選了牌之後才會亮（橘色）。
SWAP_KEEP_BUTTON = (756, 640)
SWAP_CONFIRM_BUTTON = (1164, 640)

# 多種吃法時的選項框（由 train/吃牌2.png 量測）：每框三張正面牌，依順子由小到大排列。
CHOW_PANEL_ROWS = (650, 790)
CHOW_PANEL_WIDTH = (230, 340)

# 打完後要按的「繼續」類按鈕：名稱 → 區域 (左, 上, 右, 下)。參考圖放 train/buttons/<名稱>.png，
# 用 `python -m perception.continue_button <截圖> <名稱>` 從截圖裁出；沒有參考圖的按鈕不會被點。
CONTINUE_BUTTONS = {
    "小結算繼續": (1279, 912, 1639, 1043),  # 每局結算畫面的藍色「繼續」（train/經典_小結算.png）
    # 流局時沒有小結算，直接出現「再贏一局 (倒數)」；只取「再贏一局」四個字（倒數每秒在變、
    # 右邊還有手指動畫）。參考圖取自 result/讀不到_20260930_134645.png。
    "再贏一局": (1480, 915, 1720, 990),
}
NEXT_GAME_REGION = (1457, 893, 1869, 1005)
"""按下小結算「繼續」後出現的下一場按鈕（使用者量測）；不做辨識，延遲後直接點。"""
NEXT_GAME_DELAY = 3.0
NEXT_GAME_CLICKS = 3
"""讀不到手牌時最多點幾次（每次間隔 NEXT_GAME_DELAY）；讀到手牌就停。"""
BUTTON_TEMPLATE_DIR = TRAIN_DIR / "buttons"
CONTINUE_MIN_SCORE = 0.9
CONTINUE_SEARCH_MARGIN = 10
"""參考圖在區域外擴這麼多像素的範圍內搜尋，容許畫面些微位移。"""

# 牌河：四家打出的牌在桌面中央，牌面正立、有透視（遠小近大，左右兩家有斜度）。
# 格位排列（何時換行、換欄）還沒在影片中完整出現，因此不寫死格線，而是在各家區域內
# 找「牌面上緣」：連續兩列亮白（>RIVER_FACE_MIN），上方一兩列較暗（<RIVER_GAP_MAX，
# 可能是綠色桌面、牌縫，或上一張疊牌的灰色底邊與牌面之間的暗線）。
RIVER_REGIONS = {
    "top": (600, 240, 1320, 370),    # 對家：一排由左往右
    "left": (440, 370, 760, 630),    # 上家：兩欄由上往下
    "right": (1100, 370, 1460, 630),  # 下家
    "me": (520, 630, 1320, 780),     # 自己：一排由左往右
}
RIVER_ROW_TOPS = {"top": (284,), "left": (397, 457, 519), "right": (397, 457, 519), "me": (651,)}
"""已觀察到的各列牌面上緣 y；不在這些列（±RIVER_ROW_TOLERANCE）的偵測多半是動畫或放大顯示。
影片中牌河最長 9 張，換行後的列還沒出現過，收集到畫面後要補上。"""
RIVER_ROW_TOLERANCE = 6
RIVER_FACE_HEIGHT = {"top": 50, "left": 52, "right": 52, "me": 66}
"""由牌面上緣往下取的高度（斜看的牌面在垂直方向被壓扁）。"""
RIVER_FACE_PAD = 5
"""上緣白色區段因圓角比牌面窄，左右各補這麼多像素。"""
RIVER_FACE_MIN = 235
RIVER_GAP_MAX = 215
RIVER_FACE_WIDTH = (40, 90)
RIVER_TEMPLATE_DIR = TRAIN_DIR / "river_templates"
"""牌河模板：train/river_templates/<牌代碼>/*.png（`python -m perception.river_reader` 產生）。"""

# 對手宣告聽牌後，他牌河旁會一直顯示「聽」圓形標記（上下跳動約 ±20px）。模板
# train/buttons/聽.png 取自 21.12.08 影片自己的標記中間的字。上家實測在 (400~440, 300~340)；
# 對家、下家的位置還沒出現過，先用較大範圍（避開畫面中央的「聽」字動畫與聽按鈕）。
DECLARE_REGIONS = {
    "left": (360, 240, 540, 440),
    "top": (450, 120, 1450, 290),
    "right": (1380, 240, 1720, 700),
}
DECLARE_TEMPLATE = TRAIN_DIR / "buttons" / "聽.png"
DECLARE_MIN_SCORE = 0.85
"""兩支影片：真正的標記 0.97~0.99，其他畫面最高 0.63。"""
