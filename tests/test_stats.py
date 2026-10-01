import cv2
import pytest

from eval.stats import MY_NAME, ROW_TOPS, Row, classify, load_digits, read_row, report, Round, Hand
from perception.capture import read_image
from perception.config import TRAIN_DIR


def test_classify_outcomes():
    def rows(me, *others):
        return [Row(me, False, True)] + [Row(amount, False, False) for amount in others]

    assert classify(rows(2100, -700, -700, -700)) == (2100, "自摸")
    assert classify(rows(350, -350, None, None)) == (350, "胡牌")
    assert classify(rows(-350, 350, None, None)) == (-350, "放槍")
    assert classify(rows(-650, 1950, -650, -650)) == (-650, "被自摸")
    assert classify(rows(None, 300, -300, None)) == (0, "沒輸贏")
    assert classify([Row(100, False, False)] * 4) == (0, "未知")


def test_report_counts():
    rounds = [Round([Hand("20260930_180000", 500, "胡牌"), Hand("20260930_180500", -300, "放槍")]),
              Round([Hand("20260930_190000", 0, "沒輸贏")])]
    text = report(rounds)
    # 贏的 500 抽 15% 實得 425
    assert "共 2 圈、3 局，實際輸贏 +125（畫面金額合計 +200" in text
    assert "第 1 圈 18:00–18:05  2 局  +125" in text


def test_rake_rounds_down_like_the_game():
    # 實測：+130 → 金幣 +111、+310 → +264、+3900 → +3315；輸的照扣
    assert [Hand("x", a, "").net for a in (130, 310, 3900, -700)] == [111, 264, 3315, -700]


@pytest.mark.parametrize("name, amounts, me", [
    ("經典_結算自摸", [-700, -700, -700, 2100], 3),   # 18:41 自己自摸 +2100
    ("經典_結算放槍", [-350, None, 350, None], 0),    # 18:19 自己放槍 -350
])
def test_real_settlement_screens(name, amounts, me):
    path = TRAIN_DIR / f"{name}.png"
    if not path.exists():
        pytest.skip("缺少訓練截圖")
    image = read_image(path)
    my_name = cv2.cvtColor(read_image(MY_NAME), cv2.COLOR_BGR2GRAY)
    rows = [read_row(image, top, load_digits(), my_name) for top in ROW_TOPS]
    assert [row.amount for row in rows] == amounts
    assert [row.is_me for row in rows].index(True) == me and sum(r.is_me for r in rows) == 1
