import pytest

from game.tiles import parse
from hint.advisor import Observation, Readers, _consistent_layout, advise, observe
from perception.capture import read_image
from perception.config import TEMPLATE_DIR, TRAIN_DIR
from perception.hand_reader import _cell_box, drawn_tile_box
from perception.table_reader import ClaimTile


def T(value: str) -> tuple[int, ...]:
    return tuple(parse(value))


def test_my_turn_suggests_a_discard_and_declares_when_tenpai():
    # 打掉 9s 後聽 3p、6p
    observation = Observation(T("123456789m123p45p11z9s"), None, None, ())
    assert observation.my_turn and observation.meld_count == 0
    assert advise(observation) == "打 九條，並按「聽」"


def test_self_drawn_win_is_suggested():
    observation = Observation(T("123456789m123456p11z"), None, None, ())
    assert advise(observation) == "自摸胡！"


def test_claim_suggestions_follow_buttons_and_source():
    hand = T("11m123456789p139s57z")  # 16 張；碰一萬不會讓向聽變差
    pung = Observation(hand, frozenset({"pung"}), ClaimTile(parse("1m")[0], 0.9, "right"), ())
    assert advise(pung) == "碰 一萬"
    win = Observation(T("123456789m123456p1z"), frozenset({"win"}),
                      ClaimTile(parse("1z")[0], 0.9, "left"), ())
    assert advise(win) == "胡！"


def test_no_advice_while_waiting_without_buttons():
    assert advise(Observation(T("123456789m123456p1z"), None, None, ())) is None


def test_claim_score_does_not_affect_frame_stability():
    a = Observation(T("1m"), frozenset(), ClaimTile(0, 0.81, "left"), ())
    b = Observation(T("1m"), frozenset(), ClaimTile(0, 0.83, "left"), ())
    assert a == b


def test_layout_rejects_partially_hidden_hands():
    grid = [_cell_box(index) for index in reversed(range(16))]
    assert _consistent_layout(grid[3:])                        # 1 組副露，等別人打牌
    assert _consistent_layout(grid[3:] + [drawn_tile_box()])   # 1 組副露，輪到自己
    assert not _consistent_layout(grid[:11])                   # 右邊被動畫擋住
    assert not _consistent_layout(grid[11:])                   # 左邊被「放槍」字樣擋住


@pytest.mark.skipif(
    not (TRAIN_DIR / "吃牌畫面.png").exists() or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_real_screenshot_end_to_end():
    readers = Readers()
    observation = observe(read_image(TRAIN_DIR / "吃牌畫面.png"), readers)
    assert advise(observation) == "吃，組成 六萬七萬八萬"


@pytest.mark.skipif(
    not (TRAIN_DIR / "缺的牌1.png").exists() or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_swap_screen_is_recognised_with_selected_tiles():
    readers = Readers()
    selecting = observe(read_image(TRAIN_DIR / "缺的牌1.png"), readers)
    assert selecting.swap_prompt is True
    assert [i for i, up in enumerate(selecting.raised) if up] == [11, 12, 13]
    assert advise(selecting) == "換三張：換 東、南、西"
    waiting = observe(read_image(TRAIN_DIR / "缺的牌5.png"), readers)
    assert waiting.swap_prompt is None
