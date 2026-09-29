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
    not (TRAIN_DIR / "經典_碰.png").exists() or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_real_screenshot_end_to_end():
    readers = Readers()
    observation = observe(read_image(TRAIN_DIR / "經典_碰.png"), readers)
    assert advise(observation) == "碰 二萬"


@pytest.mark.skipif(
    not (TRAIN_DIR / "經典_換三張.png").exists() or not TEMPLATE_DIR.exists(),
    reason="缺少訓練截圖或模板",
)
def test_swap_screen_is_recognised_with_selected_tiles():
    readers = Readers()
    selecting = observe(read_image(TRAIN_DIR / "經典_換三張.png"), readers)
    assert selecting.swap_prompt is True
    assert [i for i, up in enumerate(selecting.raised) if up] == [14, 15]
    assert advise(selecting).startswith("換三張：換 西、白")
    waiting = observe(read_image(TRAIN_DIR / "經典_碰.png"), readers)
    assert waiting.swap_prompt is None


def test_overlay_colors_and_position_file(tmp_path):
    from hint.overlay import DEFAULT_POSITION, color_for, load_position, save_position
    assert color_for("胡！") != color_for("打 九筒") != color_for("碰 二萬")
    assert color_for("打 四筒，並按「聽」") == color_for("聽")
    path = tmp_path / "pos.txt"
    assert load_position(path) == DEFAULT_POSITION
    save_position(12, 34, path)
    assert load_position(path) == (12, 34)


def test_loop_remembers_claim_and_forbids_tiles_on_the_next_discard(monkeypatch):
    import hint.__main__ as hint_main

    # 影片 55.5→57.5 秒：上家打二萬，碰；碰完遊戲把最右邊的九條移到第 17 張位置
    waiting = Observation(T("11223789m99m4p56779s"), frozenset({"pung", "chow"}),
                          ClaimTile(parse("2m")[0], 0.9, "left"), ())
    after = Observation(T("113789m99m4p56779s"), None, None, ())
    unrelated = Observation(T("113789m99m4p56789s"), None, None, ())
    seen = []
    real_decide = hint_main.decide
    monkeypatch.setattr(hint_main, "decide",
                        lambda observation: seen.append(observation) or real_decide(observation))
    advice = []
    frames = [waiting, waiting, after, after, unrelated, unrelated]
    hint_main.run(((str(i), f) for i, f in enumerate(frames)), readers=None,
                  observe_fn=lambda frame, readers: frame, show=advice.append)
    assert advice[0] == "碰 二萬"
    assert seen[1].claimed and seen[1].forbidden == (parse("2m")[0],)
    assert not seen[2].claimed  # 手牌對不上就不是剛碰完


def test_claimed_observation_never_suggests_a_forbidden_tile():
    # 用 67s 吃 5s 後手上剩 5s、89s：平常打孤張五條就聽牌，但吃完不能打 5s、8s
    hand = T("111m222m333m44p5s89s")
    assert advise(Observation(hand, None, None, ())) == "打 五條，並按「聽」"
    forbidden = (parse("5s")[0], parse("8s")[0])
    claimed = Observation(hand, None, None, (), claimed=True, forbidden=forbidden)
    assert advise(claimed) == "打 九條"
