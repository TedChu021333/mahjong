import pytest

from game.tiles import parse
from hint.advisor import Observation, Readers, _consistent_layout, advise, observe
from game.rules import ActionType
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
def test_real_screenshot_end_to_end(monkeypatch):
    import agent.rule_agent

    monkeypatch.setattr(agent.rule_agent, "CLAIM_MODE", "loose")  # 這手碰了只是進張變多
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
    import agent.rule_agent
    import hint.__main__ as hint_main

    monkeypatch.setattr(agent.rule_agent, "CLAIM_MODE", "loose")  # 測流程：這手照舊標準會碰

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


def test_loop_saves_a_screenshot_when_win_is_lit_but_not_chosen(monkeypatch):
    import hint.__main__ as hint_main

    waiting = Observation(T("11223789m99m4p56779s"), frozenset({"pung", "win"}),
                          ClaimTile(parse("2m")[0], 0.9, "left"), ())
    monkeypatch.setattr(hint_main, "decide", lambda observation: None)  # 模擬 AI 沒選胡
    saved, lines = [], []
    hint_main.run(((str(i), "frame") for i in range(4)), readers=None,
                  observe_fn=lambda frame, readers: waiting,
                  save_result=lambda frame, prefix="": saved.append(prefix) or "x.png",
                  log=lambda line, detail=None: lines.append((line, detail)))
    assert saved == ["未胡_"]  # 同一個畫面只存一次
    assert "手牌 一萬一萬二萬" in lines[0][1] and "win" in lines[0][1]


def test_game_log_writes_detail_to_file(tmp_path):
    from hint.__main__ import GameLog

    log = GameLog(tmp_path)
    log("[1] 打 九條", "手牌 九條")
    log("[2] 按下一場")
    text = log.path.read_text(encoding="utf-8")
    assert "[1] 打 九條　（手牌 九條）" in text and "[2] 按下一場" in text


def test_hint_mode_records_settlement_without_clicking():
    import hint.__main__ as hint_main
    from perception.continue_button import ContinueButton

    button = ContinueButton("小結算繼續", (1459, 977))
    saved = []
    hint_main.run(((str(i), button) for i in range(10)), readers=None,
                  observe_fn=lambda frame, readers: None, find_continue=lambda frame: frame,
                  save_result=lambda frame, prefix="": saved.append(frame) or "x.png",
                  log=lambda line, detail=None, echo=True: None)
    assert saved == [button]  # 沒有 actuator 也不會出錯，只存一次


def test_claim_tile_is_remembered_when_the_circle_is_gone(monkeypatch):
    import hint.__main__ as hint_main

    # 影片 21.12.08：下家打一萬，圓框 0.7 秒後縮進牌河，「碰」按鈕還亮 5 秒
    hand = T("11m234567p123456s45z")
    circle = Observation(hand, None, ClaimTile(parse("1m")[0], 0.85, "right"), ())
    prompt = Observation(hand, frozenset({"pung"}), None, ())
    advice = []
    now = [0.0]

    def frames():
        for step, screen in enumerate([circle, circle] + [prompt] * 4):
            now[0] = step * 0.3
            yield str(step), screen

    hint_main.run(frames(), readers=None, observe_fn=lambda frame, readers: frame,
                  clock=lambda: now[0], show=advice.append,
                  log=lambda line, detail=None, echo=True: None)
    assert [a for a in advice if a] and all("碰" in a or "不要" in a for a in advice if a)


def test_opponent_information_reaches_the_rule_agent():
    from hint.advisor import build_state

    observation = Observation(T("1479m258p369s1234567z"), None, None, (),
                              discards=((), (), tuple(parse("9m1z")), ()),
                              melds=(0, 0, 2, 0), declared=frozenset({"top"}))
    state = build_state(observation)
    assert state.declared == [False, False, True, False]
    assert len(state.players[2].melds) == 2 and state.players[2].discards == parse("9m1z")
    assert advise(observation) in ("打 九萬", "打 東")  # 對家聽牌，棄胡打現物


def test_ev_policy_can_be_selected():
    import pytest
    from hint.advisor import use_policy

    observation = Observation(T("123456789m12345p9s17z"), None, None, ())
    try:
        use_policy("ev")
        assert advise(observation).startswith("打 ")
    finally:
        use_policy("rule")
    with pytest.raises(ValueError):
        use_policy("nope")


def test_wall_is_estimated_from_discards_and_claims():
    from hint.advisor import build_state, estimate_wall

    fresh = Observation(T("123456789m12345p9s17z"), None, None, ())
    assert estimate_wall(fresh) == 60 and build_state(fresh).drawable == 60
    later = Observation(T("123456789m12345p9s17z"), None, None, (),
                        discards=(tuple(range(8)), tuple(range(9)), tuple(range(9)), tuple(range(9))),
                        melds=(0, 1, 0, 1))
    assert estimate_wall(later) == 60 - 35 + 2


def test_added_kong_when_the_kong_button_is_lit():
    from hint.advisor import decide

    # 摸進 5z（碰過的中）：「槓」亮著、手上沒有 4 張一樣的 → 加槓
    hand = T("123456789m123p9s5z")
    assert len(hand) == 14
    observation = Observation(hand, frozenset({"kong"}), None, ())
    action = decide(observation)
    assert action.kind == ActionType.KONG and action.tile == parse("5z")[0]
    # 沒亮就照常打牌
    assert decide(Observation(hand, None, None, ())).kind != ActionType.KONG


def test_loop_survives_errors_and_reports_stuck_screens(monkeypatch):
    import hint.__main__ as hint_main

    hand = T("123456789m12345p9s17z")
    screens = ["boom"] + [Observation(hand, None, None, ())] * 12
    saved, lines = [], []

    def observe_fn(frame, readers):
        if frame == "boom":
            raise RuntimeError("辨識出錯")
        return frame

    class StuckActuator:
        def follow_up(self, observation):
            return False

        def perform(self, action, observation):
            return True  # 點了但遊戲沒反應

    now = [0.0]

    def frames():
        for step, screen in enumerate(screens):
            now[0] = step * 1.0
            yield str(step), screen

    hint_main.run(frames(), readers=None, actuator=StuckActuator(), observe_fn=observe_fn,
                  clock=lambda: now[0], save_result=lambda frame, prefix="": saved.append(prefix) or "x",
                  log=lambda line, detail=None, echo=True: lines.append(line))
    assert saved == ["錯誤_", "卡住_"]
    assert any("程式錯誤" in line for line in lines)


def test_long_unreadable_stretch_saves_one_screenshot():
    import hint.__main__ as hint_main

    hand = T("123456789m12345p9s17z")
    screens = [Observation(hand, None, None, ())] * 2 + [None] * 30
    saved = []
    now = [0.0]

    def frames():
        for step, screen in enumerate(screens):
            now[0] = step * 0.5
            yield str(step), screen

    hint_main.run(frames(), readers=None, observe_fn=lambda frame, readers: frame,
                  clock=lambda: now[0], save_result=lambda frame, prefix="": saved.append(prefix) or "x",
                  log=lambda line, detail=None, echo=True: None)
    assert saved == ["讀不到_"]


def test_settlement_does_not_count_as_unreadable():
    import hint.__main__ as hint_main
    from perception.continue_button import ContinueButton

    hand = T("123456789m12345p9s17z")
    button = ContinueButton("小結算繼續", (1459, 977))
    screens = [Observation(hand, None, None, ())] * 2 + [button] * 3 + [None] * 30
    saved = []
    now = [0.0]

    def frames():
        for step, screen in enumerate(screens):
            now[0] = step * 0.5
            yield str(step), screen

    hint_main.run(frames(), readers=None,
                  observe_fn=lambda frame, readers: frame if isinstance(frame, Observation) else None,
                  find_continue=lambda frame: frame if isinstance(frame, ContinueButton) else None,
                  clock=lambda: now[0], save_result=lambda frame, prefix="": saved.append(prefix) or "x",
                  log=lambda line, detail=None, echo=True: None)
    assert "讀不到_" not in saved


def test_win_is_pressed_even_when_the_hand_is_covered():
    import hint.__main__ as hint_main
    from perception.config import ACTION_BUTTON_CENTERS

    # 摸牌的手蓋住手牌、「胡」亮著、倒數只剩 1 秒（實戰截圖）
    clicks, shown = [], []

    class Recorder:
        def click(self, point):
            clicks.append(point)

        def follow_up(self, observation):
            return False

    hint_main.run(((str(i), "covered") for i in range(3)), readers=None, actuator=Recorder(),
                  observe_fn=lambda frame, readers: None, find_win=lambda frame: True,
                  show=shown.append, clock=lambda: 0.0,
                  log=lambda line, detail=None, echo=True: None)
    assert clicks == [ACTION_BUTTON_CENTERS["win"]]  # 同一次只按一下
    assert shown == ["胡！"]


def test_win_button_reader_on_real_screens():
    import pytest
    from hint.__main__ import win_is_lit

    path = TRAIN_DIR / "經典_碰.png"
    if not path.exists():
        pytest.skip("缺少訓練截圖")
    assert not win_is_lit(read_image(path))  # 只有吃、碰亮著


def test_unknown_dealer_is_not_assumed_to_be_me():
    from hint.advisor import build_state

    state = build_state(Observation(T("123456789m12345p9s17z"), None, None, ()))
    assert state.known_dealer is None and state.dealer_streak == 0


def test_dealer_badge_reaches_the_state():
    from hint.advisor import build_state

    hand = T("123456789m12345p9s17z")
    assert build_state(Observation(hand, None, None, (), dealer="right")).known_dealer == 1
    assert build_state(Observation(hand, None, None, (), dealer="me")).known_dealer == 0


def test_dealer_badge_on_real_screens():
    from perception.table_reader import read_dealer

    expected = {"經典_碰": "left", "經典_牌河290": "me", "經典_b10": "top", "經典_小結算": None}
    for name, seat in expected.items():
        path = TRAIN_DIR / f"{name}.png"
        if path.exists():
            assert read_dealer(read_image(path)) == seat, name


def test_unhandled_claim_prompt_is_reported():
    import hint.__main__ as hint_main

    # 手牌讀不到、但吃碰按鈕亮著 2 秒：存一張截圖
    now = [0.0]

    def frames():
        for step in range(20):
            now[0] = step * 0.1
            yield str(step), "frame"

    def play(buttons):
        saved, lines = [], []
        hint_main.run(frames(), readers=None, observe_fn=lambda frame, readers: None,
                      find_buttons=lambda frame: buttons, clock=lambda: now[0],
                      save_result=lambda frame, prefix="": saved.append(prefix) or "x",
                      log=lambda line, detail=None, echo=True: lines.append(line))
        return saved, lines

    assert play(None) == ([], [])
    saved, lines = play(frozenset({"chow"}))
    assert saved == ["吃碰未處理_"] and "讀不到手牌" in lines[-1]


def test_popup_is_closed_in_auto_mode():
    import hint.__main__ as hint_main
    from perception.continue_button import ContinueButton

    clicks = []

    class Recorder:
        def click(self, point):
            clicks.append(point)

        def follow_up(self, observation):
            return False

    now = [0.0]

    def frames():
        for step in range(10):
            now[0] = step * 0.5
            yield str(step), "frame"

    hint_main.run(frames(), readers=None, actuator=Recorder(), observe_fn=lambda f, r: None,
                  find_popup=lambda frame: ContinueButton("神預測", (825, 270)),
                  clock=lambda: now[0], log=lambda line, detail=None, echo=True: None)
    assert clicks and set(clicks) == {(825, 270)} and len(clicks) <= 4  # 每 1.5 秒最多一次

def test_dealer_streak_reaches_the_state():
    from hint.advisor import STREAK_UNKNOWN_GUESS, build_state

    hand = T("123456789m12345p9s17z")
    state = build_state(Observation(hand, None, None, (), dealer="top", dealer_streak=2))
    assert state.known_dealer == 2 and state.dealer_streak == 2
    # 有數字但沒有模板（3 以上）
    state = build_state(Observation(hand, None, None, (), dealer="top", dealer_streak=None))
    assert state.dealer_streak == STREAK_UNKNOWN_GUESS


def test_dealer_streak_on_real_screens():
    from perception.table_reader import load_streak_templates, read_dealer_streak

    templates = load_streak_templates()
    if set(templates) != {1, 2}:
        pytest.skip("缺少連莊數字模板")
    expected = {"經典_上家第二欄吃": 1, "經典_上家第二欄吃2": 2, "經典_碰": 0, "經典_牌河290": 0}
    for name, streak in expected.items():
        path = TRAIN_DIR / f"{name}.png"
        if path.exists():
            assert read_dealer_streak(read_image(path), templates) == streak, name
    # 沒有模板時認不出數字
    path = TRAIN_DIR / "經典_上家第二欄吃.png"
    if path.exists():
        assert read_dealer_streak(read_image(path), {2: templates[2]}) is None


def test_autoplay_is_not_reported_as_unreadable():
    import hint.__main__ as hint_main

    hand = T("123456789m12345p9s17z")
    screens = [Observation(hand, None, None, ())] * 2 + [None] * 30
    saved = []
    now = [0.0]

    def frames():
        for step, screen in enumerate(screens):
            now[0] = step * 0.5
            yield str(step), screen

    hint_main.run(frames(), readers=None, observe_fn=lambda frame, readers: frame,
                  find_autoplay=lambda frame: frame is None, clock=lambda: now[0],
                  save_result=lambda frame, prefix="": saved.append(prefix) or "x",
                  log=lambda line, detail=None, echo=True: None)
    assert "讀不到_" not in saved


def test_exposed_tiles_count_as_visible():
    from agent.rule_agent import visible_tile_counts
    from hint.advisor import build_state

    # 自己吃過一組（用掉六萬八萬），推算對家碰了東：東另外兩張看得到，加上牌河一張、手上一張 = 4
    hand = T("123456789m123p9s1z")
    assert len(hand) == 14
    observation = Observation(hand, None, None, (),
                              discards=((), (), (), tuple(parse("1z"))),
                              melds=(0, 0, 1, 0),
                              exposed=(tuple(parse("68m")), (), tuple(parse("11z")), ()))
    visible = visible_tile_counts(build_state(observation), 0)
    assert visible[parse("1z")[0]] == 4 and visible[parse("6m")[0]] == 2
    # 推算有誤、會超過 4 張的丟掉；沒有副露組數可放的也丟掉
    wrong = Observation(hand, None, None, (), discards=((), (), (), tuple(parse("11z"))),
                        melds=(0, 0, 1, 0), exposed=((), tuple(parse("9s")), tuple(parse("11z")), ()))
    state = build_state(wrong)
    assert visible_tile_counts(state, 0)[parse("1z")[0]] == 4
    assert state.players[1].melds == []


def test_loop_saves_frames_from_before_a_claim(monkeypatch):
    import hint.__main__ as hint_main
    from perception.river_tracker import RiverEvent

    now = [0.0]
    hands = [Observation(T("11223789m99m4p56779s"), None, None, ()) for _ in range(8)]

    def frames():
        for i, observation in enumerate(hands):
            now[0] = i * 0.35
            yield str(i), observation

    class FakeRivers:
        def update(self, frame, observation):
            return [RiverEvent("claimed", "left", parse("7s")[0], (0, 0, 1, 1))] \
                if frame is hands[6] else []

        def discards(self):
            return ((), (), (), ())

        def meld_counts(self):
            return (0, 0, 0, 0)

        def exposed_tiles(self):
            return ((), (), (), ())

    monkeypatch.setattr(hint_main, "decide", lambda observation: None)
    saved = []
    hint_main.run(frames(), readers=None, observe_fn=lambda frame, readers: frame,
                  clock=lambda: now[0], rivers=FakeRivers(), show=lambda advice: None,
                  save_result=lambda frame, prefix="": saved.append((frame, prefix)) or prefix,
                  log=lambda *args, **kwargs: None)
    # 2.1 秒時發現被拿走：存 0.6、1.0 秒前（1.5、1.1 秒）最接近的畫面（1.4、1.05 秒）
    index = [next(i for i, hand in enumerate(hands) if hand is frame) for frame, _ in saved]
    assert index == [4, 3] and {prefix for _, prefix in saved} == {"吃碰動畫_left_7s_"}
