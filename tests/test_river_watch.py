from game.tiles import parse
from hint.advisor import Observation
from hint.river_watch import RiverWatch, summary
from perception.river_reader import SEATS, RiverTile
from perception.river_tracker import CONFIRM_FRAMES
from perception.table_reader import ClaimTile


def P(value):
    return parse(value)[0]


class FakeReader:
    """依序回傳預先排好的牌河讀數。"""

    def __init__(self):
        self.current = {seat: [] for seat in SEATS}

    def read(self, frame):
        return {seat: list(tiles) for seat, tiles in self.current.items()}


def watch_with(reader):
    saved = []
    watch = RiverWatch(reader, save=lambda path, image: saved.append(path))
    return watch, saved


def frames(watch, observation, count=CONFIRM_FRAMES):
    events = []
    for _ in range(count):
        events += watch.update(None, observation)
    return events


def waiting(hand="11223789m99m4p56779s", claim=None):
    return Observation(tuple(parse(hand)), None, claim, ())


def test_own_discard_is_labelled_from_the_hand_difference(tmp_path):
    import numpy as np

    reader = FakeReader()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    watch.update(frame, Observation(tuple(parse("11223789m99m4p56779s9p")), None, None, ()))
    reader.current["me"] = [RiverTile(None, 0.2, (580, 651, 650, 717))]
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert watch.tracker.discards["me"] == [P("9p")]
    assert len(saved) == 1 and saved[0].parent.name == "9p"
    assert watch.discards()[0] == (P("9p"),)


def test_opponent_discard_is_labelled_from_the_enlarged_tile():
    import numpy as np

    reader = FakeReader()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    watch.update(frame, waiting(claim=ClaimTile(P("5z"), 0.9, "top")))
    reader.current["top"] = [RiverTile(P("6z"), 0.75, (660, 284, 717, 334))]  # 認錯
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert watch.tracker.discards["top"] == [P("5z")]
    assert [p.parent.name for p in saved] == ["5z"]
    assert watch.discards()[2] == (P("5z"),)
    assert "top 中" in summary(watch)


def test_known_tiles_are_not_saved_again():
    import numpy as np

    reader = FakeReader()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    watch.update(frame, waiting(claim=ClaimTile(P("1s"), 0.9, "right")))
    reader.current["right"] = [RiverTile(P("1s"), 0.95, (1151, 398, 1212, 450))]
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert saved == [] and watch.discards()[1] == (P("1s"),)


def test_new_hand_clears_everything():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["left"] = [RiverTile(P("3m"), 0.9, (569, 397, 630, 449))]
    frames(watch, waiting())
    watch.new_hand()
    assert watch.discards() == ((), (), (), ())


def left_river(*tiles):
    return [RiverTile(tile, 0.9, (569, 397 + 60 * i, 630, 449 + 60 * i)) for i, tile in enumerate(tiles)]


def test_claimer_is_the_next_one_to_discard():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["right"] = [RiverTile(P("5z"), 0.9, (1151, 398, 1212, 450))]
    frames(watch, waiting())                         # 下家打中，落進牌河
    reader.current["right"] = []
    frames(watch, waiting())                         # 中被拿走
    frames(watch, waiting(claim=ClaimTile(P("7s"), 0.9, "left")), 1)  # 上家接著打七條（圓框）
    assert watch.melds == {"me": 0, "right": 0, "top": 0, "left": 1}
    assert watch.meld_counts() == (0, 0, 0, 1)
    # 上家不是下家的下家，不能吃：一定是碰，另外兩張中也看得到了
    assert watch.exposed_tiles() == ((), (), (), (P("5z"), P("5z")))


def test_tile_that_comes_back_was_only_covered():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["left"] = left_river(P("7s"))
    frames(watch, waiting())
    reader.current["left"] = []
    frames(watch, waiting())                         # 看起來被拿走
    reader.current["left"] = left_river(P("7s"))
    frames(watch, waiting(), 1)                      # 其實只是被蓋住
    reader.current["top"] = [RiverTile(P("1z"), 0.9, (660, 284, 717, 334))]
    frames(watch, waiting())
    assert all(count == 0 for count in watch.melds.values())


def test_own_claims_are_not_counted():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["left"] = left_river(P("7s"))
    frames(watch, waiting())
    reader.current["left"] = []
    frames(watch, waiting())                         # 我們吃了上家的七條
    reader.current["me"] = [RiverTile(P("1z"), 0.9, (580, 651, 650, 717))]
    frames(watch, waiting("11223789m99m4p5s"))       # 手牌少 3 張：多一組副露
    assert all(count == 0 for count in watch.melds.values())


def test_hidden_claim_by_the_left_player():
    # 影片 21.12.08：下家的一萬被上家碰走，上家接著打的牌被押注面板蓋住，下一個看到的是自己出牌
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["right"] = [RiverTile(P("1m"), 0.9, (1151, 398, 1212, 450))]
    frames(watch, waiting())
    reader.current["right"] = []
    frames(watch, waiting())
    reader.current["me"] = [RiverTile(P("8m"), 0.9, (580, 651, 650, 717))]
    frames(watch, waiting())                         # 手牌張數沒變：自己沒吃碰
    assert watch.melds["left"] == 1
    assert watch.exposed["left"] == [P("1m"), P("1m")]


def test_credit_is_undone_when_the_tile_comes_back_later():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["left"] = left_river(P("4z"))
    frames(watch, waiting())
    reader.current["left"] = []
    frames(watch, waiting())                          # 北看起來被拿走
    reader.current["top"] = [RiverTile(P("1z"), 0.9, (660, 284, 717, 334))]
    frames(watch, waiting())                          # 對家出牌 → 先算給對家
    assert watch.melds["top"] == 1 and watch.exposed["top"] == [P("4z"), P("4z")]
    reader.current["left"] = left_river(P("4z"))
    frames(watch, waiting(), 1)                       # 北又回來：只是被蓋住
    assert watch.melds["top"] == 0 and watch.exposed["top"] == []


def test_claim_by_the_next_player_may_be_a_chow():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["top"] = [RiverTile(P("3m"), 0.9, (660, 284, 717, 334))]
    frames(watch, waiting())
    reader.current["top"] = []
    frames(watch, waiting())                          # 對家的三萬被拿走
    frames(watch, waiting(claim=ClaimTile(P("7s"), 0.9, "left")), 1)  # 上家（對家的下家）接著出牌
    assert watch.melds["left"] == 1 and watch.exposed["left"] == []  # 可能是吃，不知道另外兩張


def test_own_claim_tiles_are_exposed():
    watch, _ = watch_with(FakeReader())
    watch.own_claim((P("6m"), P("8m")))
    assert watch.exposed_tiles()[0] == (P("6m"), P("8m"))
    watch.new_hand()
    assert watch.exposed_tiles() == ((), (), (), ())


def watch_with_banner(reader, banners):
    """banners：每次 update 依序回傳的大字（用完後為 None）。"""
    queue = list(banners)
    watch = RiverWatch(reader, save=lambda path, image: None,
                       read_banner=lambda frame: queue.pop(0) if queue else None)
    return watch


def test_banner_tells_a_pung_by_the_next_player():
    # 下家的五萬被對家（下家的下家）拿走：只看出牌順序分不出吃碰，大字是「碰」
    reader = FakeReader()
    reader.current["right"] = [RiverTile(P("5m"), 0.9, (1151, 398, 1212, 450))]
    watch = watch_with_banner(reader, [None] * CONFIRM_FRAMES + [("top", "pung")])
    frames(watch, waiting(), CONFIRM_FRAMES)            # 下家打五萬
    reader.current["right"] = []
    frames(watch, waiting(), CONFIRM_FRAMES + 1)        # 大字出現、五萬被拿走
    assert watch.melds["top"] == 1 and watch.exposed["top"] == [P("5m"), P("5m")]


def test_banner_chow_counts_a_meld_without_known_tiles():
    reader = FakeReader()
    reader.current["right"] = [RiverTile(P("5m"), 0.9, (1151, 398, 1212, 450))]
    watch = watch_with_banner(reader, [None] * CONFIRM_FRAMES + [("top", "chow")])
    frames(watch, waiting(), CONFIRM_FRAMES)
    reader.current["right"] = []
    frames(watch, waiting(), CONFIRM_FRAMES + 1)
    assert watch.melds["top"] == 1 and watch.exposed["top"] == []


def test_frame_is_passed_to_the_banner_reader(monkeypatch):
    import hint.river_watch as river_watch

    seen = []
    monkeypatch.setattr(river_watch, "read_claim_banner", lambda frame: seen.append(frame))
    watch = RiverWatch(FakeReader(), save=lambda path, image: None)
    watch.update("frame", waiting())
    watch.update(None, waiting())  # 沒有畫面（測試）就不讀
    assert seen == ["frame"]


def test_own_discards_are_counted_from_the_hand_even_when_the_river_misses_them():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    turn = Observation(tuple(parse("11223789m99m4p56779s9p")), None, None, ())
    for _ in range(2):
        watch.update(None, turn)
        watch.update(None, waiting())  # 少一張：打出去了（牌河沒讀到）
    assert watch.my_discards == 2 and watch.discards()[0] == ()
    watch.new_hand()
    assert watch.my_discards == 0


def test_own_discard_is_relabelled_when_the_hand_change_arrives_late():
    # 10/18 03:24：牌河先確認自己的出牌（讀成二筒），手牌變化晚幾幀才看到
    import numpy as np

    reader = FakeReader()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    turn = Observation(tuple(parse("11223789m99m4p56779s9p")), None, None, ())
    reader.current["me"] = [RiverTile(P("2p"), 0.7, (580, 651, 650, 717))]
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, turn)  # 手牌還是打牌前的樣子
    assert watch.tracker.discards["me"] == [P("2p")]
    watch.update(frame, waiting())
    assert watch.tracker.discards["me"] == [P("9p")]
    assert len(saved) == 1 and saved[0].parent.name == "9p"


def test_tiles_discarded_after_a_declaration_are_safe_for_the_declarer():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["left"] = left_river(P("7s"))
    frames(watch, waiting())                         # 宣告前打的：不算
    declared = Observation(tuple(parse("11223789m99m4p56779s")), None, None, (),
                           declared=frozenset({"top"}))
    frames(watch, declared, 1)
    reader.current["right"] = [RiverTile(P("5z"), 0.9, (1151, 398, 1212, 450))]
    reader.current["top"] = [RiverTile(P("1z"), 0.9, (660, 284, 717, 334))]
    frames(watch, declared)
    safe = watch.declared_safe()
    assert safe[2] == (P("5z"),)                     # 對家自己打的東不算
    assert safe[1] == safe[3] == safe[0] == ()
    watch.new_hand()
    assert watch.declared_safe() == ((), (), (), ())


def test_empty_table_starts_a_new_hand():
    # 流局沒有結算畫面：牌河整個清空就當成新的一局
    from hint.river_watch import EMPTY_RESET_FRAMES

    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["top"] = [RiverTile(P("1z"), 0.9, (660, 284, 717, 334))]
    frames(watch, waiting())
    assert watch.has_discards()
    reader.current["top"] = []
    frames(watch, waiting(), EMPTY_RESET_FRAMES - 1)
    assert watch.has_discards()  # 短暫被蓋住不算
    frames(watch, waiting(), 1)
    assert not watch.has_discards() and watch.reset_by_empty_table == 1


def test_label_that_does_not_look_like_the_river_tile_is_rejected():
    import numpy as np

    class Classifier:
        def scores(self, image):
            return {P("9p"): 0.3, P("2p"): 0.9}

    reader = FakeReader()
    reader.classifier = Classifier()
    watch, saved = watch_with(reader)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    watch.update(frame, Observation(tuple(parse("11223789m99m4p56779s9p")), None, None, ()))
    reader.current["me"] = [RiverTile(P("2p"), 0.9, (580, 651, 650, 717))]
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert watch.tracker.discards["me"] == [P("2p")]  # 標註說九筒，但這張看起來不像：不改
    assert saved == [] and watch.rejected_labels == 1


def test_unrecognized_river_tiles_are_saved_for_labelling(tmp_path):
    import numpy as np

    reader = FakeReader()
    saved = []
    watch = RiverWatch(reader, save=lambda path, image: saved.append(path), unknown_dir=tmp_path)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    reader.current["top"] = [RiverTile(None, 0.3, (660, 284, 717, 334))]
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert len(saved) == 1 and saved[0].name.endswith("_top_660_284.png") and tmp_path in saved[0].parents
    reader.current["top"].append(RiverTile(P("1z"), 0.9, (720, 284, 777, 334)))
    for _ in range(CONFIRM_FRAMES):
        watch.update(frame, waiting())
    assert len(saved) == 1  # 認得的不存


def test_own_discard_claimed_before_the_river_saw_it_is_credited_to_the_claimer():
    # 10/22 00:06：打發，對家馬上碰，牌河從來沒確認到這張
    reader = FakeReader()
    watch = watch_with_banner(reader, [None, None, ("top", "pung")])
    watch.update(None, Observation(tuple(parse("11223789m99m4p56779s6z")), None, None, ()))
    events = []
    for _ in range(3):
        events += watch.update(None, waiting())
    assert [e.kind for e in events] == ["claimed_unseen"] and events[0].tile == P("6z")
    assert watch.melds["top"] == 1 and watch.exposed["top"] == [P("6z"), P("6z")]
    assert P("6z") in watch.discards()[0]
    watch.new_hand()
    assert watch.discards()[0] == ()


def test_meld_hints_list_claimed_tiles_not_yet_read():
    from types import SimpleNamespace

    from game.rules import Meld

    reader = FakeReader()
    watch, _ = watch_with(reader)
    watch._credit("right", "top", (), P("4p"), None)                    # 下家的四筒被對家拿走：吃或碰
    watch._credit("me", "top", (P("1s"), P("1s")), P("1s"), "pung")      # 我打的一條被對家碰
    watch._credit("me", "left", (), P("7z"), None)                      # 字牌一定是碰
    assert watch.meld_hints() == {"top": [(P("4p"), None), (P("1s"), "pung")], "left": [(P("7z"), "pung")]}
    watch.meld_watch = SimpleNamespace(melds={"top": (Meld("chow", tuple(parse("456p")), -1),)})
    assert watch.meld_hints() == {"top": [(P("1s"), "pung")], "left": [(P("7z"), "pung")]}
    watch.meld_watch = None
    watch.new_hand()
    assert watch.meld_hints() == {}


def test_concealed_kong_is_credited_to_the_player_after_the_last_discard():
    from game.rules import Meld
    from hint.meld_watch import MeldWatch

    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["top"] = [RiverTile(P("1p"), 0.9, (700, 200, 760, 280))]
    frames(watch, waiting())
    # 對家打完一張後金牌多一種：輪到上家，他沒有碰、也沒有推算的吃碰 → 上家暗槓
    watch.kong_started()
    assert watch.kong_confirmed() == ("暗槓", "left")
    assert watch.hidden_kongs["left"] == 1 and watch.meld_counts()[3] == 1
    # 那家只有一組碰：是加槓，把碰改成槓（10/29 02:06 下家加槓北，疊上去那張讀成二條）
    melds = MeldWatch(None, template_dir=None, background=False)
    pung = Meld("pung", (P("1s"),) * 3, -1)
    melds._vote("left", 100, pung)
    melds.melds = {"left": (pung,)}
    watch.meld_watch = melds
    watch.kong_started()
    assert watch.kong_confirmed() == ("加槓", "left")
    assert watch.meld_watch.melds["left"] == (pung,)  # 先等讀副露，剛槓的可能是別組
    melds.reads += 2
    assert watch._resolve_added_kong()
    assert watch.meld_watch.melds["left"] == (Meld("kong", (P("1s"),) * 4, -1),)
    # 兩組碰分不出是哪組：不改
    melds = MeldWatch(None, template_dir=None, background=False)
    melds._vote("left", 100, pung)
    melds._vote("left", 200, Meld("pung", (P("9s"),) * 3, -1))
    melds.melds = {"left": tuple(melds._seat_melds("left"))}
    watch.meld_watch = melds
    watch.kong_started()
    assert watch.kong_confirmed() == ("加槓", "left")
    melds.reads += 2
    assert not watch._resolve_added_kong()
    assert all(meld.kind == "pung" for meld in melds.melds["left"])
    # 等的期間讀到新的槓（例如 10/29 01:03 下家已有九萬碰，那次槓的是白）：不改碰
    watch.kong_started()
    assert watch.kong_confirmed() == ("加槓", "left")
    white = Meld("kong", (P("7z"),) * 4, -1)
    melds._vote("left", 300, white)
    melds.melds = {"left": tuple(melds._seat_melds("left"))}
    melds.reads += 2
    assert not watch._resolve_added_kong()
    assert sum(meld.kind == "kong" for meld in melds.melds["left"]) == 1
    # 下一家是自己：是自己槓的
    reader.current["left"] = [RiverTile(P("2p"), 0.9, (400, 300, 460, 380))]
    frames(watch, waiting())
    watch.kong_started()
    assert watch.kong_confirmed() is None
    watch.new_hand()
    assert watch.hidden_kongs["left"] == 0


def test_kong_right_after_a_claim_is_not_a_concealed_kong():
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["right"] = [RiverTile(P("5s"), 0.9, (1300, 300, 1360, 380))]
    frames(watch, waiting())
    reader.current["right"] = []
    frames(watch, waiting())  # 那張被拿走：明槓，吃碰推算已經算過
    watch.kong_started()
    assert watch.kong_confirmed() is None


def test_concealed_kongs_seen_on_screen_count_as_melds():
    from types import SimpleNamespace

    reader = FakeReader()
    watch, _ = watch_with(reader)
    watch.meld_watch = SimpleNamespace(melds={}, concealed={"top": 1}, new_hand=lambda: None)
    assert watch.meld_counts()[2] == 1 and watch.concealed_counts() == {"left": 0, "top": 1, "right": 0}
    # 同一組推算也算到了：取較多的，不重複算
    watch.hidden_kongs["top"] = 1
    assert watch.meld_counts()[2] == 1


def test_claim_seen_after_the_gold_changed_is_an_exposed_kong():
    # 10/29 16:57 對家明槓我打的一筒：金牌先變多，牌河晚一點才報那張被拿走
    reader = FakeReader()
    watch, _ = watch_with(reader)
    reader.current["me"] = [RiverTile(P("1p"), 0.9, (700, 600, 760, 680))]
    frames(watch, waiting())
    watch.kong_started()
    reader.current["me"] = []
    frames(watch, waiting())
    assert watch.kong_confirmed() is None


def test_exposed_kong_turns_the_read_pung_into_a_kong():
    # 疊上去的第四張常讀不到：金牌變多時被拿走的那張，哪一家讀到它的碰就改成槓（之後才讀到的碰也算）
    from game.rules import Meld
    from hint.meld_watch import MeldWatch

    reader = FakeReader()
    watch, _ = watch_with(reader)
    melds = MeldWatch(None, template_dir=None, background=False)
    watch.meld_watch = melds
    reader.current["me"] = [RiverTile(P("1p"), 0.9, (700, 600, 760, 680))]
    frames(watch, waiting())
    watch.kong_started()
    reader.current["me"] = []
    frames(watch, waiting())
    assert watch.kong_confirmed() is None
    assert watch.open_kong_tiles == [P("1p")] and watch.kong_pending()
    pung = Meld("pung", (P("1p"),) * 3, -1)
    melds._vote("top", 1300, pung)
    melds.melds = {"top": (pung,)}
    assert watch._resolve_open_kongs()
    assert melds.melds["top"] == (Meld("kong", (P("1p"),) * 4, -1),)
    assert watch.open_kong_tiles == [] and "對家的一筒碰改成明槓" in watch.kong_notes[-1]


def test_concealed_kong_waits_for_meld_reads():
    from game.rules import Meld
    from hint.meld_watch import MeldWatch

    reader = FakeReader()
    watch, _ = watch_with(reader)
    melds = MeldWatch(None, template_dir=None, background=False)
    watch.meld_watch = melds
    reader.current["top"] = [RiverTile(P("1p"), 0.9, (700, 200, 760, 280))]
    frames(watch, waiting())
    watch.kong_started()
    assert watch.kong_confirmed() == ("暗槓", "left")
    assert watch.hidden_kongs["left"] == 0 and watch.kong_pending()
    melds.reads += 1
    assert not watch._resolve_hidden_kong()
    melds.reads += 1
    assert watch._resolve_hidden_kong()
    assert watch.hidden_kongs["left"] == 1 and not watch.kong_pending()
    assert watch.kong_notes and "確認上家暗槓" in watch.kong_notes[-1]
    # 10/29 15:51 對家的八筒馬上被下家明槓：之後讀到下家的槓，推算的暗槓取消
    watch.kong_started()
    assert watch.kong_confirmed() == ("暗槓", "left")
    kong = Meld("kong", (P("8p"),) * 4, -1)
    melds._vote("right", 100, kong)
    melds.melds = {"right": (kong,)}
    melds.reads += 2
    assert not watch._resolve_hidden_kong()
    assert watch.hidden_kongs["left"] == 1 and "取消推算的上家暗槓" in watch.kong_notes[-1]
