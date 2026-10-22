import numpy as np

from game.rules import Meld
from game.tiles import parse
from hint.meld_watch import MeldWatch, correct_manzu, describe
from perception.meld_reader import MeldTile


def P(text):
    return parse(text)[0]


def test_manzu_pung_is_corrected_by_the_tile_the_river_saw_taken():
    # 10/22 18:43：下家打六萬、對家碰，讀成一萬一萬一萬或一萬六萬六萬；牌河推算對家副露裡有兩張六萬
    known = [P("6m"), P("6m")]
    assert correct_manzu(parse("111m123s"), known) == parse("666m123s")
    assert correct_manzu(parse("166m123s"), known) == parse("666m123s")
    # 讀到的和推算的一致、萬子的吃、或推算沒有萬子碰：不改
    assert correct_manzu(parse("666m"), known) == parse("666m")
    assert correct_manzu(parse("345m"), known) == parse("345m")
    assert correct_manzu(parse("111m"), [P("5p"), P("5p")]) == parse("111m")


class FakeReader:
    def __init__(self, read):
        self.result = read

    def read(self, image):
        return self.result


def hits(text):
    crop = np.full((27, 84, 3), 200, np.uint8)
    return [MeldTile(tile, 0.8, index * 30, crop) for index, tile in enumerate(parse(text))]


def test_watch_applies_reads_and_harvests_confirmed_pungs(tmp_path):
    saved = []
    reader = FakeReader({"left": hits("111m"), "top": hits("123s5p"), "right": []})
    watch = MeldWatch(reader, template_dir=tmp_path, save=lambda path, image: saved.append(path),
                      background=False)
    frame = np.zeros((10, 10, 3), np.uint8)
    assert watch.request(frame)
    assert watch.collect({"left": [P("6m"), P("6m")], "top": [], "right": []})
    # 上家的萬子碰用推算的六萬，三張切圖存成六萬的模板；對家的吃（推算沒有）不存，多出的五筒不算
    assert watch.melds["left"] == (Meld("pung", (P("6m"),) * 3, -1),)
    assert watch.melds["top"] == (Meld("chow", tuple(parse("123s")), -1),)
    assert len(saved) == 3 and all(path.parent == tmp_path / "left" / "6m" for path in saved)
    assert "上家 六萬六萬六萬" in describe(watch.melds)
    assert not watch.collect({})  # 沒有新的讀取
    watch.new_hand()
    assert watch.melds == {}


def test_read_melds_reach_the_game_state_as_known_melds():
    from hint.advisor import Observation, build_state

    observation = Observation(tuple(parse("11223789m99m4p56779s")), None, None, (),
                              melds=(0, 0, 0, 1),
                              opponent_melds=((), (), (), (Meld("pung", (P("5z"),) * 3, -1),)))
    state = build_state(observation, waiting=True)
    melds = state.players[3].melds
    assert len(melds) == 1 and melds[0].kind == "pung" and melds[0].tiles == (P("5z"),) * 3
    assert melds[0].from_player is not None and melds[0].from_player >= 0


def test_seat_labels_list_claims_and_unread_melds():
    from hint.meld_watch import seat_labels

    melds = {"left": (Meld("pung", (P("7z"),) * 3, -1), Meld("chow", tuple(parse("978p")), -1),
                      Meld("kong", (P("2p"),) * 4, -1))}
    labels = seat_labels(melds, {"left": 3, "top": 1, "right": 0})
    assert labels["left"] == "上家　碰 白　吃 七八九筒　槓 二筒"
    assert labels["top"] == "對家　另有 1 組沒讀到"
    assert labels["right"] is None


def test_melds_do_not_disappear_within_a_hand():
    reader = FakeReader({"left": hits("111p"), "top": [], "right": []})
    watch = MeldWatch(reader, template_dir=None, background=False)
    frame = np.zeros((10, 10, 3), np.uint8)
    watch.request(frame)
    watch.collect({})
    assert watch.melds["left"] == (Meld("pung", (P("1p"),) * 3, -1),)
    # 下一次被擋住（或碰變成槓）只讀到一張：沿用之前讀到的碰
    reader.result = {"left": hits("1p"), "top": [], "right": []}
    watch.request(frame)
    assert not watch.collect({})
    assert watch.melds["left"] == (Meld("pung", (P("1p"),) * 3, -1),)
    # 組數一樣或更多就用新的（萬子校正後的結果能更新上去）
    reader.result = {"left": hits("111p234s"), "top": [], "right": []}
    watch.request(frame)
    assert watch.collect({}) and len(watch.melds["left"]) == 2
    watch.new_hand()
    assert watch.melds == {}


def test_chow_hint_guides_the_search_and_harvests_its_tiles(tmp_path):
    saved = []

    class GuidedReader(FakeReader):
        def guided_meld(self, image, seat, tile, kinds):
            assert (seat, tile, kinds) == ("top", P("4p"), ("chow",))
            return [MeldTile(t, score, 1300 + 50 * i, np.zeros((45, 52, 3), np.uint8))
                    for i, (t, score) in enumerate(zip(parse("645p"), (0.6, 0.55, 0.3)))]

    watch = MeldWatch(GuidedReader({"left": [], "top": [], "right": []}), template_dir=tmp_path,
                      save=lambda path, image: saved.append(path), background=False)
    watch.request(np.zeros((10, 10, 3), np.uint8), {"top": [P("4p")]})
    assert watch.collect({})
    assert watch.melds["top"] == (Meld("chow", tuple(parse("456p")), -1),)
    # 分數太低的那張（可能被擋住）不存
    assert len(saved) == 2 and {path.parent.name for path in saved} == {"6p", "4p"}


def test_guided_chow_is_ignored_where_a_meld_was_already_read():
    class GuidedReader(FakeReader):
        def guided_meld(self, image, seat, tile, kinds):
            return [MeldTile(t, 0.5, 30 * i, None) for i, t in enumerate(parse("345s"))]

    watch = MeldWatch(GuidedReader({"left": hits("777z"), "top": [], "right": []}), template_dir=None,
                      background=False)
    watch.request(np.zeros((10, 10, 3), np.uint8), {"left": [P("5s")]})
    watch.collect({})
    assert watch.melds["left"] == (Meld("pung", (P("7z"),) * 3, -1),)


def test_confusable_manzu_are_fixed_when_only_one_change_makes_a_meld():
    from hint.meld_watch import fix_confusable_manzu

    # 20:08 對家七九八萬讀成七萬九萬一萬
    assert fix_confusable_manzu(parse("791m")) == parse("798m")
    assert fix_confusable_manzu(parse("1m666p")) == parse("1m666p")      # 不是三張萬子
    assert fix_confusable_manzu(parse("345m")) == parse("345m")          # 本來就成組
    assert fix_confusable_manzu(parse("166m")) == parse("666m")          # 一萬→六萬才湊得成碰
    assert fix_confusable_manzu(parse("249m")) == parse("249m")          # 沒有可疑的牌


def test_manzu_from_guided_chows_are_not_harvested(tmp_path):
    saved = []

    class GuidedReader(FakeReader):
        def guided_meld(self, image, seat, tile, kinds):
            return [MeldTile(t, 0.5, 1300 + 50 * i, np.zeros((45, 52, 3), np.uint8))
                    for i, t in enumerate(parse("213m"))]

    watch = MeldWatch(GuidedReader({"left": [], "top": [], "right": []}), template_dir=tmp_path,
                      save=lambda path, image: saved.append(path), background=False)
    watch.request(np.zeros((10, 10, 3), np.uint8), {"top": [(P("1m"), None)]})
    watch.collect({})
    assert watch.melds["top"] == (Meld("chow", tuple(parse("123m")), -1),)
    assert saved == []  # 一二三萬的排列可能錯，不存


def test_same_count_with_different_content_keeps_both_up_to_the_inferred_count():
    reader = FakeReader({"left": [], "top": hits("123m"), "right": []})
    watch = MeldWatch(reader, template_dir=None, background=False)
    frame = np.zeros((10, 10, 3), np.uint8)
    watch.request(frame)
    watch.collect({}, {"top": 1})
    # 20:35 下一次只讀到新的一條×3（一二三萬被擋住）：牌河推算對家 2 組 → 兩組都留
    reader.result = {"left": [], "top": hits("111s"), "right": []}
    watch.request(frame)
    watch.collect({}, {"top": 2})
    assert set(watch.melds["top"]) == {Meld("chow", tuple(parse("123m")), -1), Meld("pung", (P("1s"),) * 3, -1)}
    # 只推算 1 組：新讀到的取代舊的（例如萬子校正）
    watch.new_hand()
    reader.result = {"left": [], "top": hits("111m"), "right": []}
    watch.request(frame); watch.collect({}, {"top": 1})
    reader.result = {"left": [], "top": hits("666m"), "right": []}
    watch.request(frame); watch.collect({}, {"top": 1})
    assert watch.melds["top"] == (Meld("pung", (P("6m"),) * 3, -1),)
