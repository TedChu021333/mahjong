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
