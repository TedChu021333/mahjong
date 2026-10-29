"""即時模式的牌河追蹤：記錄各家打過的牌，順便自動收集牌河模板。

牌河模板一開始很少，辨識不出的牌記為未知。每次出牌都有可靠的標註來源：
- 自己打的牌：比對打牌前後的暗手牌就知道。
- 別家打的牌：能吃碰時會放大顯示（辨識很準），就是那家牌河接著出現的那張。
辨識器認不出或認錯時，把這張牌存成新模板（train/river_templates/<牌>/auto_*.png），
並以標註修正記錄。

副露數：牌打出後會先落進牌河，有人吃碰才從牌河被拿走（追蹤器的 "claimed" 事件）。
接著第一個出牌的人（牌河多一張，或出現新的放大圓框）就是吃碰的人（吃碰完要馬上打一張），
副露數加 1；被拿走的牌之後又回到原位（其實是被蓋住）就取消。下一個出牌的是自己、而自己
並沒有吃碰時，吃碰的一定是上家（他吃碰完打一張就輪到我們，只是那張被遮住還沒看到）。
吃只能吃上家的牌：拿走牌的人不是打牌者的下家，就一定是碰（或槓），那種牌另外兩張也看得到了。
別家吃碰時吃碰者那邊會出現「吃」／「碰」大字（read_claim_banner）：剛出現的大字之後幾秒內有牌被拿走，
直接照大字算給那家（碰的話另外兩張也看得到），不必等下一個出牌的人；大字比牌河記的「被拿牌的那家」可靠。
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from game.tiles import tile_code, tile_name
from hint.advisor import RIVER_SEAT, Observation
from hint.meld_watch import SEAT_NAMES
from perception.capture import write_image
from perception.config import RIVER_TEMPLATE_DIR, TRAIN_DIR
from perception.river_reader import SEATS, RiverReader, crop_face
from perception.river_tracker import RiverEvent, RiverTracker
from perception.table_reader import read_claim_banner

NEXT_SEAT = {"me": "right", "right": "top", "top": "left", "left": "me"}
"""出牌順序：自己 → 下家 → 對家 → 上家。"""
BANNER_FRAMES = 10
"""大字出現後這麼多次更新（約 3 秒）內的「被拿走」才算這次吃碰。"""
LABEL_FRAMES = 15
"""標註在這麼多幀內等不到那家牌河多一張就作廢。"""
UNKNOWN_DIR = TRAIN_DIR / "river_unknown"
UNKNOWN_LIMIT = 200
"""認不出的牌河牌切圖存到 train/river_unknown/<日期>/（每次執行最多這麼多張），之後標註補模板用（10/20 起）。"""
LABEL_MIN_SCORE = 0.7
"""標註（手牌變化、放大的牌）和這張牌河的牌比對，至少要這麼像才採用、存成模板。10/19 發現牌河模板 6753 張
有 1447 張放錯資料夾或不是牌（吃碰動畫、暗條；標註套到錯的那張），自己打的牌只讀對約一半；清掉後今天讀錯的
126 張全部讀對。門檻 0.7：正確的標註留下 99%、錯的擋下 91%。"""
ADDED_KONG_READS = 2
"""推算加槓後等讀幾次副露（約 8 秒一次）才把碰改成槓。"""
EMPTY_RESET_FRAMES = 10
"""四家牌河連續這麼多次都讀不到任何牌、追蹤器卻記著牌時，當成新的一局（沒有換三張的場、流局沒有結算畫面）。"""


def _read_banner(frame):
    return read_claim_banner(frame) if frame is not None else None


class RiverWatch:
    def __init__(self, reader: RiverReader, template_dir: Path = RIVER_TEMPLATE_DIR,
                 save=write_image, read_banner=None, unknown_dir: Path | None = UNKNOWN_DIR,
                 meld_watch=None) -> None:
        self.reader = reader
        self.meld_watch = meld_watch
        """讀對手副露（hint.meld_watch.MeldWatch，可為 None）。"""
        self.unknown_dir = unknown_dir
        self.unknown_saved = 0
        self.read_banner = read_banner or _read_banner
        self._banner: tuple[str, str, int] | None = None
        """剛出現、還沒對到被拿走的牌的大字：(吃碰者, chow/pung, 出現時的更新次數)。"""
        self._banner_visible = False
        self.tracker = RiverTracker()
        self.template_dir = template_dir
        self.save = save
        self.saved = 0
        self._frame = 0
        self._labels: dict[str, tuple[int, int]] = {}
        self._unlabeled: dict[str, tuple[np.ndarray, RiverEvent, int]] = {}
        """剛出現、還沒等到標註的出牌（畫面、事件、幀數）：自己的手牌變化常比牌河晚幾幀才看到。"""
        """座位 → (牌, 取得標註時的幀數)。"""
        self._turn_hand: tuple[int, ...] | None = None
        self.melds = {seat: 0 for seat in SEATS}
        """推算的各家副露數（自己的由手牌張數得知，這裡不算）。"""
        self.exposed: dict[str, list[int]] = {seat: [] for seat in SEATS}
        self.claims: dict[str, list[tuple[int | None, str | None]]] = {seat: [] for seat in SEATS}
        """推算出的各家吃碰：(被拿走的牌, "chow"／"pung"／None 分不出)，給讀副露時引導對家的吃用。"""
        """各家副露裡已知、不在牌河的牌：推算出的碰（另外兩張）、自己吃碰用掉的手牌。"""
        self._taken_from: tuple[str, int, int | None] | None = None
        """最新一張被拿走的牌是誰打的、當時自己的副露數、那張牌，等著看誰接著出牌。"""
        self._my_melds = 0
        self._credited: tuple[str, str, tuple[int, ...]] | None = None
        """最近一次推算：(被拿走牌的那家, 算給誰, 加進 exposed 的牌)；那張牌之後又回來就撤銷。"""
        self._last_circle: tuple[str, int] | None = None
        self.my_discards = 0
        """這局自己打出幾張（由手牌張數變化得知，不靠牌河；漏看輪到自己的畫面時會少算）。"""
        self.my_pungs: list[int] = []
        """這局自己碰過的牌（加槓只可能是這些）。"""
        self.rejected_labels = 0
        """和牌河的牌不像、沒採用的標註次數。"""
        self._empty_frames = 0
        self.reset_by_empty_table = 0
        """因為牌河整個清空而自動當成新局的次數（記錄用）。"""
        self._own_pending: tuple[int | None, int] | None = None
        """自己剛打出、牌河還沒看到的牌：(牌, 打出時的更新次數)。"""
        self._own_seen_at = -LABEL_FRAMES - 1
        """牌河最近一次出現自己出牌（或被拿走）時的更新次數。"""
        self.unseen_claimed: list[int] = []
        """自己打出、牌河還沒確認就被吃碰拿走的牌（由手牌變化＋吃碰大字得知），算進自己打過的牌。"""
        self.hidden_kongs = {seat: 0 for seat in SEATS}
        """推算的各家暗槓數（kong_confirmed）：牌背朝上、讀不到牌，只算一組副露。"""
        self._last_mover: tuple[str, str] | None = None
        """牌河最近一個事件：(座位, "discard"／"claimed")，推算槓的是誰用。"""
        self._kong_from: tuple[str, str] | None = None
        self._added_pending: tuple[str, int, int, int] | None = None
        """推算某家加槓、等讀副露確認：(座位, 當時讀到他的槓數, 畫面上看到他的暗槓數, 當時讀副露的次數)。"""
        self._hidden_pending: tuple[str, int, int] | None = None
        """推算某家暗槓、等讀副露確認：(座位, 當時讀到的各家槓合計, 當時讀副露的次數)。"""
        self._claimed_since_kong = False
        """金牌變多之後牌河報過「被拿走」：是明槓。"""
        self._claimed_tile_since_kong: int | None = None
        """那張被拿走的牌（讀得出來時）。"""
        self.open_kong_tiles: list[int] = []
        """推算被明槓的牌、還沒在副露裡標成槓的：之後哪一家讀到這種牌的碰，就改成槓（一種牌最多 4 張，碰只會在一家）。
        疊上去的第四張常讀不到，明槓原本一直算成碰（10/28 使用者：對手的槓都被當成碰）。"""
        self.kong_notes: list[str] = []
        """暗槓推算確認或取消的說明，提示迴圈寫進記錄後清掉。"""
        self._declare_marks: dict[str, dict[str, int]] = {}
        """第一次認到某家宣告時，各家牌河（含被吃碰的）已有幾張；之後別家打出的牌他一定不聽
        （遊戲代打、自動胡），見 declared_safe。"""

    def new_hand(self) -> None:
        self.tracker.reset()
        self._labels.clear()
        self._unlabeled.clear()
        self._turn_hand = None
        self.melds = {seat: 0 for seat in SEATS}
        self.exposed = {seat: [] for seat in SEATS}
        self.claims = {seat: [] for seat in SEATS}
        self._taken_from = None
        self._last_circle = None
        self._credited = None
        self._banner = None
        self.my_discards = 0
        self._declare_marks = {}
        self._empty_frames = 0
        self.my_pungs = []
        self._own_pending = None
        self.unseen_claimed = []
        self.hidden_kongs = {seat: 0 for seat in SEATS}
        self._last_mover = self._kong_from = self._added_pending = self._hidden_pending = None
        self._claimed_tile_since_kong = None
        self.open_kong_tiles = []
        if getattr(self, "meld_watch", None) is not None:
            self.meld_watch.new_hand()

    def update(self, frame: np.ndarray, observation: Observation) -> list[RiverEvent]:
        self._frame += 1
        self._my_melds = observation.meld_count
        self._collect_labels(observation)
        banner = self.read_banner(frame)
        if banner is not None and not self._banner_visible:
            self._banner = (*banner, self._frame)  # 只記剛出現的那一次
        self._banner_visible = banner is not None
        reading = self.reader.read(frame)
        if any(reading.values()) or not self.has_discards():
            self._empty_frames = 0
        else:
            self._empty_frames += 1
            if self._empty_frames >= EMPTY_RESET_FRAMES:
                self.new_hand()
                self.reset_by_empty_table += 1
        events = self.tracker.update(reading)
        for seat in observation.declared:
            if seat not in self._declare_marks:
                # 同一幀剛出現的牌不算（可能是宣告前打的）
                self._declare_marks[seat] = {other: len(tiles)
                                             for other, tiles in self.tracker.history.items()}
        for event in events:
            if event.seat == "me" and event.kind in ("discard", "claimed"):
                self._own_pending = None
                self._own_seen_at = self._frame
            if event.kind in ("discard", "claimed"):
                self._last_mover = (event.seat, event.kind)
            if event.kind == "claimed":
                self._claimed_since_kong = True
                self._claimed_tile_since_kong = event.tile
            if event.kind == "discard":
                self._apply_label(frame, event)
                self._save_unknown(frame, event)
                self._someone_discarded(event.seat)
            elif event.kind == "claimed":
                if self._banner and self._frame - self._banner[2] <= BANNER_FRAMES \
                        and self._banner[0] != event.seat:
                    claimer, kind, _ = self._banner
                    self._banner = self._taken_from = None
                    tile = event.tile
                    self._credit(event.seat, claimer,
                                 (tile, tile) if kind == "pung" and tile is not None else (), tile, kind)
                    continue
                self._taken_from = (event.seat, self._my_melds, event.tile)
            elif event.kind == "restored":
                if self._taken_from and self._taken_from[0] == event.seat:
                    self._taken_from = None
                if self._credited and self._credited[0] == event.seat:
                    _, claimer, tiles = self._credited
                    self.melds[claimer] -= 1  # 其實只是被蓋住
                    for tile in tiles:
                        self.exposed[claimer].remove(tile)
                    if self.claims[claimer]:
                        self.claims[claimer].pop()
                    self._credited = None
        unseen = self._claimed_before_seen()
        if unseen is not None:
            events.append(unseen)
        return events

    def _claimed_before_seen(self) -> RiverEvent | None:
        """自己剛打出（手牌少一張）、牌河還沒確認，就有別家出現吃碰大字：那張牌被他吃碰走了。
        10/22 「自己的牌沒進牌河」截圖看了 3 張都是這樣（對家碰發、上家碰白、下家吃筒）——牌停不到 3 幀就被拿走，
        牌河追蹤沒有事件，那家的副露數、碰走的另外兩張、自己打過的牌都漏記。"""
        pending, banner = self._own_pending, self._banner
        # 大字可能比手牌變化早一兩次更新被看到；早更多的是前一次別人的吃碰
        if pending is None or banner is None or banner[0] == "me" \
                or not -2 <= banner[2] - pending[1] <= BANNER_FRAMES:
            return None
        tile = pending[0]
        claimer, kind, _ = banner
        self._own_pending = self._banner = None
        self._own_seen_at = self._frame
        self._last_mover = ("me", "claimed")
        self._claimed_since_kong = True
        self._claimed_tile_since_kong = tile
        self._credit("me", claimer, (tile, tile) if kind == "pung" and tile is not None else (), tile, kind)
        if tile is not None:
            self.unseen_claimed.append(tile)
        return RiverEvent("claimed_unseen", "me", tile, (0, 0, 0, 0))

    def _someone_discarded(self, seat: str) -> None:
        taken, self._taken_from = self._taken_from, None
        if taken is None or seat == taken[0]:
            return
        claimer = seat
        if seat == "me":
            if self._my_melds != taken[1] or taken[0] == "left":
                return  # 自己吃碰的
            claimer = "left"  # 自己沒吃碰卻輪到自己：是上家吃碰
        tile = taken[2]
        chow_possible = claimer == NEXT_SEAT[taken[0]]
        tiles = (tile, tile) if tile is not None and not chow_possible else ()
        self._credit(taken[0], claimer, tiles, tile, None if chow_possible else "pung")

    def _credit(self, source: str, claimer: str, tiles: tuple[int, ...], tile: int | None = None,
                kind: str | None = None) -> None:
        self.claims[claimer].append((tile, kind))
        self.melds[claimer] += 1
        self.exposed[claimer] += tiles
        self._credited = (source, claimer, tiles)

    def own_claim(self, tiles) -> None:
        """自己吃碰完成：用掉的手牌（被吃碰的那張已在別家牌河）。"""
        self.exposed["me"] += list(tiles)
        tiles = list(tiles)
        if len(tiles) == 2 and tiles[0] == tiles[1]:
            self.my_pungs.append(tiles[0])

    def exposed_tiles(self) -> tuple[tuple[int, ...], ...]:
        """依 advisor 的座位編號排列。"""
        result = [()] * 4
        for seat, tiles in self.exposed.items():
            result[RIVER_SEAT[seat]] = tuple(tiles)
        return tuple(result)

    def meld_counts(self) -> tuple[int, int, int, int]:
        """依 advisor 的座位編號排列；自己的欄位為 0。讀到的副露組數比推算的多就用讀到的，再加推算的暗槓。"""
        result = [0] * 4
        read = self.meld_watch.melds if self.meld_watch is not None else {}
        for seat, count in self.melds.items():
            if seat != "me":
                result[RIVER_SEAT[seat]] = min(max(count, len(read.get(seat, ()))) + self.concealed(seat), 5)
        return tuple(result)

    def concealed(self, seat: str) -> int:
        """這家的暗槓組數：畫面上看到的（MeldWatch.concealed）和金牌變多推算的（hidden_kongs）取較多的。"""
        seen = self.meld_watch.concealed.get(seat, 0) if self.meld_watch is not None else 0
        return max(seen, self.hidden_kongs[seat])

    def concealed_counts(self) -> dict[str, int]:
        return {seat: self.concealed(seat) for seat in SEATS if seat != "me"}

    def kong_started(self) -> None:
        """金牌剛變多（有人槓）：記下當時牌河最近的事件，槓的是那之後輪到的人。"""
        self._kong_from = self._last_mover
        self._claimed_since_kong = False
        self._claimed_tile_since_kong = None

    def kong_confirmed(self) -> tuple[str, str] | None:
        """金牌變多確認了、又不是自己槓的：推算是誰槓、哪一種，回傳 ("暗槓"／"加槓", 座位)，推算不出來回傳 None。
        暗槓：副露數加 1。加槓：那家只有一組碰時把它改成槓（MeldWatch.mark_added_kong）。
        暗槓牌背朝上、副露辨識讀不到，原本整組漏掉（10/29 對手的槓 33 次只讀到 9 次，抽看的多是暗槓）。
        槓的人＝最近一張打出的牌的下一家（自己 → 下家 → 對家 → 上家）。保守起見這些不算：
        最近的事件是有牌被拿走（明槓，已由吃碰推算）、輪到自己、那家已有碰（可能是加槓，不多一組；
        10/29 02:33 上家加槓一條）、那家推算的吃碰裡有還沒讀到、可能是碰的。
        重放 10/29 一晚 40 次槓：算成暗槓 8 次，看了其中 5 張截圖都是暗槓（三張牌背、中間疊一張）。
        金牌變多到確認之間牌河才報「被拿走」的也是明槓：金牌一槓就變，牌河要那張不見 3 幀才報（10/29 16:57
        對家明槓我打的一筒，金牌變多時最近的事件還是我打牌，被當成下家暗槓）。
        暗槓先不算（_hidden_pending），等之後讀 ADDED_KONG_READS 次副露、哪一家都沒有多讀到槓才算：
        10/29 15:51 對家打的八筒馬上被下家明槓、沒進牌河，最近的事件是更早的出牌，被當成上家暗槓。"""
        mover, self._kong_from = self._kong_from, None
        if self._claimed_since_kong:
            tile = self._claimed_tile_since_kong
            if tile is not None and self.meld_watch is not None:
                self.open_kong_tiles.append(tile)
                self.kong_notes.append(f"金牌變多時{tile_name(tile)}被拿走：是明槓，讀到{tile_name(tile)}的碰就算槓")
                self._resolve_open_kongs()
            return None
        if mover is None or mover[1] != "discard":
            return None
        seat = NEXT_SEAT[mover[0]]
        if seat == "me":
            return None
        read = self.meld_watch.melds.get(seat, ()) if self.meld_watch is not None else ()
        if any(meld.kind == "pung" for meld in read):
            # 先等讀副露：剛槓的可能是別組（暗槓、或疊上去的那張讀得到），讀到新的槓就不改
            kongs = sum(meld.kind == "kong" for meld in read)
            self._added_pending = (seat, kongs, self.meld_watch.concealed.get(seat, 0), self.meld_watch.reads)
            return ("加槓", seat)
        if any(meld.kind != "chow" for meld in read):
            return None
        claims = self.claims[seat]
        if len(claims) > len(read) and any(kind != "chow" for _, kind in claims):
            return None
        if self.meld_watch is None:
            self.hidden_kongs[seat] += 1
        else:
            self._hidden_pending = (seat, self._kong_total(), self.meld_watch.reads)
        return ("暗槓", seat)

    def _kong_total(self) -> int:
        """讀到的各家槓（含畫面上看到的暗槓）合計。"""
        return (sum(meld.kind == "kong" for melds in self.meld_watch.melds.values() for meld in melds)
                + sum(self.meld_watch.concealed.values()))

    def kong_pending(self) -> bool:
        """有推算的槓在等讀副露確認（提示迴圈照樣排讀副露，場上沒有副露時也讀）。"""
        return self._hidden_pending is not None or self._added_pending is not None or bool(self.open_kong_tiles)

    def meld_hints(self) -> dict[str, list[tuple[int, str | None]]]:
        """推算有吃碰、讀到的副露裡還沒有的：各家 (被拿走的牌, "chow"／"pung"／None 分不出)，引導讀副露用。"""
        read = self.meld_watch.melds if self.meld_watch is not None else {}
        hints: dict[str, list[tuple[int, str | None]]] = {}
        for seat, claims in self.claims.items():
            if seat == "me":
                continue
            covered = [tile for meld in read.get(seat, ()) for tile in meld.tiles]
            for tile, kind in claims:
                if tile is None or (kind != "pung" and tile >= 27 and kind is not None):
                    continue
                if tile in covered:
                    covered.remove(tile)
                    continue
                hints.setdefault(seat, []).append((tile, "pung" if tile >= 27 else kind))
        return hints

    def request_melds(self, frame: np.ndarray) -> bool:
        """在背景讀對手副露（沒有 meld_watch 或上一次還沒讀完就不讀）；順便帶上推算吃碰了的牌引導搜尋。"""
        return self.meld_watch is not None and self.meld_watch.request(frame, self.meld_hints())

    def collect_melds(self) -> bool:
        """背景讀完的副露套用進來（用牌河推算的碰校正萬子）；有變化回傳 True。"""
        if self.meld_watch is None:
            return False
        known = {seat: list(tiles) for seat, tiles in self.exposed.items() if seat != "me"}
        changed = self.meld_watch.collect(known, {seat: count for seat, count in self.melds.items() if seat != "me"})
        hidden = self._resolve_hidden_kong()
        opened = self._resolve_open_kongs()
        return self._resolve_added_kong() or hidden or opened or changed

    def _resolve_open_kongs(self) -> bool:
        """推算被明槓的牌：哪一家讀到這種牌的碰就改成槓（MeldWatch.mark_kong）；還沒讀到的留著等下次讀副露。"""
        changed = False
        for tile in self.open_kong_tiles[:]:
            seat = self.meld_watch.mark_kong(tile)
            if seat is not None:
                self.open_kong_tiles.remove(tile)
                self.kong_notes.append(f"{SEAT_NAMES[seat]}的{tile_name(tile)}碰改成明槓")
                changed = True
        return changed

    def _resolve_hidden_kong(self) -> bool:
        """推算的暗槓：之後讀了 ADDED_KONG_READS 次副露，哪一家都沒有多讀到槓，才算進那家的暗槓；結果寫進 kong_notes。"""
        if self._hidden_pending is None:
            return False
        seat, total, reads = self._hidden_pending
        name = SEAT_NAMES[seat]
        if self._kong_total() > total:
            self._hidden_pending = None
            self.kong_notes.append(f"取消推算的{name}暗槓（之後讀到別的槓，金牌變多是那一組）")
            return False
        if self.meld_watch.reads - reads < ADDED_KONG_READS:
            return False
        self._hidden_pending = None
        self.hidden_kongs[seat] += 1
        self.kong_notes.append(f"確認{name}暗槓（讀了 {ADDED_KONG_READS} 次副露都沒有別的槓），副露數加 1")
        return True

    def _resolve_added_kong(self) -> bool:
        """推算的加槓：之後讀了 ADDED_KONG_READS 次副露，那家都沒有多讀到槓，就把他唯一的碰改成槓。"""
        if self._added_pending is None:
            return False
        seat, kongs, concealed, reads = self._added_pending
        read = self.meld_watch.melds.get(seat, ())
        if sum(meld.kind == "kong" for meld in read) > kongs or self.meld_watch.concealed.get(seat, 0) > concealed:
            self._added_pending = None  # 讀到了新的槓（或畫面上看到暗槓）：不用改
            return False
        if self.meld_watch.reads - reads < ADDED_KONG_READS:
            return False
        self._added_pending = None
        return self.meld_watch.mark_added_kong(seat) is not None

    def read_melds(self) -> tuple[tuple, ...]:
        """讀到的對手副露（Meld），依 advisor 的座位編號排列；自己的欄位是空的。"""
        result = [()] * 4
        if self.meld_watch is not None:
            for seat, melds in self.meld_watch.melds.items():
                result[RIVER_SEAT[seat]] = tuple(melds)
        return tuple(result)

    def has_discards(self) -> bool:
        return any(self.tracker.history.values())

    def declared_safe(self) -> tuple[tuple[int, ...], ...]:
        """各宣告者在認到他宣告之後，別家打出的牌（依 advisor 的座位編號排列）。"""
        result = [()] * 4
        history = self.tracker.discards
        for declarer, marks in self._declare_marks.items():
            result[RIVER_SEAT[declarer]] = tuple(
                tile for seat, tiles in history.items() if seat != declarer
                for tile in tiles[marks.get(seat, 0):] if tile is not None)
        return tuple(result)

    def discards(self) -> tuple[tuple[int, ...], ...]:
        """依 advisor 的座位編號排列，未知的牌略過。"""
        result = [()] * 4
        for seat, tiles in self.tracker.discards.items():
            known = [tile for tile in tiles if tile is not None]
            if seat == "me":
                known += self.unseen_claimed
            result[RIVER_SEAT[seat]] = tuple(known)
        return tuple(result)

    def _collect_labels(self, observation: Observation) -> None:
        claim = observation.claim
        if claim is not None and claim.tile is not None:
            self._labels[claim.source] = (claim.tile, self._frame)
            key = (claim.source, claim.tile)
            if key != self._last_circle:  # 新的放大圓框：這家剛打出一張
                self._last_circle = key
                self._last_mover = (claim.source, "discard")
                self._someone_discarded(claim.source)
        if observation.my_turn:
            self._turn_hand = observation.hand
        elif self._turn_hand is not None and len(observation.hand) == len(self._turn_hand) - 1:
            self.my_discards += 1
            gone = Counter(self._turn_hand) - Counter(observation.hand)
            if self._frame - self._own_seen_at > LABEL_FRAMES:  # 牌河已經先看到就不用等
                self._own_pending = (next(iter(gone)) if sum(gone.values()) == 1 else None, self._frame)
            if sum(gone.values()) == 1:
                self._labels["me"] = (next(iter(gone)), self._frame)
                pending = self._unlabeled.pop("me", None)
                if pending is not None and self._frame - pending[2] <= LABEL_FRAMES:
                    # 牌河先看到這張出牌、手牌變化晚到（10/18 03:24 打的八萬、九萬被記成二筒、三萬）
                    self._apply_label(pending[0], pending[1], retry=True)
            self._turn_hand = None

    def _save_unknown(self, frame: np.ndarray | None, event: RiverEvent) -> None:
        """套用標註後還是認不出的牌：切圖存起來，之後人工標註補模板。"""
        entries = self.tracker.history[event.seat]
        if (self.unknown_dir is None or frame is None or self.unknown_saved >= UNKNOWN_LIMIT
                or not entries or entries[-1].tile is not None):
            return
        self.unknown_saved += 1
        # 檔名附框的左上角座標：10/20 對家區域反覆出現一種灰白色、不是牌的東西，要知道在哪
        path = self.unknown_dir / time.strftime("%Y%m%d") /             f"{time.strftime('%H%M%S')}_{self._frame}_{event.seat}_{event.box[0]}_{event.box[1]}.png"
        self.save(path, crop_face(frame, event.box))

    def _apply_label(self, frame: np.ndarray, event: RiverEvent, retry: bool = False) -> None:
        """retry：標註晚到，回頭補標剛才的出牌。"""
        label = self._labels.pop(event.seat, None)
        if label is None or self._frame - label[1] > LABEL_FRAMES:
            if not retry and event.seat == "me":
                self._unlabeled["me"] = (frame, event, self._frame)  # 等手牌變化
            return
        tile = label[0]
        entry = self.tracker.history[event.seat][-1]
        if entry.tile == tile:
            return  # 已經認得，不必再存
        classifier = getattr(self.reader, "classifier", None)
        if classifier is not None and frame is not None                 and classifier.scores(crop_face(frame, event.box)).get(tile, -1.0) < LABEL_MIN_SCORE:
            self.rejected_labels += 1  # 標註和這張牌不像：多半套錯了牌（或偵測到的不是牌），不改也不存
            return
        path = self.template_dir / tile_code(tile) / \
            f"auto_{time.strftime('%Y%m%d_%H%M%S')}_{self._frame}_{event.seat}.png"
        self.save(path, crop_face(frame, event.box))
        self.saved += 1
        entry.tile, entry.score = tile, 1.0


def summary(watch: RiverWatch) -> str:
    from game.tiles import tile_name

    parts = []
    for seat in SEATS:
        tiles = watch.tracker.discards[seat]
        text = f"{seat} " + "".join(tile_name(t) if t is not None else "?" for t in tiles)
        if watch.melds[seat]:
            text += f"（副露 {watch.melds[seat]}）"
        parts.append(text)
    return "；".join(parts)
