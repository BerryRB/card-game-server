# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - 游戏引擎（房间级）"""

from __future__ import annotations
import random
import asyncio
from collections import Counter
from typing import Optional
from dataclasses import dataclass, field

from server.card import Card, Deck, card_from_str
from server.rules import (
    get_dui_and_liandui, card_type_analyze, compare_outcards,
    determine_play_type, compare_zhu_cards, get_zhu_rank
)
from server.ai import AI
from server.constants import (
    GamePhase, SCORE_CARDS, FIXED_ZHU_NAMES, LIANG_TYPE_RANK, LEVEL_ORDER,
    COLOR_NAMES, COLOR_SYMBOLS
)


@dataclass
class Player:
    """房间内的玩家"""
    player_id: int       # 座位号 0-3
    sid: str = ''        # WebSocket session id
    name: str = ''       # 昵称
    cards_in_hand: dict[str, list[Card]] = field(default_factory=dict)
    player_level: int = 1
    is_banker: bool = False
    is_robot: bool = False
    ready: bool = False
    client_id: str = ''

    @property
    def card_count(self) -> int:
        return sum(len(cards) for cards in self.cards_in_hand.values())

    def has_color(self, color: str) -> bool:
        return color in self.cards_in_hand and len(self.cards_in_hand[color]) > 0

    def count_zhu(self, now_level: str, now_color: str) -> int:
        """计算手中主牌数量"""
        count = 0
        for color, cards in self.cards_in_hand.items():
            for card in cards:
                if card.is_zhu(now_level, now_color):
                    count += 1
        return count

    def remove_cards(self, cards_to_remove: list[Card]) -> None:
        remaining = list(cards_to_remove)
        for color in list(self.cards_in_hand.keys()):
            new_hand = []
            for card in self.cards_in_hand[color]:
                if remaining and card in remaining:
                    remaining.remove(card)
                else:
                    new_hand.append(card)
            self.cards_in_hand[color] = new_hand
            if not new_hand:
                del self.cards_in_hand[color]

    def add_cards(self, cards_to_add: list[Card]) -> None:
        for card in cards_to_add:
            color = card.color
            if color in self.cards_in_hand:
                self.cards_in_hand[color].append(card)
            else:
                self.cards_in_hand[color] = [card]
        for color in self.cards_in_hand:
            self.cards_in_hand[color].sort(key=lambda c: c.rank)

    def hand_to_str_list(self) -> list[str]:
        """手牌转为字符串列表（用于传输）"""
        result = []
        for color in sorted(self.cards_in_hand.keys()):
            for card in self.cards_in_hand[color]:
                result.append(card.card_type)
        return result


class GameRoom:
    """游戏房间"""

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.players: list[Optional[Player]] = [None, None, None, None]
        self.phase: GamePhase = GamePhase.WAITING
        self.deck: Deck = Deck()

        # 游戏状态
        self.now_level: str = 'A'       # 当前升级牌
        self.now_color: Optional[str] = None  # 亮主花色
        self.bankers: list[int] = []     # 庄家座位号
        self.hole_cards: list[Card] = []  # 底牌
        self.koupai_cards: list[Card] = []  # 扣回的牌
        self.huanpai_offer: list[Card] = []  # 换牌：扣回牌中亮主花色的牌
        self.huanpai_accepted: bool = False   # 换牌：亮主玩家是否接受
        self.huanpai_picked_cards: list[Card] = []  # 换牌：亮主玩家拾起的牌（仅亮主者可见）
        self.huanpai_return_cards: list[Card] = []  # 换牌：还回的牌（用于展示）
        self.bottom_picker: Optional[int] = None  # 最后摸底牌的人，决定首圈先出

        # 亮主
        self.liangzhu_player: Optional[int] = None
        self.liangzhu_cards: list[Card] = []
        self.liangzhu_type: str = ''
        self.liangzhu_candidates: list = []  # 多人亮主候选列表

        # 吃牌（先到先得）
        self.chipai_claimed_by: Optional[int] = None  # 谁先点了吃牌
        self.chipai_passed: set = set()  # 已选择不吃的人
        self.chipai_result: Optional[dict] = None  # 吃牌结果（返牌等）
        self.chipai_return_cards: list[Card] = []  # 返给被吃者的牌（仅被吃者可见）

        # 出牌
        self.current_turn: int = 0       # 当前该谁出牌
        self.epoch_cards: list[list[Card]] = []  # 本轮出牌
        self.epoch_players: list[int] = []       # 本轮出牌者
        self.epoch_count: int = 0

        # 上一轮出牌记录（用于前端展示）
        self.last_epoch_cards: list[list[Card]] = []
        self.last_epoch_players: list[int] = []
        self.last_epoch_winner: int = -1

        # 计分
        self.score_now: int = 0          # 闲家得分
        self.score_koupai: int = 0       # 底牌分数

        # 回合数
        self.round_num: int = 0
        self.game_num: int = 0  # 局数计数器（第一局=1）

        # 游戏结果
        self.game_result: Optional[dict] = None

        # 消息队列
        self.messages: list[dict] = []

        # 亮主超时计时器
        self._liangzhu_timer = None

    def add_player(self, seat: int, sid: str, name: str) -> bool:
        """玩家入座"""
        if seat < 0 or seat > 3:
            return False
        if self.players[seat] is not None:
            return False
        # 检查是否已在其他座位
        for p in self.players:
            if p and p.sid == sid:
                return False
        self.players[seat] = Player(
            player_id=seat, sid=sid, name=name,
            player_level=1
        )
        return True

    def remove_player(self, sid: str) -> Optional[int]:
        """玩家离开，返回座位号"""
        for i, p in enumerate(self.players):
            if p and p.sid == sid:
                self.players[i] = None
                return i
        return None

    def get_player_by_sid(self, sid: str) -> Optional[Player]:
        for p in self.players:
            if p and p.sid == sid:
                return p
        return None

    def get_seat_by_sid(self, sid: str) -> int:
        for i, p in enumerate(self.players):
            if p and p.sid == sid:
                return i
        return -1

    @property
    def player_count(self) -> int:
        return sum(1 for p in self.players if p is not None)

    @property
    def is_full(self) -> bool:
        return self.player_count == 4

    def get_state(self, for_sid: str) -> dict:
        """获取房间状态（为特定玩家，隐藏其他玩家手牌）"""
        seat = self.get_seat_by_sid(for_sid)
        my_player = self.players[seat] if seat >= 0 else None

        players_info = []
        for i, p in enumerate(self.players):
            if p:
                info = {
                    'seat': i,
                    'name': p.name,
                    'is_banker': p.is_banker,
                    'card_count': p.card_count,
                    'ready': p.ready,
                    'is_robot': p.is_robot,
                }
                if i == seat and my_player:
                    info['cards'] = my_player.hand_to_str_list()
                    info['level'] = my_player.player_level
                players_info.append(info)

        # 扣牌者信息
        koupai_picker = -1
        if self.phase == GamePhase.KOUPAI:
            koupai_picker = self._get_koupai_picker()

        # 游戏结果信息
        game_result = None
        if self.phase == GamePhase.GAME_OVER:
            game_result = self.game_result

        # 信息隐藏：扣底拾牌完成前，now_color只对亮主者可见
        visible_now_color = self.now_color
        if self.phase in (GamePhase.LIANGZHU, GamePhase.CHIPAI, GamePhase.KOUPAI, GamePhase.HUANPAI):
            if seat >= 0 and seat == self.liangzhu_player:
                visible_now_color = self.now_color
            else:
                visible_now_color = None

        # CHIPAI阶段：异队玩家是否可以吃牌/不吃
        can_chipai_claim = False
        can_chipai_pass = False
        if self.phase == GamePhase.CHIPAI and self.liangzhu_player is not None:
            liang_team = self.liangzhu_player % 2
            if seat >= 0 and seat % 2 != liang_team and self.chipai_claimed_by is None and seat not in self.chipai_passed:
                # 异队且还没人吃、自己也没pass
                if my_player and not my_player.is_robot:
                    # 检查是否有亮主牌型
                    liangzhu_types = self._find_chipai_liangzhu_candidates(my_player)
                    can_chipai_claim = len(liangzhu_types) > 0
                    can_chipai_pass = True

        # 亮主/吃牌：当前玩家是否有可用的亮主牌型
        can_liangzhu = False
        liangzhu_available_types = []
        if self.phase in (GamePhase.LIANGZHU, GamePhase.CHIPAI) and seat >= 0 and my_player:
            # LIANGZHU: 未ready且同队没人亮
            # CHIPAI: 同队没人亮且自己没亮过
            if self.phase == GamePhase.LIANGZHU:
                can_act = not my_player.ready
                # 同队已有人亮主，不能亮
                if self.liangzhu_player is not None and (seat % 2) == (self.liangzhu_player % 2):
                    can_act = False
            else:  # CHIPAI
                liang_team = self.liangzhu_player % 2 if self.liangzhu_player is not None else -1
                can_act = (
                    seat % 2 != liang_team
                    and self.chipai_claimed_by is None
                    and seat not in self.chipai_passed
                    and not any(c[0] == seat for c in self.liangzhu_candidates)
                )
            if can_act:
                if self.phase == GamePhase.CHIPAI:
                    liangzhu_available_types = self._find_chipai_liangzhu_candidates(my_player)
                else:
                    liangzhu_available_types = self._find_liangzhu_candidates(my_player)
                can_liangzhu = len(liangzhu_available_types) > 0

        # 亮主牌展示：扣底完成后(koupai_cards非空)/playing/game_over时对所有人展示
        # 规则11：扣底完成前，除亮主者外其他三人不知道亮主的具体牌型和花色
        visible_liangzhu_cards = []
        visible_liangzhu_type = self.liangzhu_type
        if self.phase in (GamePhase.PLAYING, GamePhase.GAME_OVER):
            visible_liangzhu_cards = [c.to_dict() for c in self.liangzhu_cards]
        elif self.phase in (GamePhase.KOUPAI, GamePhase.HUANPAI):
            # 扣底牌已出现（扣牌完成），亮主牌对所有人公开
            if self.koupai_cards:
                visible_liangzhu_cards = [c.to_dict() for c in self.liangzhu_cards]
            elif seat >= 0 and seat == self.liangzhu_player:
                visible_liangzhu_cards = [c.to_dict() for c in self.liangzhu_cards]
            else:
                visible_liangzhu_type = ''
        elif self.phase in (GamePhase.LIANGZHU, GamePhase.CHIPAI):
            # 扣底完成前，只对亮主者展示牌型和花色
            if seat >= 0 and seat == self.liangzhu_player:
                visible_liangzhu_cards = [c.to_dict() for c in self.liangzhu_cards]
            else:
                visible_liangzhu_type = ''
        
        # 返牌信息：只给被吃者展示
        visible_return_cards = []
        if self.chipai_return_cards and self.chipai_result:
            if seat >= 0 and seat == self.chipai_result.get('small_seat'):
                visible_return_cards = [c.to_dict() for c in self.chipai_return_cards]

        visible_huanpai_picked_cards = []
        visible_huanpai_return_cards = []
        visible_huanpai_return_candidates = []
        if seat >= 0 and seat == self.liangzhu_player:
            visible_huanpai_picked_cards = [c.to_dict() for c in self.huanpai_picked_cards]
            visible_huanpai_return_cards = [c.to_dict() for c in self.huanpai_return_cards]
            if self.phase == GamePhase.HUANPAI and my_player:
                visible_huanpai_return_candidates = [
                    c.to_dict() for c in self._get_huanpai_return_candidates(my_player, self.huanpai_offer)
                ]

        # 底牌：只给扣牌者展示（KOUPAI阶段）
        visible_hole_cards = []
        if self.phase == GamePhase.KOUPAI:
            picker = self._get_koupai_picker()
            if seat >= 0 and seat == picker:
                visible_hole_cards = [c.to_dict() for c in self.hole_cards]

        # 公开扣底：必须等扣牌/拾牌返牌流程彻底完成后再展示给所有人。
        # HUANPAI阶段底牌仍可能变化，因此不公开。
        public_koupai_cards = []
        if self.koupai_cards and self.phase in (GamePhase.PLAYING, GamePhase.GAME_OVER):
            public_koupai_cards = [c.to_dict() for c in self.koupai_cards]

        return {
            'room_id': self.room_id,
            'phase': self.phase.value if isinstance(self.phase, GamePhase) else self.phase,
            'players': players_info,
            'now_level': self.now_level,
            'now_color': visible_now_color,
            'score_now': self.score_now,
            'current_turn': self.current_turn,
            'epoch_cards': [[c.to_dict() for c in cards] for cards in self.epoch_cards],
            'epoch_players': self.epoch_players,
            'round_num': self.round_num,
            'hole_cards_count': len(self.hole_cards),
            'hole_cards': visible_hole_cards,
            'bankers': self.bankers,
            'my_seat': seat,
            'koupai_picker': koupai_picker,
            'bottom_picker': self.bottom_picker,
            'liangzhu_player': self.liangzhu_player,
            'liangzhu_type': visible_liangzhu_type,
            'last_epoch_cards': [[c.to_dict() for c in cards] for cards in self.last_epoch_cards],
            'last_epoch_players': self.last_epoch_players,
            'last_epoch_winner': self.last_epoch_winner,
            'game_result': game_result,
            'can_chipai_claim': can_chipai_claim,
            'can_chipai_pass': can_chipai_pass,
            'chipai_needs_action': can_chipai_pass,
            'chipai_claimed_by': self.chipai_claimed_by,
            'can_liangzhu': can_liangzhu,
            'liangzhu_available_types': liangzhu_available_types,
            'liangzhu_cards': visible_liangzhu_cards,
            'koupai_cards': public_koupai_cards,
            'public_koupai_cards': public_koupai_cards,
            'huanpai_offer': [c.to_dict() for c in self.huanpai_offer] if self.huanpai_offer else [],
            'huanpai_accepted': self.huanpai_accepted,
            'huanpai_picked_cards': visible_huanpai_picked_cards,
            'huanpai_return_cards': visible_huanpai_return_cards,
            'huanpai_return_candidates': visible_huanpai_return_candidates,
            'chipai_return_cards': visible_return_cards,
            'chipai_result': self.chipai_result,
        }

    def start_game(self) -> dict:
        """开始游戏"""
        # 补充机器人
        for i in range(4):
            if self.players[i] is None:
                self.players[i] = Player(
                    player_id=i, name=f'机器人{i + 1}',
                    is_robot=True, player_level=1
                )

        # 初始化
        self.deck = Deck()
        hands, self.hole_cards = self.deck.deal()
        self.now_level = self._banker_level_name()

        for i in range(4):
            self.players[i].cards_in_hand = hands[i]
            self.players[i].is_banker = False  # 会在_set_bankers中重新设置
            self.players[i].ready = False

        self.phase = GamePhase.LIANGZHU
        self.score_now = 0
        self.score_koupai = 0
        self.epoch_cards = []
        self.epoch_players = []
        self.epoch_count = 0
        self.round_num = 1
        self.game_num += 1  # 局数+1
        self.liangzhu_player = None
        self.liangzhu_cards = []
        self.liangzhu_type = ''
        self.koupai_cards = []
        self.game_result = None
        self._pending_clear_epoch = False
        self.liangzhu_candidates = []
        self.huanpai_offer = []
        self.huanpai_accepted = False
        self.huanpai_picked_cards = []
        self.huanpai_return_cards = []
        self.bottom_picker = None
        self.chipai_claimed_by = None
        self.chipai_passed = set()
        self.chipai_result = None
        self.chipai_return_cards = []
        for p in self.players:
            if p:
                p._chipai_evaluated = False

        # 计算底牌分数
        self.score_koupai = sum(SCORE_CARDS.get(c.name, 0) for c in self.hole_cards)

        return {'status': 'ok', 'phase': 'liangzhu'}

    def _banker_level_name(self) -> str:
        if self.bankers:
            seat = self.bankers[0]
            if 0 <= seat < len(self.players) and self.players[seat]:
                level_index = (self.players[seat].player_level - 1) % len(LEVEL_ORDER)
                return LEVEL_ORDER[level_index]
        return 'A'

    def handle_no_liang(self):
        """所有人都不亮主时，翻底牌定主色"""
        # 从底牌中取第一张非王牌确定主色
        if self.hole_cards:
            for hc in self.hole_cards:
                if not hc.is_joker:
                    self.now_color = hc.color
                    break
            else:
                self.now_color = None
        else:
            self.now_color = None

        self._set_bankers()
        self.phase = GamePhase.KOUPAI

        return {
            'status': 'ok',
            'now_color': self.now_color,
            'bankers': self.bankers,
            'msg': '无人亮主，翻底牌定主',
        }

    def handle_liangzhu(self, seat: int, card_strs: list[str], selected_color: Optional[str] = None) -> dict:
        """处理亮主 — 4人同时可亮，同队不能亮"""
        player = self.players[seat]
        if not player:
            return {'status': 'error', 'msg': '无效座位'}

        if self.phase not in (GamePhase.LIANGZHU, GamePhase.CHIPAI):
            return {'status': 'error', 'msg': '当前不是亮主阶段'}

        # CHIPAI阶段只有异队能亮主（吃牌）
        if self.phase == GamePhase.CHIPAI:
            liang_team = self.liangzhu_player % 2 if self.liangzhu_player is not None else -1
            if seat % 2 == liang_team:
                return {'status': 'error', 'msg': '只有对方可以吃牌'}

        # 同队已有人亮主，不能亮（LIANGZHU阶段）
        if self.phase == GamePhase.LIANGZHU and self.liangzhu_player is not None:
            if (seat % 2) == (self.liangzhu_player % 2):
                return {'status': 'error', 'msg': '队友已亮主，你不能亮'}

        # 已亮过主不能再亮
        if any(c[0] == seat for c in self.liangzhu_candidates):
            return {'status': 'error', 'msg': '你已经亮过主'}

        # 验证亮牌
        shot_cards = [card_from_str(s) for s in card_strs]

        # 检查牌是否在手中
        hand_strs = player.hand_to_str_list()
        for sc in card_strs:
            if sc not in hand_strs:
                return {'status': 'error', 'msg': '亮出的牌不在手中'}

        flag, liang_type = self._check_liang(shot_cards)
        if not flag:
            return {'status': 'error', 'msg': '亮牌不符合规则'}

        # 修正2：三王牌型需要手中有至少一个对子，对子花色为主花色
        threeking_pair_color = None
        if liang_type == 'threeking':
            # 检查玩家手牌中是否有对子（除已亮出的王之外）
            remaining = []
            for color in player.cards_in_hand:
                for c in player.cards_in_hand[color]:
                    if c.card_type not in card_strs:  # 排除已亮出的牌
                        remaining.append(c)
            # 按rank和color分组找对子
            pair_found = False
            best_pair_color = None
            for c in remaining:
                if c.is_joker:
                    continue
                if selected_color and c.color != selected_color:
                    continue
                # 检查同color同rank是否有2张
                same = [x for x in remaining if x.name == c.name and x.color == c.color and not x.is_joker]
                if len(same) >= 2 and c.color not in ('z',):
                    best_pair_color = c.color
                    pair_found = True
                    break
            if not pair_found:
                return {'status': 'error', 'msg': '三王亮主需要手中有至少一个对子'}
            threeking_pair_color = best_pair_color

        new_priority = LIANG_TYPE_RANK.get(liang_type, 99)
        if self.phase == GamePhase.CHIPAI:
            current_priority = LIANG_TYPE_RANK.get(self.liangzhu_type, 99)
            if new_priority >= current_priority:
                return {'status': 'error', 'msg': '吃牌牌型必须大于当前亮主牌型'}

        # 确定亮主花色
        liang_color = None
        if liang_type == 'threeking' and threeking_pair_color:
            # 修正2：三王牌型，主花色=手中对子的花色
            liang_color = threeking_pair_color
        else:
            for card in shot_cards:
                if not card.is_joker:
                    liang_color = card.color
                    break
        if selected_color and liang_type != 'threeking' and liang_color and selected_color != liang_color:
            return {'status': 'error', 'msg': '选择花色与亮主牌型不一致'}

        # 添加到候选人列表；第5项记录三王等牌型最终定主花色
        self.liangzhu_candidates.append((seat, shot_cards, liang_type, new_priority, liang_color))

        if self.phase == GamePhase.LIANGZHU:
            # 首次亮主：设置主花色和庄家
            self.liangzhu_player = seat
            self.liangzhu_cards = shot_cards
            self.liangzhu_type = liang_type
            self.now_color = liang_color
            self._set_bankers()

            # 进入吃牌阶段（等异队回应）
            self.phase = GamePhase.CHIPAI
            # 重置吃牌状态
            self.chipai_claimed_by = None
            self.chipai_passed = set()
            self.chipai_result = None
            self.chipai_return_cards = []
        # CHIPAI阶段：吃牌者亮主，只加入candidates，不覆盖liangzhu_player/now_color

        return {
            'status': 'ok',
            'liang_color': liang_color,
            'liang_type': liang_type,
            'bankers': self.bankers,
        }

    def _find_chipai_liangzhu_candidates(self, player: Player) -> list[dict]:
        """吃牌阶段只展示能压过当前亮主牌型的候选。"""
        current_priority = LIANG_TYPE_RANK.get(self.liangzhu_type, 99)
        return [
            candidate for candidate in self._find_liangzhu_candidates(player)
            if LIANG_TYPE_RANK.get(candidate.get('type'), 99) < current_priority
        ]

    def _infer_liang_color(self, cards: list[Card]) -> Optional[str]:
        for card in cards:
            if not card.is_joker:
                return card.color
        return None

    def _check_liang(self, shot_cards: list[Card]) -> tuple[bool, str]:
        """验证亮牌

        亮主条件：
        - 单对子(3张=1对+1王)：需要至少1张王
        - 双连对(5张=2对+1王)：需要至少1张王
        - 多连对(≥6张，纯副牌无王)：三连对及以上不需要王也可以亮
        - 三王(3张=3王)：三个王是一个牌型，大小比多连对小，比双连对大
        """
        num = len(shot_cards)
        flag = False
        liang_type = ''
        joker_count = sum(1 for c in shot_cards if c.is_joker)
        non_jokers = [c for c in shot_cards if not c.is_joker]

        # 三王牌型：3张全是王，但还需要手中至少有一个对子（对子花色为主花色）
        # 注意：提交的3张全是王，但需验证玩家手牌中有对子
        if num == 3 and joker_count == 3:
            flag = True
            liang_type = 'threeking'
        # 单对子+王：3张=1对同花色同rank+1张王
        elif num == 3 and joker_count >= 1:
            if len(non_jokers) == 2 and non_jokers[0].name == non_jokers[1].name and non_jokers[0].color == non_jokers[1].color:
                flag = True
                liang_type = 'danlian'
        # 双连对+王：5张=2对连对+1张王
        elif num == 5 and joker_count >= 1:
            if len(non_jokers) == 4:
                pairs = self._check_consecutive_pairs(non_jokers, 2)
                if pairs:
                    flag = True
                    liang_type = 'shuanglian'
        # 多连对（≥3连对）：不需要王
        elif num >= 6 and num % 2 == 0 and joker_count == 0:
            if len(non_jokers) == num:
                pair_count = num // 2
                pairs = self._check_consecutive_pairs(non_jokers, pair_count)
                if pairs:
                    flag = True
                    liang_type = 'duolian'
        # 多连对+王：5+2=7张等奇数情况不合法

        return flag, liang_type

    def _check_consecutive_pairs(self, non_jokers: list[Card], expected_pairs: int) -> bool:
        """检查非王牌是否构成expected_pairs个连续对子"""
        # 按rank分组
        rank_groups: dict[int, list[Card]] = {}
        for c in non_jokers:
            if c.rank not in rank_groups:
                rank_groups[c.rank] = []
            rank_groups[c.rank].append(c)

        # 每个rank必须有2张且同花色
        valid_pairs = []
        for rank in sorted(rank_groups.keys()):
            cards = rank_groups[rank]
            if len(cards) != 2:
                continue
            if cards[0].color != cards[1].color:
                continue
            valid_pairs.append((rank, cards[0].color))

        if len(valid_pairs) < expected_pairs:
            return False

        # 检查是否有expected_pairs个连续且同花色的对子
        # 找最长连续同花色序列
        if not valid_pairs:
            return False

        # 按花色分组
        color_pairs: dict[str, list[int]] = {}
        for rank, color in valid_pairs:
            if color not in color_pairs:
                color_pairs[color] = []
            color_pairs[color].append(rank)

        for color, ranks in color_pairs.items():
            ranks.sort()
            # 找长度>=expected_pairs的连续序列
            consec = 1
            for i in range(1, len(ranks)):
                if ranks[i] - ranks[i-1] == 1:
                    consec += 1
                else:
                    consec = 1
                if consec >= expected_pairs:
                    return True

        return False

    def _find_liangzhu_candidates(self, player: Player) -> list[dict]:
        """查找玩家手牌中所有可用的亮主牌型，返回 [{type, cards: [card_str]}]"""
        all_cards = []
        for cards in player.cards_in_hand.values():
            all_cards.extend(cards)

        candidates = []

        # 提取王牌
        jokers = [c for c in all_cards if c.is_joker]
        non_jokers = [c for c in all_cards if not c.is_joker]

        # 按花色+name分组找对子
        pair_map: dict[str, list[Card]] = {}  # key="color:name", value=[card1, card2]
        for card in non_jokers:
            key = f"{card.color}:{card.name}"
            if key not in pair_map:
                pair_map[key] = []
            pair_map[key].append(card)

        # 找到所有对子
        pairs = []
        for key, cards in pair_map.items():
            while len(cards) >= 2:
                pairs.append((cards[0], cards[1]))
                cards = cards[2:]
            pair_map[key] = cards

        # 按花色分组对子，找连对
        color_pairs: dict[str, list[tuple[Card, Card]]] = {}
        for c1, c2 in pairs:
            color = c1.color
            if color not in color_pairs:
                color_pairs[color] = []
            color_pairs[color].append((c1, c2))

        # 1. 多连对（duolian）: >=3连续对（无王），>=6张
        for color, pair_list in color_pairs.items():
            # 按rank排序
            pair_list_sorted = sorted(pair_list, key=lambda p: p[0].rank)
            # 找最长连续序列
            chains = self._find_consecutive_pairs(pair_list_sorted)
            for chain in chains:
                if len(chain) >= 3:  # 3连对=6张
                    chain_cards = []
                    for c1, c2 in chain:
                        chain_cards.extend([c1, c2])
                    card_strs = [c.card_type for c in chain_cards]
                    candidates.append({'type': 'duolian', 'cards': card_strs,
                                       'label': f'多连对({len(chain)}连)', 'color': color})

        # 2. 双连对（shuanglian）: 王x1 + 连对(>=2连续对，4张副牌)
        if len(jokers) >= 1:
            for color, pair_list in color_pairs.items():
                pair_list_sorted = sorted(pair_list, key=lambda p: p[0].rank)
                chains = self._find_consecutive_pairs(pair_list_sorted)
                for chain in chains:
                    if len(chain) >= 2:  # 2连对=4张副牌
                        chain_cards = []
                        for c1, c2 in chain:
                            chain_cards.extend([c1, c2])
                        all_cards_liang = chain_cards + [jokers[0]]
                        card_strs = [c.card_type for c in all_cards_liang]
                        candidates.append({'type': 'shuanglian', 'cards': card_strs,
                                           'label': '双连对', 'color': color})

        # 3. 三王（threeking）: 只亮3张王，主花色由手中可用对子决定
        if len(jokers) >= 3:
            seen_pair_keys = set()
            joker_cards = jokers[:3]
            for c1, c2 in pairs:
                pair_key = f"{c1.color}:{c1.name}"
                if pair_key in seen_pair_keys:
                    continue
                seen_pair_keys.add(pair_key)
                card_strs = [c.card_type for c in joker_cards]
                candidates.append({'type': 'threeking', 'cards': card_strs,
                                   'label': f'三王+{c1.name}对子定主', 'color': c1.color})

        # 4. 单连对（danlian）: 王x1 + 对子(2张)
        if len(jokers) >= 1:
            seen_pair_keys = set()
            for c1, c2 in pairs:
                pair_key = f"{c1.color}:{c1.name}"
                if pair_key in seen_pair_keys:
                    continue
                seen_pair_keys.add(pair_key)
                all_cards_liang = [c1, c2, jokers[0]]
                card_strs = [c.card_type for c in all_cards_liang]
                candidates.append({'type': 'danlian', 'cards': card_strs,
                                   'label': '单连对(对子+王)', 'color': c1.color})

        return candidates

    def _find_consecutive_pairs(self, pair_list_sorted: list[tuple[Card, Card]]) -> list[list[tuple[Card, Card]]]:
        """从已排序的对子列表中找到所有连续对序列"""
        if not pair_list_sorted:
            return []

        chains = []
        current_chain = [pair_list_sorted[0]]

        for i in range(1, len(pair_list_sorted)):
            prev_rank = current_chain[-1][0].rank
            curr_rank = pair_list_sorted[i][0].rank
            if curr_rank == prev_rank + 1:
                current_chain.append(pair_list_sorted[i])
            else:
                if len(current_chain) >= 2:
                    chains.append(current_chain[:])
                current_chain = [pair_list_sorted[i]]

        if len(current_chain) >= 2:
            chains.append(current_chain[:])

        return chains

    def suggest_play(self, seat: int) -> list[dict]:
        """推荐出牌：返回所有合法出牌选项，按AI策略排序（最优在前）

        返回 [{cards: [card_str], label: str}]
        """
        if self.phase != GamePhase.PLAYING or seat != self.current_turn:
            return []

        player = self.players[seat]
        if not player:
            return []

        from server.ai import AI
        ai = AI(player, self.now_level, self.now_color)
        is_first = len(self.epoch_cards) == 0

        # 获取AI推荐的首选出牌
        try:
            ai_choice = ai.decide_play(
                self.epoch_cards, self.epoch_players,
                is_first, self.score_now
            )
            ai_card_strs = [c.card_type for c in ai_choice] if ai_choice else []
        except Exception:
            ai_card_strs = []

        # 枚举所有合法出牌选项
        all_options = self._enumerate_legal_plays(player, is_first)

        # 将AI首选排到最前面，其余按牌力从小到大排列
        result = []
        if ai_card_strs:
            result.append({'cards': ai_card_strs, 'label': '💡推荐'})

        for opt in all_options:
            if opt['cards'] == ai_card_strs:
                continue  # 已添加为推荐
            result.append(opt)

        return result

    def _enumerate_legal_plays(self, player: Player, is_first: bool) -> list[dict]:
        """枚举所有合法出牌选项"""
        from server.rules import get_dui_and_liandui, card_type_analyze, determine_play_type
        from server.constants import LIANG_TYPE_RANK

        all_cards = []
        for cards in player.cards_in_hand.values():
            all_cards.extend(cards)

        if not all_cards:
            return []

        options = []

        if is_first:
            # 首位出牌：可以出任意合法牌型
            # 单牌
            for c in all_cards:
                options.append({'cards': [c.card_type], 'label': '单牌'})

            # 对子
            rank_groups = {}
            for c in all_cards:
                if c.name not in rank_groups:
                    rank_groups[c.name] = []
                rank_groups[c.name].append(c)

            for name, cards in rank_groups.items():
                while len(cards) >= 2:
                    pair = cards[:2]
                    if pair[0].color == pair[1].color:
                        options.append({'cards': [c.card_type for c in pair],
                                        'label': '对子'})
                    cards = cards[2:]

            # 连对（>=4张）
            dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
                player.cards_in_hand)
            for color, chains in liandui_dict.items():
                for chain in chains:
                    if len(chain) >= 4:
                        options.append({'cards': [c.card_type for c in chain],
                                        'label': f'连对({len(chain)//2}连)'})

        else:
            # 跟牌：必须跟出同花色、同数量
            first_cards = self.epoch_cards[0]
            first_type = determine_play_type(first_cards, self.now_level, self.now_color)
            n = len(first_cards)

            if first_type in ('fudan',):
                # 跟副单：必须出同花色副牌，或绝门出主单/其他副单
                first_color = first_cards[0].color
                # 只取同花色副牌（排除主牌——当主花色==首出花色时，主花色牌是主牌不是副牌）
                color_cards = [c for c in all_cards
                               if c.color == first_color
                               and not c.is_zhu(self.now_level, self.now_color)]
                if color_cards:
                    for c in color_cards:
                        options.append({'cards': [c.card_type], 'label': '跟牌'})
                else:
                    # 绝门（无同花色副牌）：可出任意牌
                    for c in all_cards:
                        options.append({'cards': [c.card_type], 'label': '绝门'})

            elif first_type in ('fudui',):
                # 跟副对：有对出对，没对出单
                first_color = first_cards[0].color
                # 只取同花色副牌（排除主牌）
                color_cards = [c for c in all_cards
                               if c.color == first_color
                               and not c.is_zhu(self.now_level, self.now_color)]
                # 两副牌：对子需同name同color
                rank_color_count = {}
                for c in color_cards:
                    key = (c.name, c.color)
                    if key not in rank_color_count:
                        rank_color_count[key] = []
                    rank_color_count[key].append(c)
                has_pair = any(len(v) >= 2 for v in rank_color_count.values())

                if has_pair:
                    for (name, color), cards in rank_color_count.items():
                        while len(cards) >= 2:
                            options.append({'cards': [cards[0].card_type, cards[1].card_type],
                                            'label': '跟对'})
                            cards = cards[2:]
                else:
                    # 没有同花色对子，需要凑2张
                    if len(color_cards) >= 2:
                        # 有2张以上同花色单牌
                        for i in range(len(color_cards)):
                            for j in range(i+1, min(len(color_cards), i+5)):
                                options.append({'cards': [color_cards[i].card_type, color_cards[j].card_type],
                                                'label': '跟牌(2单凑)'})
                    elif len(color_cards) == 1:
                        # 只有1张同花色，补1张其他牌
                        non_color = [c for c in all_cards if c not in color_cards]
                        for c in non_color[:5]:
                            options.append({'cards': [color_cards[0].card_type, c.card_type],
                                            'label': '跟牌(1同+1补)'})
                    else:
                        # 绝门，出任意2张
                        for i in range(min(len(all_cards), 8)):
                            for j in range(i+1, min(len(all_cards), i+5)):
                                options.append({'cards': [all_cards[i].card_type, all_cards[j].card_type],
                                                'label': '绝门出任意2张'})

            elif first_type.startswith('fulian') or first_type.startswith('zhulian'):
                # 跟连对
                if first_type.startswith('zhulian'):
                    # 主连对：筛选所有主牌（跨花色）
                    color_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
                else:
                    # 副连对：筛选同花色副牌（排除主牌）
                    first_color = first_cards[0].color
                    color_cards = [c for c in all_cards
                                   if c.color == first_color
                                   and not c.is_zhu(self.now_level, self.now_color)]
                if len(color_cards) >= n:
                    # 尝试凑连对
                    rank_groups = {}
                    for c in color_cards:
                        if c.name not in rank_groups:
                            rank_groups[c.name] = []
                        rank_groups[c.name].append(c)

                    # 找连续的对子
                    available_ranks = sorted(set(
                        r for r, cs in rank_groups.items() if len(cs) >= 2))
                    chain_len = n // 2

                    for start_i in range(len(available_ranks) - chain_len + 1):
                        chain_ranks = available_ranks[start_i:start_i + chain_len]
                        is_consecutive = all(
                            rank_groups[chain_ranks[j]][0].rank + 1 == rank_groups[chain_ranks[j + 1]][0].rank
                            for j in range(len(chain_ranks) - 1))
                        if is_consecutive:
                            chain_cards = []
                            for r in chain_ranks:
                                chain_cards.extend(rank_groups[r][:2])
                            options.append({'cards': [c.card_type for c in chain_cards],
                                            'label': '跟连对'})

                # 没有连对时，出同花色单牌
                if not options:
                    for c in color_cards[:n]:
                        options.append({'cards': [c.card_type for c in color_cards[:min(n, len(color_cards))]],
                                        'label': '跟牌'})
                    if not color_cards:
                        for c in all_cards:
                            options.append({'cards': [c.card_type], 'label': '绝门'})

            elif first_type in ('zhudan',):
                # 跟主单：出主牌
                zhu_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
                if zhu_cards:
                    for c in zhu_cards:
                        options.append({'cards': [c.card_type], 'label': '跟主牌'})
                else:
                    for c in all_cards:
                        options.append({'cards': [c.card_type], 'label': '出牌'})

            elif first_type in ('zhusan',):
                # 跟主散牌：有主牌先出主牌（最多n张），不够用副牌补
                # 规则：必须出n张牌（手牌足够时）
                zhu_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
                fu_cards = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)]
                must_zhu = min(len(zhu_cards), n)
                total_in_hand = len(all_cards)
                
                if total_in_hand < n:
                    # 手牌不够n张，全部出
                    options.append({'cards': [c.card_type for c in all_cards], 'label': '跟主散(全出)'})
                elif must_zhu >= n:
                    # 主牌够n张，出n张主牌
                    # 枚举不同组合（取前几组避免爆炸）
                    if n == 1:
                        for c in zhu_cards:
                            options.append({'cards': [c.card_type], 'label': '跟主散'})
                    elif n == 2:
                        for i in range(min(len(zhu_cards), 5)):
                            for j in range(i+1, min(len(zhu_cards), i+5)):
                                options.append({'cards': [zhu_cards[i].card_type, zhu_cards[j].card_type],
                                                'label': '跟主散'})
                    else:
                        options.append({'cards': [c.card_type for c in zhu_cards[:n]],
                                        'label': '跟主散'})
                elif must_zhu > 0:
                    # 有主牌但不够n张，出must_zhu张主牌+剩余用副牌补
                    remaining = n - must_zhu
                    if len(fu_cards) >= remaining:
                        # 枚举：前must_zhu张主牌 + 前remaining张副牌
                        base = [c.card_type for c in zhu_cards[:must_zhu]]
                        # 简化：取前remaining张副牌
                        for fi_start in range(min(len(fu_cards) - remaining + 1, 5)):
                            fu_pick = [c.card_type for c in fu_cards[fi_start:fi_start+remaining]]
                            options.append({'cards': base + fu_pick, 'label': '跟主散+副补'})
                    else:
                        # 副牌也不够，出所有牌
                        options.append({'cards': [c.card_type for c in all_cards[:n]], 'label': '跟主散(不够补)'})
                else:
                    # 无主牌，出n张副牌
                    if n == 1:
                        for c in fu_cards:
                            options.append({'cards': [c.card_type], 'label': '出副牌'})
                    elif n == 2:
                        for i in range(min(len(fu_cards), 5)):
                            for j in range(i+1, min(len(fu_cards), i+5)):
                                options.append({'cards': [fu_cards[i].card_type, fu_cards[j].card_type],
                                                'label': '出副牌'})
                    else:
                        options.append({'cards': [c.card_type for c in fu_cards[:n]], 'label': '出副牌'})

            elif first_type in ('fusan',):
                # 跟副散牌：有同花色副牌先出，不够用主牌/其他副牌补
                # 规则：必须出n张牌（手牌足够时），不足n张时全部出
                first_color = first_cards[0].color
                # 同花色副牌（排除主牌）
                color_fu = [c for c in all_cards
                            if c.color == first_color
                            and not c.is_zhu(self.now_level, self.now_color)]
                zhu_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
                other_fu = [c for c in all_cards
                            if c.color != first_color
                            and not c.is_zhu(self.now_level, self.now_color)]
                total_in_hand = len(all_cards)
                must_color = min(len(color_fu), n)
                
                if total_in_hand < n:
                    # 手牌不够n张，全部出
                    options.append({'cards': [c.card_type for c in all_cards], 'label': '跟副散(全出)'})
                elif len(color_fu) >= n:
                    # 有足够同花色副牌
                    if n == 1:
                        for c in color_fu:
                            options.append({'cards': [c.card_type], 'label': '跟副散'})
                    elif n == 2:
                        for i in range(min(len(color_fu), 5)):
                            for j in range(i+1, min(len(color_fu), i+5)):
                                options.append({'cards': [color_fu[i].card_type, color_fu[j].card_type],
                                                'label': '跟副散'})
                    else:
                        options.append({'cards': [c.card_type for c in color_fu[:n]],
                                        'label': '跟副散'})
                elif len(color_fu) > 0:
                    # 同花色副牌不够n张，必须先出完同花色，剩余用主牌补
                    remaining = n - len(color_fu)
                    if len(zhu_cards) >= remaining:
                        # 主牌够补
                        base = [c.card_type for c in color_fu]
                        if remaining == 1:
                            for zc in zhu_cards[:5]:
                                options.append({'cards': base + [zc.card_type], 'label': '跟副散+主补'})
                        else:
                            options.append({'cards': base + [c.card_type for c in zhu_cards[:remaining]],
                                            'label': '跟副散+主补'})
                    else:
                        # 主牌也不够补，出所有同花色+所有主牌+尽可能多的其他副牌
                        still_need = n - len(color_fu) - len(zhu_cards)
                        base = [c.card_type for c in color_fu] + [c.card_type for c in zhu_cards]
                        if still_need > 0 and len(other_fu) >= still_need:
                            base += [c.card_type for c in other_fu[:still_need]]
                        elif still_need > 0:
                            base += [c.card_type for c in other_fu]
                        options.append({'cards': base, 'label': '跟副散+主补+副补'})
                else:
                    # 绝门：无同花色副牌，必须先出主牌
                    must_zhu = min(len(zhu_cards), n)
                    if len(zhu_cards) >= n:
                        # 主牌够n张
                        if n == 1:
                            for c in zhu_cards:
                                options.append({'cards': [c.card_type], 'label': '绝门出主牌'})
                        elif n == 2:
                            for i in range(min(len(zhu_cards), 5)):
                                for j in range(i+1, min(len(zhu_cards), i+5)):
                                    options.append({'cards': [zhu_cards[i].card_type, zhu_cards[j].card_type],
                                                    'label': '绝门出主牌'})
                        else:
                            options.append({'cards': [c.card_type for c in zhu_cards[:n]], 'label': '绝门出主牌'})
                    elif must_zhu > 0:
                        # 主牌不够n张，补其他副牌
                        remaining = n - must_zhu
                        base = [c.card_type for c in zhu_cards[:must_zhu]]
                        if len(other_fu) >= remaining:
                            options.append({'cards': base + [c.card_type for c in other_fu[:remaining]],
                                            'label': '绝门主+副补'})
                        else:
                            options.append({'cards': base + [c.card_type for c in other_fu],
                                            'label': '绝门主+副补(不够)'})
                    else:
                        # 无主牌无同花色，出任意n张
                        options.append({'cards': [c.card_type for c in all_cards[:n]],
                                        'label': '绝门出任意牌'})

            elif first_type in ('zhudui',):
                # 跟主对
                zhu_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
                # 两副牌：对子需同name同color（如2-1-b×2），同name不同color不是对子
                rank_color_count = {}
                for c in zhu_cards:
                    key = (c.name, c.color)
                    if key not in rank_color_count:
                        rank_color_count[key] = []
                    rank_color_count[key].append(c)
                has_pair = any(len(v) >= 2 for v in rank_color_count.values())

                if has_pair:
                    for (name, color), cards in rank_color_count.items():
                        while len(cards) >= 2:
                            options.append({'cards': [cards[0].card_type, cards[1].card_type],
                                            'label': '跟主对'})
                            cards = cards[2:]
                else:
                    # 无主对时，必须出最大的2张主牌（_validate_follow要求）
                    # 两副牌中主牌按get_zhu_rank排序，取最大2张
                    zhu_sorted = sorted(zhu_cards, 
                                        key=lambda c: get_zhu_rank(c, self.now_level, self.now_color),
                                        reverse=True)
                    if len(zhu_sorted) >= 2:
                        options.append({'cards': [zhu_sorted[0].card_type, zhu_sorted[1].card_type],
                                        'label': '跟主牌(最大2张)'})
                    elif len(zhu_sorted) == 1:
                        # 只有1张主牌+1张副牌补齐
                        non_zhu = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)]
                        if non_zhu:
                            options.append({'cards': [zhu_sorted[0].card_type, non_zhu[0].card_type],
                                            'label': '跟主牌+副牌补'})
                        else:
                            options.append({'cards': [zhu_sorted[0].card_type],
                                            'label': '仅1张主牌'})
                    # 无主牌时出任意2张
                    if not zhu_sorted:
                        for i in range(min(len(all_cards), 2)):
                            for j in range(i+1, len(all_cards)):
                                options.append({'cards': [all_cards[i].card_type, all_cards[j].card_type],
                                                'label': '无主牌出任意2张'})

            else:
                # 其他情况：按同数量出
                for c in all_cards[:n]:
                    options.append({'cards': [c.card_type for c in all_cards[:min(n, len(all_cards))]],
                                    'label': '出牌'})

        # 去重
        seen = set()
        unique = []
        for opt in options:
            key = tuple(sorted(opt['cards']))
            if key not in seen:
                seen.add(key)
                unique.append(opt)

        return unique

    def _set_bankers(self):
        """设置庄家
        
        规则修正1：只有第一局(game_num==1)时亮主方=庄家，
        后续局庄家由上一局结算决定（保持不变）。
        """
        if self.game_num == 1:
            # 第一局：亮主方为庄家
            if self.liangzhu_player is not None:
                liang_seat = self.liangzhu_player
                partner = (liang_seat + 2) % 4
                self.bankers = [liang_seat, partner]
                for s in self.bankers:
                    self.players[s].is_banker = True
            else:
                # 无人亮主，随机选一对
                self.bankers = [0, 2]
                for s in self.bankers:
                    self.players[s].is_banker = True
        else:
            # 后续局：庄家由上一局结果决定，不变
            if not self.bankers:
                self.bankers = [0, 2]
            for s in self.bankers:
                self.players[s].is_banker = True

    def handle_chipai_claim(self, seat: int) -> dict:
        """吃牌：异队先到先得，只有1人能吃
        
        吃牌者亮主牌与被吃者亮主牌比较，牌型大的赢。
        赢者吃掉输者的亮主牌，然后返牌。
        """
        if self.phase != GamePhase.CHIPAI:
            return {'status': 'error', 'msg': '当前不是吃牌阶段'}
        
        if self.liangzhu_player is None:
            return {'status': 'error', 'msg': '无人亮主，不能吃牌'}
        
        # 检查是异队
        liang_team = self.liangzhu_player % 2
        if seat % 2 == liang_team:
            return {'status': 'error', 'msg': '同队不能吃牌'}
        
        # 已经有人吃了
        if self.chipai_claimed_by is not None:
            return {'status': 'error', 'msg': '已有人吃牌'}
        
        # 必须有亮主牌型才能吃
        player = self.players[seat]
        if not player:
            return {'status': 'error', 'msg': '无效座位'}
        
        # 标记吃牌者
        self.chipai_claimed_by = seat
        
        # 执行吃牌返牌逻辑
        result = self._execute_chipai()
        return result

    def handle_chipai_pass(self, seat: int) -> dict:
        """不吃：异队玩家选择不吃"""
        if self.phase != GamePhase.CHIPAI:
            return {'status': 'error', 'msg': '当前不是吃牌阶段'}
        
        if self.liangzhu_player is None:
            return {'status': 'error', 'msg': '无人亮主'}
        
        liang_team = self.liangzhu_player % 2
        if seat % 2 == liang_team:
            return {'status': 'error', 'msg': '同队不需要回应'}
        
        if self.chipai_claimed_by is not None:
            return {'status': 'error', 'msg': '已有人吃牌，不能再选择'}
        
        self.chipai_passed.add(seat)
        
        # 异队2人都不吃 → 跳过吃牌，进扣牌
        opponent_seats = [i for i in range(4) if i % 2 != liang_team]
        if all(s in self.chipai_passed for s in opponent_seats):
            self.phase = GamePhase.KOUPAI
            return {'status': 'ok', 'chipai': False, 'all_passed': True}
        
        return {'status': 'ok', 'chipai': False, 'passed': True}

    def _execute_chipai(self) -> dict:
        """执行吃牌返牌：吃牌者(chipai_claimed_by)与亮主者比较"""
        claimer = self.chipai_claimed_by
        original = self.liangzhu_player  # 原亮主者
        
        # 找到两人的亮主牌
        claimer_entry = None
        original_entry = None
        for c in self.liangzhu_candidates:
            if c[0] == claimer:
                claimer_entry = c
            elif c[0] == original:
                original_entry = c
        
        if not claimer_entry or not original_entry:
            return {'status': 'error', 'msg': '吃牌数据异常'}
        
        seat1, cards1, type1, prio1 = original_entry[:4]
        seat2, cards2, type2, prio2 = claimer_entry[:4]
        color1 = original_entry[4] if len(original_entry) > 4 else self._infer_liang_color(cards1)
        color2 = claimer_entry[4] if len(claimer_entry) > 4 else self._infer_liang_color(cards2)
        
        # 决定谁赢：优先级低（数值小）的赢
        if prio1 < prio2:
            big_seat, big_cards, small_seat, small_cards = seat1, cards1, seat2, cards2
        elif prio2 < prio1:
            big_seat, big_cards, small_seat, small_cards = seat2, cards2, seat1, cards1
        else:
            # 修正3：牌型相同时，先亮者大（original先亮，所以original赢）
            big_seat, big_cards, small_seat, small_cards = seat1, cards1, seat2, cards2

        player_big = self.players[big_seat]
        player_small = self.players[small_seat]

        # 吃牌：player_big获得player_small的亮主牌
        player_big.add_cards(small_cards)
        player_small.remove_cards(small_cards)

        # 返牌（修正5：返牌不影响吃牌结果，无吃牌失败）
        return_cards = self._select_return_cards(player_big, small_cards, big_cards)

        # 执行返牌
        if return_cards:
            player_big.remove_cards(return_cards)
            player_small.add_cards(return_cards)

        # 设定最终亮主者
        self.liangzhu_player = big_seat
        self.liangzhu_cards = big_cards
        self.liangzhu_type = type1 if big_seat == seat1 else type2
        self.now_color = color1 if big_seat == seat1 else color2
        self._set_bankers()

        # 保存吃牌结果和返牌信息（用于单独展示给被吃者）
        self.chipai_result = {
            'big_seat': big_seat,
            'small_seat': small_seat,
        }
        self.chipai_return_cards = return_cards

        self.phase = GamePhase.KOUPAI

        return {
            'status': 'ok',
            'chipai': True,
            'big_seat': big_seat,
            'small_seat': small_seat,
            'return_cards': [c.to_dict() for c in return_cards],
            'liang_color': self.now_color,
            'bankers': self.bankers,
        }

    def _select_return_cards(self, player_big: Player,
                              eaten_cards: list[Card],
                              big_liang_cards: list[Card]) -> list[Card]:
        """选择返牌：修正A规则

        返牌只有主牌和副牌两个层级：
        1. 吃的牌中有N张王牌 → 返N张主牌
        2. 吃的牌中有M张副牌 → 返M张同花色副牌（不要求同rank）
        3. 返牌总数=吃的牌数（吃的牌已先加入手牌，从中选即可保证总数）
        4. 返牌不影响吃牌结果，即使全返回去吃牌仍成功
        """
        joker_count = sum(1 for c in eaten_cards if c.is_joker)
        total_needed = len(eaten_cards)

        # 被吃者副牌的花色
        eaten_color = None
        for c in eaten_cards:
            if not c.is_joker:
                eaten_color = c.color
                break

        # 手牌中选返牌（吃的牌已通过add_cards加入手牌）
        all_cards = []
        for cards in player_big.cards_in_hand.values():
            all_cards.extend(cards)

        # 找出玩家手中的对子和连对，避免破坏
        pair_ranks = self._find_pair_ranks(player_big)
        liandui_ranks = self._find_liandui_ranks(player_big)
        protected = pair_ranks | liandui_ranks

        # 分类：主牌 / 副牌（同花色）/ 其他副牌
        zhu_cards = []
        eaten_color_fu = []
        other_fu = []

        big_liang_ids = set(id(c) for c in big_liang_cards)

        for card in all_cards:
            if id(card) in big_liang_ids:
                continue  # 排除亮出的牌
            if card.is_zhu(self.now_level, self.now_color):
                zhu_cards.append(card)
            elif eaten_color and card.color == eaten_color:
                eaten_color_fu.append(card)
            else:
                other_fu.append(card)

        # 排序：对子/连对中的牌排后面（优先保留），其余按rank升序（小牌优先返）
        def sort_key(card):
            is_protected = 1 if card.name in protected else 0
            return (is_protected, card.rank)

        selected = []
        used_ids = set()

        # 第一步：返joker_count张主牌（对应吃的王牌数）
        for card in sorted(zhu_cards, key=sort_key):
            zhu_selected = sum(1 for c in selected
                               if c.is_zhu(self.now_level, self.now_color))
            if zhu_selected >= joker_count:
                break
            selected.append(card)
            used_ids.add(id(card))

        # 第二步：返同花色副牌（凑够total_needed）
        for card in sorted(eaten_color_fu, key=sort_key):
            if id(card) in used_ids:
                continue
            if len(selected) >= total_needed:
                break
            selected.append(card)
            used_ids.add(id(card))

        # 第三步：如果还不够，用其他花色副牌补齐
        # （吃的牌已在手牌中，此步骤仅处理边缘情况如花色=主牌色）
        for card in sorted(other_fu, key=sort_key):
            if id(card) in used_ids:
                continue
            if len(selected) >= total_needed:
                break
            selected.append(card)
            used_ids.add(id(card))

        # 第四步：兜底——从剩余主牌中补齐
        for card in sorted(zhu_cards, key=sort_key):
            if id(card) in used_ids:
                continue
            if len(selected) >= total_needed:
                break
            selected.append(card)
            used_ids.add(id(card))

        return selected

    def _find_pair_ranks(self, player: Player) -> set:
        """找出玩家手中对子的rank集合（用于保护不被返走）"""
        pair_ranks = set()
        all_cards = []
        for cards in player.cards_in_hand.values():
            all_cards.extend(cards)
        # 按 name+color 分组
        groups = {}
        for c in all_cards:
            if c.is_joker:
                continue
            key = (c.name, c.color)
            if key not in groups:
                groups[key] = []
            groups[key].append(c)
        for key, cards in groups.items():
            if len(cards) >= 2:
                pair_ranks.add(cards[0].name)
        return pair_ranks

    def _find_liandui_ranks(self, player: Player) -> set:
        """找出玩家手中连对的rank集合（用于保护不被返走）"""
        liandui_ranks = set()
        all_cards = []
        for cards in player.cards_in_hand.values():
            all_cards.extend(cards)
        # 按花色分组
        color_groups = {}
        for c in all_cards:
            if c.is_joker:
                continue
            if c.color not in color_groups:
                color_groups[c.color] = {}
            if c.name not in color_groups[c.color]:
                color_groups[c.color][c.name] = 0
            color_groups[c.color][c.name] += 1
        # 每个花色找连续对子
        for color, rank_counts in color_groups.items():
            pair_ranks = sorted([name for name, cnt in rank_counts.items() if cnt >= 2],
                                key=lambda n: LEVEL_ORDER.index(n) if n in LEVEL_ORDER else 99)
            # 找连续对子（至少2连续）
            for i in range(len(pair_ranks) - 1):
                r1_idx = LEVEL_ORDER.index(pair_ranks[i]) if pair_ranks[i] in LEVEL_ORDER else -1
                r2_idx = LEVEL_ORDER.index(pair_ranks[i + 1]) if pair_ranks[i + 1] in LEVEL_ORDER else -1
                if r2_idx - r1_idx == 1:
                    liandui_ranks.add(pair_ranks[i])
                    liandui_ranks.add(pair_ranks[i + 1])
        return liandui_ranks

    def handle_shipai(self, seat: int, pick_strs: list[str], return_strs: list[str]) -> dict:
        """处理拾牌（原换牌阶段重写）

        规则：
        - 扣底完成后，亮主者可以选择拾起底牌中和亮主花色相同的牌
        - 拾起后必须还同rank的牌（不要求同花色）
        - 可以选择全拾或者拾一部分
        - 如果底牌没有亮主花色牌，跳过拾牌
        - 扣牌者（非亮主者）扣4张，亮主者拾牌
        """
        if self.phase != GamePhase.HUANPAI:
            return {'status': 'error', 'msg': '当前不是拾牌阶段'}

        if seat != self.liangzhu_player:
            return {'status': 'error', 'msg': '只有亮主玩家可以拾牌'}

        if not self.huanpai_offer:
            # 无亮主花色牌可拾，直接进入出牌
            self.phase = GamePhase.PLAYING
            self.current_turn = self.bottom_picker if self.bottom_picker is not None else (self.bankers[0] if self.bankers else 0)
            return {'status': 'ok', 'shipai': False}

        # 不拾（pick_strs为空）
        if not pick_strs:
            self.huanpai_accepted = False
            self.huanpai_picked_cards = []
            self.huanpai_return_cards = []
            self.phase = GamePhase.PLAYING
            self.current_turn = self.bottom_picker if self.bottom_picker is not None else (self.bankers[0] if self.bankers else 0)
            return {'status': 'ok', 'shipai': False, 'picked': False}

        # 验证拾牌
        pick_cards = [card_from_str(s) for s in pick_strs]

        # 拾的牌必须在offer中
        offer_counts = Counter(c.card_type for c in self.huanpai_offer)
        pick_counts = Counter(pick_strs)
        for ps, count in pick_counts.items():
            if count > offer_counts.get(ps, 0):
                return {'status': 'error', 'msg': '拾的牌不在底牌亮主花色牌中'}

        # 拾牌数量
        pick_count = len(pick_cards)

        # 验证还牌
        if not return_strs or len(return_strs) != pick_count:
            return {'status': 'error', 'msg': f'必须还{pick_count}张牌'}

        return_cards = [card_from_str(s) for s in return_strs]

        # 验证还牌在手中
        player = self.players[seat]
        hand_strs = player.hand_to_str_list()
        hand_counts = Counter(hand_strs)
        return_counts = Counter(return_strs)
        for cs, count in return_counts.items():
            if count > hand_counts.get(cs, 0):
                return {'status': 'error', 'msg': '还的牌不在手中'}

        # 验证还牌的rank必须和拾牌的rank一一对应（不要求同花色）
        pick_ranks = sorted([c.rank for c in pick_cards])
        return_ranks = sorted([c.rank for c in return_cards])
        if pick_ranks != return_ranks:
            return {'status': 'error', 'msg': '还的牌必须和拾的牌rank一致'}

        # 执行拾牌
        # 1. 亮主玩家获得拾起的牌
        player.add_cards(pick_cards)
        # 2. 亮主玩家扣出还牌
        player.remove_cards(return_cards)
        # 3. 更新底牌：用还牌替换拾起的牌
        for pick_card in pick_cards:
            if pick_card in self.koupai_cards:
                self.koupai_cards.remove(pick_card)
        self.koupai_cards.extend(return_cards)

        # 更新底牌分数
        self.score_koupai = sum(SCORE_CARDS.get(c.name, 0) for c in self.koupai_cards)

        self.huanpai_accepted = True
        self.huanpai_picked_cards = pick_cards
        self.huanpai_return_cards = return_cards  # 保存换出的牌用于展示
        self.bottom_picker = seat
        self.phase = GamePhase.PLAYING
        self.current_turn = seat

        return {
            'status': 'ok',
            'shipai': True,
            'picked': True,
            'pick_cards': [c.to_dict() for c in pick_cards],
            'return_cards': [c.to_dict() for c in return_cards],
        }

    def handle_koupai(self, seat: int, card_strs: list[str]) -> dict:
        """处理扣牌"""
        if self.phase != GamePhase.KOUPAI:
            return {'status': 'error', 'msg': '当前不是扣牌阶段'}

        # 确认是拾底牌者
        picker = self._get_koupai_picker()
        if seat != picker:
            return {'status': 'error', 'msg': '不是你扣牌'}

        player = self.players[seat]
        if not player:
            return {'status': 'error', 'msg': '无效座位'}

        # 验证牌数
        if len(card_strs) != 4:
            return {'status': 'error', 'msg': '必须扣4张牌'}

        # 扣牌者实际先拾起底牌，再从原手牌+底牌中扣回4张。
        available_counts = Counter(player.hand_to_str_list())
        available_counts.update(c.card_type for c in self.hole_cards)
        selected_counts = Counter(card_strs)
        for cs, count in selected_counts.items():
            if count > available_counts.get(cs, 0):
                return {'status': 'error', 'msg': '扣的牌不在手中'}

        cards_to_kou = [card_from_str(s) for s in card_strs]

        # 先加底牌到手中
        player.add_cards(self.hole_cards)
        self.bottom_picker = picker

        # 扣牌
        player.remove_cards(cards_to_kou)
        self.koupai_cards = cards_to_kou

        # 更新底牌分数
        self.score_koupai = sum(SCORE_CARDS.get(c.name, 0) for c in self.koupai_cards)
        self.huanpai_offer = []
        self.huanpai_picked_cards = []
        self.huanpai_return_cards = []

        # 检查扣回的牌中是否有亮主花色的牌 → 进入换牌阶段
        liang_color = self.now_color
        liang_player = self.liangzhu_player

        if liang_color and liang_player is not None:
            # 找出扣牌中亮主花色的牌
            self.huanpai_offer = [c for c in self.koupai_cards if c.color == liang_color and not c.is_joker]
            if self.huanpai_offer:
                # 有亮主花色牌，进入换牌阶段
                self.phase = GamePhase.HUANPAI
                self.current_turn = liang_player
                return {
                    'status': 'ok',
                    'koupai_count': len(cards_to_kou),
                    'huanpai': True,
                    'huanpai_offer': [c.to_dict() for c in self.huanpai_offer],
                    'huanpai_player': liang_player,
                }

        # 无亮主花色牌或无亮主者，直接进入出牌阶段
        self.phase = GamePhase.PLAYING
        self.current_turn = picker

        return {
            'status': 'ok',
            'koupai_count': len(cards_to_kou),
            'huanpai': False,
        }

    def _get_huanpai_return_candidates(self, player: Player, pick_cards: list[Card]) -> list[Card]:
        """Return hand cards that may be put back for the currently selected bottom picks."""
        if not pick_cards:
            return []
        needed_ranks = Counter(c.rank for c in pick_cards)
        candidates = []
        for cards in player.cards_in_hand.values():
            for card in cards:
                if needed_ranks.get(card.rank, 0) > 0:
                    candidates.append(card)
        return candidates

    def _get_koupai_picker(self) -> int:
        """获取拾底牌者：庄家中非亮主者，AI选手牌更好的，人类先选的优先"""
        if not self.bankers:
            return 0

        liang_seat = self.liangzhu_player
        # 庄家中排除亮主者
        picker_candidates = [b for b in self.bankers if b != liang_seat]

        if not picker_candidates:
            # 两个庄家都是亮主者（不应该发生），默认第一个
            return self.bankers[0]

        if len(picker_candidates) == 1:
            return picker_candidates[0]

        # 两个候选庄家，检查是否有人类玩家
        human_candidates = [b for b in picker_candidates
                          if self.players[b] and not self.players[b].is_robot]
        if human_candidates:
            # 人类玩家优先拾底牌
            return human_candidates[0]

        # 都是AI，选手牌更好的（出牌后能快速跑分）
        from server.ai import AI
        best_seat = picker_candidates[0]
        best_score = -1
        for b in picker_candidates:
            p = self.players[b]
            if p:
                ai = AI(p, self.now_level, self.now_color, self.score_koupai)
                score = ai.evaluate_hand()
                if score > best_score:
                    best_score = score
                    best_seat = b
        return best_seat

    def handle_play(self, seat: int, card_strs: list[str]) -> dict:
        """处理出牌"""
        # 清空上一轮的出牌记录（延迟清空机制，让前端能看到本轮出牌）
        if getattr(self, '_pending_clear_epoch', False):
            self.epoch_cards = []
            self._pending_clear_epoch = False

        if self.phase != GamePhase.PLAYING:
            return {'status': 'error', 'msg': '当前不是出牌阶段'}

        if seat != self.current_turn:
            return {'status': 'error', 'msg': '还没轮到你'}

        player = self.players[seat]
        if not player:
            return {'status': 'error', 'msg': '无效座位'}

        # 验证牌在手中（使用Counter防止出重复牌：手中1张出2张）
        from collections import Counter as _Counter
        hand_counts = _Counter(player.hand_to_str_list())
        play_counts = _Counter(card_strs)
        for cs, count in play_counts.items():
            if cs not in hand_counts:
                return {'status': 'error', 'msg': '出的牌不在手中'}
            if count > hand_counts[cs]:
                return {'status': 'error', 'msg': f'出的牌数量超过手中数量({cs})'}

        played_cards = [card_from_str(s) for s in card_strs]

        # 验证出牌合法性
        is_first = len(self.epoch_cards) == 0
        if not is_first:
            valid, msg = self._validate_follow(played_cards, self.epoch_cards[0], player)
            if not valid:
                return {'status': 'error', 'msg': msg}
        else:
            # 首位出牌验证牌型
            play_type = determine_play_type(played_cards, self.now_level, self.now_color)
            if play_type is None and len(played_cards) > 1:
                return {'status': 'error', 'msg': '出牌牌型不合法'}

        # 出牌
        player.remove_cards(played_cards)
        self.epoch_cards.append(played_cards)
        self.epoch_players.append(seat)
        self.epoch_count += 1

        result = {'status': 'ok', 'played': [c.to_dict() for c in played_cards]}

        # 修正C：每轮4人都要出牌，active_count永远为4
        active_count = 4

        # 一轮结束：4人都出了
        if self.epoch_count >= active_count:
            # 修正C：验证每轮每人出牌数相等（核心规则，必须断言）
            if self.epoch_cards:
                first_count = len(self.epoch_cards[0])
                for i, ec in enumerate(self.epoch_cards):
                    assert len(ec) == first_count, \
                        f'ROUND {self.round_num}: 出牌数不等! seat={self.epoch_players[i]} count={len(ec)} expected={first_count}'

            winner_seat, score_flag = self._resolve_epoch()
            result['epoch_result'] = {
                'winner': winner_seat,
                'score_flag': score_flag,
                'score_now': self.score_now,
            }

            # 修正C：游戏结束条件=所有人手牌出完（每轮出牌数相等，所有人同时出完）
            game_over = all(self.players[i].card_count == 0 for i in range(4))

            if game_over:
                # 修正C：所有人同时出完，手牌数必须全为0
                assert all(self.players[i].card_count == 0 for i in range(4)), \
                    f"游戏结束但有人还有手牌: {[self.players[i].card_count for i in range(4)]}"
                self._remember_epoch_result(winner_seat)
                game_result = self._end_game()
                result['game_over'] = game_result
            else:
                # 保存上一轮出牌记录（前端展示用）
                self._remember_epoch_result(winner_seat)
                # 下一轮（延迟清空epoch_cards，让前端能显示本轮出牌）
                self._pending_clear_epoch = True
                self.current_turn = winner_seat
                self.epoch_players = []
                self.epoch_count = 0
                self.round_num += 1
                result['next_turn'] = winner_seat
        else:
            # 下一个人出牌——跳过没有手牌的玩家
            # 修正C：可能存在有人先出完但其他人还有手牌的情况（如吃牌导致手牌数不一致）
            next_turn = (self.current_turn + 1) % 4
            skipped = 0
            while self.players[next_turn].card_count == 0 and skipped < 4:
                next_turn = (next_turn + 1) % 4
                skipped += 1

            # 如果所有人都空手牌，说明该轮所有人都出了→应该在epoch_count>=4时处理
            # 如果跳了4个人还在空手牌，说明游戏应该结束
            if skipped >= 4:
                # 强制结束
                self._remember_epoch_result(self.last_epoch_winner)
                game_result = self._end_game()
                result['game_over'] = game_result
                return result

            self.current_turn = next_turn
            result['next_turn'] = next_turn

        return result

    def _remember_epoch_result(self, winner_seat: int) -> None:
        """Keep the just-finished trick available for display and bottom scoring."""
        if not self.epoch_cards or not self.epoch_players:
            return
        self.last_epoch_cards = [list(cards) for cards in self.epoch_cards]
        self.last_epoch_players = list(self.epoch_players)
        self.last_epoch_winner = winner_seat

    def _validate_follow(self, played: list[Card], first_cards: list[Card],
                         player: Player) -> tuple[bool, str]:
        """验证跟牌合法性

        升级核心规则（规则10）：
        1. 副牌同花色优先，有同花色必须出相同花色的
        2. 同花色不够或者没有可以用其他牌代替
        3. 在副牌有同花色的基础上，不可以用主牌替代
        4. 如果首轮出了主牌，必须跟主牌，没有主牌时可用副牌替代
        5. 首轮出主牌对子，手里只有1张主牌，则必须先出1张主牌+剩余用副牌代替
        6. 首轮出副牌对子，手里只有1张该花色，必须先出这张+剩余用其他牌代替
        7. 跟牌数量必须与首出者相同（除非手牌不够凑齐）
        """
        first_type = determine_play_type(first_cards, self.now_level, self.now_color)
        n = len(first_cards)

        # 规则7：跟牌数量必须相同（手牌不够时允许出更少）
        if len(played) > n:
            return False, f'出牌数量({len(played)})不能多于首出者({n})'
        if len(played) < n and player.card_count >= n:
            return False, f'手牌足够，必须出{n}张牌'

        # 判断首出是否为主牌
        first_is_zhu = all(c.is_zhu(self.now_level, self.now_color) for c in first_cards)

        if first_is_zhu:
            # 规则4+5：首轮出主牌，必须跟主牌
            played_zhu = [c for c in played if c.is_zhu(self.now_level, self.now_color)]
            played_fu = [c for c in played if not c.is_zhu(self.now_level, self.now_color)]

            # 手中有主牌时，出的主牌数量必须等于手中主牌数和需求数的较小值
            hand_zhu_count = player.count_zhu(self.now_level, self.now_color)
            must_play_zhu = min(hand_zhu_count, n)

            if len(played_zhu) < must_play_zhu:
                return False, f'有主牌必须先出主牌（需出{must_play_zhu}张，实际{len(played_zhu)}张）'

            # 首出主对：有主对必须跟主对；无主对时必须出手里最大的主牌。
            if first_type == 'zhudui' and must_play_zhu > 0:
                hand_zhu_cards = self._get_player_zhu_cards(player)
                hand_zhu_pairs = self._get_zhu_pairs(hand_zhu_cards)
                played_zhu_pairs = self._get_zhu_pairs(played_zhu)

                if hand_zhu_pairs and not played_zhu_pairs and len(played_zhu) >= 2:
                    return False, '有主对必须出主对'

                if not hand_zhu_pairs:
                    required_top = self._top_zhu_cards(hand_zhu_cards, must_play_zhu)
                    required_counts = Counter(c.card_type for c in required_top)
                    played_counts = Counter(c.card_type for c in played_zhu)
                    for card_type, count in required_counts.items():
                        if played_counts.get(card_type, 0) < count:
                            required_text = ' '.join(c.card_type for c in required_top)
                            return False, f'无主对时必须出最大的主牌：{required_text}'

        else:
            # 首出是副牌
            first_color = first_cards[0].color
            hand_color_cards = player.cards_in_hand.get(first_color, [])

            # 排除该花色中的主牌（级牌/固定主等可能是同花色的主牌）
            hand_color_fu = [c for c in hand_color_cards
                            if not c.is_zhu(self.now_level, self.now_color)]
            hand_color_count = len(hand_color_fu)

            played_color_fu = [c for c in played
                              if c.color == first_color
                              and not c.is_zhu(self.now_level, self.now_color)]
            played_zhu_as_sub = [c for c in played
                                if c.is_zhu(self.now_level, self.now_color)]

            # 规则1+3：有同花色副牌必须出同花色副牌，不可用主牌替代
            must_play_color = min(hand_color_count, n)
            if len(played_color_fu) < must_play_color:
                return False, f'有同花色副牌必须出同花色（需出{must_play_color}张，实际{len(played_color_fu)}张）'

            # 规则3：在副牌有同花色的基础上，不可以用主牌替代
            # 如果出了同花色副牌还不够，且还有同花色副牌可出，不能用主牌补
            if played_zhu_as_sub and hand_color_count > len(played_color_fu):
                return False, '有同花色副牌时不可用主牌替代'

            # 规则6：首出副牌对子，手里只有1张该花色，必须先出这张+剩余用其他牌
            # (上面已确保出了must_play_color张同花色)

            # 对于副牌对子/连对的特殊处理
            if (first_type in ('fudui',) or first_type.startswith('fulian')) and hand_color_count >= 2:
                # 有足够同花色副牌，检查是否出了对子
                color_names: dict[str, int] = {}
                for c in hand_color_fu:
                    color_names[c.name] = color_names.get(c.name, 0) + 1
                has_pair = any(v >= 2 for v in color_names.values())
                if has_pair:
                    # 检查出的牌中是否有同花色对子
                    played_names: dict[str, int] = {}
                    for c in played_color_fu:
                        played_names[c.name] = played_names.get(c.name, 0) + 1
                    played_has_pair = any(v >= 2 for v in played_names.values())
                    if not played_has_pair and len(played) >= 2:
                        return False, '有同花色对子必须出对子'

        return True, ''

    def _get_player_zhu_cards(self, player: Player) -> list[Card]:
        zhu_cards: list[Card] = []
        for cards in player.cards_in_hand.values():
            zhu_cards.extend(c for c in cards if c.is_zhu(self.now_level, self.now_color))
        return zhu_cards

    def _get_zhu_pairs(self, cards: list[Card]) -> list[list[Card]]:
        groups: dict[str, list[Card]] = {}
        for card in cards:
            groups.setdefault(card.card_type, []).append(card)
        pairs: list[list[Card]] = []
        for group in groups.values():
            group = sorted(
                group,
                key=lambda c: (get_zhu_rank(c, self.now_level, self.now_color), c.rank),
                reverse=True,
            )
            for i in range(0, len(group) - 1, 2):
                pairs.append(group[i:i + 2])
        return pairs

    def _top_zhu_cards(self, cards: list[Card], count: int) -> list[Card]:
        return sorted(
            cards,
            key=lambda c: (get_zhu_rank(c, self.now_level, self.now_color), c.rank),
            reverse=True,
        )[:count]

    def _resolve_epoch(self) -> tuple[int, bool]:
        """结算一轮出牌，返回（赢家座位号，闲家是否赢）"""
        # 找最大牌
        max_idx = 0
        for i in range(1, len(self.epoch_cards)):
            if compare_outcards(self.epoch_cards[i], self.epoch_cards[max_idx],
                                self.now_level, self.now_color):
                max_idx = i

        winner_seat = self.epoch_players[max_idx]

        # 判断赢家是庄家还是闲家
        score_flag = winner_seat not in self.bankers

        # 闲家赢则计分
        if score_flag:
            all_cards = [c for cards in self.epoch_cards for c in cards]
            epoch_score = sum(SCORE_CARDS.get(c.name, 0) for c in all_cards)
            self.score_now += epoch_score

        return winner_seat, score_flag

    def _end_game(self) -> dict:
        """游戏结束结算"""
        # 扣底结算：升级标准规则
        # 最后一轮赢家的主牌决定倍数：ceil(赢家的主牌张数/2)
        # 副牌赢不触发扣底（倍数=0）
        # 闲家赢→闲家得分+底牌分×倍数，庄家赢→闲家得分-底牌分×倍数
        bottom_base_score = self.score_koupai
        koudi_multiplier = 0
        xianjia_koudi_score = 0
        banker_guard_score = 0
        if self.score_koupai > 0:
            last_winner = self.last_epoch_winner
            if last_winner >= 0:
                is_banker_winner = last_winner in self.bankers
                # 找赢家的出牌，计算其中主牌张数
                import math
                winner_zhu_count = 0
                if self.last_epoch_cards and self.last_epoch_players:
                    # last_epoch_players[i]对应last_epoch_cards[i]
                    for i, seat in enumerate(self.last_epoch_players):
                        if seat == last_winner and i < len(self.last_epoch_cards):
                            cards = self.last_epoch_cards[i]
                            winner_zhu_count = sum(1 for c in cards if c.is_zhu(self.now_level, self.now_color))
                            break

                koudi_multiplier = math.ceil(winner_zhu_count / 2) if winner_zhu_count > 0 else 0

                koupai_score = self.score_koupai * koudi_multiplier
                if not is_banker_winner:
                    # 闲家赢最后一轮：闲家得分增加
                    xianjia_koudi_score = koupai_score
                    self.score_now += xianjia_koudi_score
                else:
                    # 庄家赢最后一轮：闲家得分减少（最低0）
                    banker_guard_score = koupai_score
                    self.score_now = max(0, self.score_now - banker_guard_score)

        # 检查是否有庄家
        if not self.bankers:
            self.bankers = [0, 2]

        # 判定结果
        if self.score_now == 0:
            result_type = 'guangtou'
            levels = 4
            winner_team = 'banker'
        elif self.score_now < 40:
            result_type = 'xiaoguang'
            levels = 2
            winner_team = 'banker'
        elif self.score_now < 80:
            result_type = 'sheng1'
            levels = 1
            winner_team = 'banker'
        elif self.score_now < 120:
            result_type = 'duozhuang0'
            levels = 0
            winner_team = 'xianjia'
        elif self.score_now < 160:
            result_type = 'duozhuang1'
            levels = 1
            winner_team = 'xianjia'
        elif self.score_now < 200:
            result_type = 'duozhuang2'
            levels = 2
            winner_team = 'xianjia'
        else:
            result_type = 'quansheng'
            levels = 4
            winner_team = 'xianjia'

        # 更新级别
        new_bankers = list(self.bankers)
        if winner_team == 'banker':
            for s in self.bankers:
                self.players[s].player_level += levels
                if self.players[s].player_level > 13:
                    self.players[s].player_level -= 13
        else:
            xianjia_seats = [i for i in range(4) if i not in self.bankers]
            for s in xianjia_seats:
                self.players[s].player_level += levels
                if self.players[s].player_level > 13:
                    self.players[s].player_level -= 13
            new_bankers = xianjia_seats

        self.phase = GamePhase.GAME_OVER

        # 更新下一局庄家
        self.bankers = new_bankers

        result = {
            'score_now': self.score_now,
            'bottom_base_score': bottom_base_score,
            'koudi_multiplier': koudi_multiplier,
            'xianjia_koudi_score': xianjia_koudi_score,
            'banker_guard_score': banker_guard_score,
            'result_type': result_type,
            'levels': levels,
            'winner_team': winner_team,
            'new_bankers': new_bankers,
            'banker_levels': [self.players[s].player_level for s in self.bankers],
            'next_level': self._banker_level_name(),
        }
        self.game_result = result

        return result

    def auto_play_robots(self) -> list[dict]:
        """AI机器人自动操作，返回操作列表"""
        actions = []

        for i, p in enumerate(self.players):
            if not p or not p.is_robot:
                continue

            if self.phase == GamePhase.LIANGZHU:
                ai = AI(p, self.now_level, self.now_color, self.score_koupai)
                liang_cards = ai.decide_liangzhu()
                if liang_cards:
                    card_strs = [c.card_type for c in liang_cards]
                    result = self.handle_liangzhu(i, card_strs)
                    if result['status'] == 'ok':
                        actions.append({
                            'type': 'liangzhu',
                            'seat': i,
                            'cards': [c.to_dict() for c in liang_cards],
                            'result': result,
                        })

            elif self.phase == GamePhase.KOUPAI:
                picker = self._get_koupai_picker()
                if i == picker:
                    # 扣牌者不知道主花色（除非扣牌者就是亮主者）
                    picker_color = self.now_color if i == self.liangzhu_player else None
                    ai = AI(p, self.now_level, picker_color, self.score_koupai)
                    kou_cards = ai.decide_koupai(self.hole_cards)
                    card_strs = [c.card_type for c in kou_cards]
                    result = self.handle_koupai(i, card_strs)
                    if result['status'] == 'ok':
                        actions.append({
                            'type': 'koupai',
                            'seat': i,
                            'result': result,
                        })

            elif self.phase == GamePhase.PLAYING and i == self.current_turn:
                ai = AI(p, self.now_level, self.now_color, self.score_koupai)
                is_first = len(self.epoch_cards) == 0
                play_cards = ai.decide_play(
                    self.epoch_cards, [self.players[s] for s in self.epoch_players],
                    is_first, self.score_now
                )
                if play_cards:
                    card_strs = [c.card_type for c in play_cards]
                    result = self.handle_play(i, card_strs)
                    if result['status'] == 'ok':
                        actions.append({
                            'type': 'play',
                            'seat': i,
                            'cards': [c.to_dict() for c in play_cards],
                            'result': result,
                        })

        return actions

    def auto_play_current_robot(self) -> list[dict]:
        """机器人自动操作
        
        新流程：
        - LIANGZHU: 4个机器人同时评估，最强的亮主（同队不能亮）
        - CHIPAI: 异队机器人决定吃牌/不吃（先到先得）
        """
        actions = []

        if self.phase == GamePhase.LIANGZHU:
            # 所有机器人同时评估亮主意愿
            liang_candidates = []
            
            for i, p in enumerate(self.players):
                if p and p.is_robot and not p.ready:
                    # 同队已有人亮主，跳过
                    if self.liangzhu_player is not None and (i % 2) == (self.liangzhu_player % 2):
                        p.ready = True
                        continue
                    ai = AI(p, self.now_level, self.now_color, self.score_koupai)
                    liang_cards = ai.decide_liangzhu()
                    hand_score = ai.evaluate_hand()
                    p.ready = True
                    if liang_cards:
                        liang_candidates.append((i, liang_cards, hand_score))

            if liang_candidates:
                # 手牌最强的亮主
                liang_candidates.sort(key=lambda x: -x[2])
                best_seat, best_cards, _ = liang_candidates[0]
                card_strs = [c.card_type for c in best_cards]
                result = self.handle_liangzhu(best_seat, card_strs)
                if result['status'] == 'ok':
                    actions.append({
                        'type': 'liangzhu',
                        'seat': best_seat,
                        'cards': [c.to_dict() for c in best_cards],
                        'result': result,
                    })
            else:
                # 纯机器人局没人亮主
                all_ready = all(p.ready for p in self.players if p)
                if all_ready and self.liangzhu_player is None:
                    self.handle_no_liang()
                    # handle_no_liang后phase变成KOUPAI，继续处理
                    if self.phase == GamePhase.KOUPAI:
                        koupai_actions = self._auto_play_koupai()
                        actions.extend(koupai_actions)
                        return actions

        elif self.phase == GamePhase.CHIPAI:
            # 吃牌阶段：异队机器人决定吃牌或不吃
            if self.liangzhu_player is None:
                return actions
            
            if self.chipai_claimed_by is not None:
                # 已有人吃了，不需要操作
                return actions
            
            liang_team = self.liangzhu_player % 2
            opponent_seats = [i for i in range(4) if i % 2 != liang_team]
            
            # 检查异队是否有人类玩家
            has_human_opponent = any(
                self.players[s] and not self.players[s].is_robot 
                for s in opponent_seats if s not in self.chipai_passed
            )
            
            if has_human_opponent:
                # 等人类操作
                return actions
            
            # 纯机器人异队：最强手牌的先决定
            robot_decisions = []
            for s in opponent_seats:
                if s in self.chipai_passed:
                    continue
                p = self.players[s]
                if p and p.is_robot:
                    ai = AI(p, self.now_level, None, self.score_koupai)
                    liang_cards = ai.decide_liangzhu(chipai_mode=True)
                    hand_score = ai.evaluate_hand()
                    robot_decisions.append((s, liang_cards, hand_score))
            
            if not robot_decisions:
                # 异队都pass了
                if len(self.chipai_passed) >= len(opponent_seats):
                    self.phase = GamePhase.KOUPAI
                return actions
            
            # 手牌最强的先决定
            robot_decisions.sort(key=lambda x: -x[2])
            for s, liang_cards, _ in robot_decisions:
                if liang_cards:
                    # 吃牌：先亮主（记录到candidates），然后claim
                    card_strs = [c.card_type for c in liang_cards]
                    # 需要先handle_liangzhu把吃牌者的牌加入candidates
                    liang_result = self.handle_liangzhu(s, card_strs)
                    if liang_result['status'] == 'ok':
                        actions.append({
                            'type': 'liangzhu',
                            'seat': s,
                            'cards': [c.to_dict() for c in liang_cards],
                            'result': liang_result,
                        })
                        # 然后claim吃牌
                        claim_result = self.handle_chipai_claim(s)
                        if claim_result['status'] == 'ok':
                            actions.append({
                                'type': 'chipai_claim',
                                'seat': s,
                                'result': claim_result,
                            })
                        break  # 先到先得，一个人吃了就行
                    pass_result = self.handle_chipai_pass(s)
                    actions.append({
                        'type': 'chipai_pass',
                        'seat': s,
                        'result': pass_result,
                    })
                    if pass_result.get('all_passed'):
                        break
                else:
                    # 不吃
                    pass_result = self.handle_chipai_pass(s)
                    if pass_result.get('all_passed'):
                        # 两人都不吃，已自动进KOUPAI
                        actions.append({
                            'type': 'chipai_pass',
                            'seat': s,
                            'result': pass_result,
                        })
                        break

        elif self.phase == GamePhase.HUANPAI:
            # 拾牌：亮主玩家决定拾哪些牌+还哪些牌
            liang_seat = self.liangzhu_player
            p = self.players[liang_seat]
            if p and p.is_robot:
                ai = AI(p, self.now_level, self.now_color, self.score_koupai)
                pick_strs, return_strs = ai.decide_shipai(self.huanpai_offer, self.now_color)
                result = self.handle_shipai(liang_seat, pick_strs, return_strs)
                if result['status'] == 'ok':
                    actions.append({
                        'type': 'shipai',
                        'seat': liang_seat,
                        'result': result,
                    })

        elif self.phase == GamePhase.KOUPAI:
            actions.extend(self._auto_play_koupai())

        elif self.phase == GamePhase.PLAYING:
            # 先处理延迟清空，确保AI看到的状态与handle_play一致
            if getattr(self, '_pending_clear_epoch', False):
                self.epoch_cards = []
                self.epoch_players = []
                self._pending_clear_epoch = False

            # 跳过空手牌的玩家（修正C：吃牌可能导致手牌数不一致）
            turn = self.current_turn
            skipped = 0
            while self.players[turn].card_count == 0 and skipped < 4:
                turn = (turn + 1) % 4
                skipped += 1
            if skipped > 0:
                self.current_turn = turn

            p = self.players[self.current_turn]
            if p and p.is_robot:
                ai = AI(p, self.now_level, self.now_color, self.score_koupai)
                is_first = len(self.epoch_cards) == 0
                play_cards = ai.decide_play(
                    self.epoch_cards, [self.players[s] for s in self.epoch_players],
                    is_first, self.score_now
                )
                if play_cards:
                    card_strs = [c.card_type for c in play_cards]
                    result = self.handle_play(self.current_turn, card_strs)
                    if result['status'] == 'ok':
                        actions.append({
                            'type': 'play',
                            'seat': self.current_turn,
                            'cards': [c.to_dict() for c in play_cards],
                            'result': result,
                        })
                    else:
                        # AI出牌非法，用安全出牌fallback
                        fallback = self._safe_play_fallback(p)
                        if fallback:
                            result = self.handle_play(self.current_turn, fallback)
                            if result['status'] == 'ok':
                                actions.append({
                                    'type': 'play',
                                    'seat': self.current_turn,
                                    'result': result,
                                })
                else:
                    # AI返回空，用安全出牌fallback
                    fallback = self._safe_play_fallback(p)
                    if fallback:
                        result = self.handle_play(self.current_turn, fallback)
                        if result['status'] == 'ok':
                            actions.append({
                                'type': 'play',
                                'seat': self.current_turn,
                                'result': result,
                            })

        return actions

    def _auto_play_koupai(self) -> list[dict]:
        """KOUPAI阶段机器人自动扣牌"""
        actions = []
        picker = self._get_koupai_picker()
        p = self.players[picker]
        if p and p.is_robot:
            # 扣牌者不知道主花色（除非扣牌者就是亮主者）
            picker_color = self.now_color if picker == self.liangzhu_player else None
            ai = AI(p, self.now_level, picker_color, self.score_koupai)
            kou_cards = ai.decide_koupai(self.hole_cards)
            card_strs = [c.card_type for c in kou_cards]
            result = self.handle_koupai(picker, card_strs)
            if result['status'] == 'ok':
                actions.append({
                    'type': 'koupai',
                    'seat': picker,
                    'result': result,
                })
        return actions

    def _safe_play_fallback(self, player: Player) -> list[str]:
        """安全的出牌fallback：出最小的合法牌"""
        all_cards = []
        for cards in player.cards_in_hand.values():
            all_cards.extend(cards)

        if not all_cards:
            return []

        is_first = len(self.epoch_cards) == 0

        if is_first:
            # 首位出牌：出最小的单牌
            card = min(all_cards, key=lambda c: c.rank)
            return [card.card_type]

        # 跟牌
        first_cards = self.epoch_cards[0]
        first_n = len(first_cards)
        first_type = determine_play_type(first_cards, self.now_level, self.now_color)

        # 判断首出是否是主牌
        first_is_zhu = first_cards[0].is_zhu(self.now_level, self.now_color)

        if first_is_zhu:
            # 跟主牌：首出主对且自己无主对时，必须出最大的主牌
            zhu_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
            if first_type == 'zhudui' and zhu_cards and not self._get_zhu_pairs(zhu_cards):
                top_zhu = self._top_zhu_cards(zhu_cards, min(first_n, len(zhu_cards)))
                if len(top_zhu) >= first_n:
                    return [c.card_type for c in top_zhu[:first_n]]
                non_zhu = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)]
                non_zhu.sort(key=lambda c: c.rank)
                combined = top_zhu + non_zhu
                return [c.card_type for c in combined[:min(first_n, len(combined))]]

            if len(zhu_cards) >= first_n:
                zhu_cards.sort(key=lambda c: c.rank)
                return [c.card_type for c in zhu_cards[:first_n]]
            elif zhu_cards:
                # 主牌不够，补最小的副牌
                zhu_cards.sort(key=lambda c: c.rank)
                non_zhu = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)]
                non_zhu.sort(key=lambda c: c.rank)
                combined = zhu_cards + non_zhu
                n = min(first_n, len(combined))
                return [c.card_type for c in combined[:n]]
            else:
                # 无主牌，出最小副牌
                all_cards.sort(key=lambda c: c.rank)
                n = min(first_n, len(all_cards))
                return [c.card_type for c in all_cards[:n]]
        
        # 跟副牌
        first_color = first_cards[0].color
        same_color = [c for c in all_cards if c.color == first_color and not c.is_zhu(self.now_level, self.now_color)]

        if same_color and len(same_color) >= first_n:
            same_color.sort(key=lambda c: c.rank)
            return [c.card_type for c in same_color[:first_n]]

        # 同花色不够，补主牌凑齐
        if same_color:
            same_color.sort(key=lambda c: c.rank)
            remaining = first_n - len(same_color)
            zhu_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
            zhu_cards.sort(key=lambda c: c.rank)
            combined = same_color + zhu_cards[:remaining]
            n = min(first_n, len(combined))
            return [c.card_type for c in combined[:n]]

        # 绝门：出最小主牌
        zhu_cards = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
        if len(zhu_cards) >= first_n:
            zhu_cards.sort(key=lambda c: c.rank)
            return [c.card_type for c in zhu_cards[:first_n]]
        elif zhu_cards:
            zhu_cards.sort(key=lambda c: c.rank)
            non_zhu = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)]
            non_zhu.sort(key=lambda c: c.rank)
            combined = zhu_cards + non_zhu
            n = min(first_n, len(combined))
            return [c.card_type for c in combined[:n]]

        # 完全不够，出最小的牌（尽可能凑齐数量）
        all_cards.sort(key=lambda c: c.rank)
        n = min(first_n, len(all_cards))
        return [c.card_type for c in all_cards[:n]]
