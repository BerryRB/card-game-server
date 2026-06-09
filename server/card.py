# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - 服务端数据模型"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import random

from server.constants import RANK_MAP, SCORE_CARDS, FIXED_ZHU_NAMES


@dataclass
class Card:
    """单张扑克牌"""
    card_type: str  # "A-13-a"

    def __post_init__(self):
        parts = self.card_type.split('-')
        self.name: str = parts[0]
        self.rank: int = int(parts[1])
        self.color: str = parts[2]

    @property
    def is_joker(self) -> bool:
        return self.color == 'z'

    @property
    def is_small_joker(self) -> bool:
        return self.name == 'w'

    @property
    def is_big_joker(self) -> bool:
        return self.name == 'W'

    @property
    def score(self) -> int:
        return SCORE_CARDS.get(self.name, 0)

    @property
    def has_score(self) -> bool:
        return self.name in SCORE_CARDS

    def is_zhu(self, now_level: str, now_color: Optional[str] = None) -> bool:
        if self.is_joker:
            return True
        if self.name == now_level:
            return True
        if self.name in FIXED_ZHU_NAMES:
            return True
        if now_color and self.color == now_color:
            return True
        return False

    def is_fixed_zhu(self, now_level: str) -> bool:
        return self.name in FIXED_ZHU_NAMES or self.name == now_level or self.is_joker

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.name == other.name and self.rank == other.rank and self.color == other.color

    def __hash__(self):
        return hash((self.name, self.rank, self.color))

    def __repr__(self):
        if self.is_joker:
            return '小王' if self.is_small_joker else '大王'
        return f'{self.name}'

    def __lt__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank < other.rank

    def to_dict(self) -> dict:
        """序列化为字典（用于JSON传输）"""
        return {
            'card_type': self.card_type,
            'name': self.name,
            'rank': self.rank,
            'color': self.color,
        }

    @staticmethod
    def from_dict(d: dict) -> Card:
        return Card(d['card_type'])


class Deck:
    """两副扑克牌，108张"""

    CARD_TYPES = (
        ['2-1-a', '2-1-b', '2-1-c', '2-1-d'] +
        ['3-2-a', '3-2-b', '3-2-c', '3-2-d'] +
        ['4-3-a', '4-3-b', '4-3-c', '4-3-d'] +
        ['5-4-a', '5-4-b', '5-4-c', '5-4-d'] +
        ['6-5-a', '6-5-b', '6-5-c', '6-5-d'] +
        ['7-6-a', '7-6-b', '7-6-c', '7-6-d'] +
        ['8-7-a', '8-7-b', '8-7-c', '8-7-d'] +
        ['9-8-a', '9-8-b', '9-8-c', '9-8-d'] +
        ['10-9-a', '10-9-b', '10-9-c', '10-9-d'] +
        ['J-10-a', 'J-10-b', 'J-10-c', 'J-10-d'] +
        ['Q-11-a', 'Q-11-b', 'Q-11-c', 'Q-11-d'] +
        ['K-12-a', 'K-12-b', 'K-12-c', 'K-12-d'] +
        ['A-13-a', 'A-13-b', 'A-13-c', 'A-13-d'] +
        ['w-14-z', 'W-15-z']
    )

    def __init__(self):
        self.cards: list[Card] = []
        self.reset()

    def reset(self):
        self.cards = []
        for ct in self.CARD_TYPES:
            self.cards.append(Card(ct))
            self.cards.append(Card(ct))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self) -> tuple[list[dict[str, list[Card]]], list[Card]]:
        hands: list[dict[str, list[Card]]] = [{} for _ in range(4)]
        for i in range(4):
            player_cards = self.cards[i * 26:(i + 1) * 26]
            cards_dict: dict[str, list[Card]] = {}
            for card in player_cards:
                if card.color in cards_dict:
                    cards_dict[card.color].append(card)
                else:
                    cards_dict[card.color] = [card]
            for color in cards_dict:
                cards_dict[color].sort(key=lambda c: c.rank)
            hands[i] = dict(sorted(cards_dict.items()))
        hole_cards = sorted(self.cards[-4:], key=lambda c: c.rank)
        return hands, hole_cards


def card_from_str(s: str) -> Card:
    return Card(s)
