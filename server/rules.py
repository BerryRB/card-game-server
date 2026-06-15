# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - 规则判定模块"""

from __future__ import annotations
from typing import Optional
from server.card import Card
from server.constants import FIXED_ZHU_NAMES, LIANG_TYPE_RANK


def get_dui_and_liandui(cards_in_hand: dict[str, list[Card]]):
    """从手牌中提取对子和连对"""
    dan_dict: dict[str, list[Card]] = {}
    dui_dict: dict[str, list[Card]] = {}
    liandui_dict: dict[str, list[list[Card]]] = {}

    all_cards: list[Card] = []
    for cards in cards_in_hand.values():
        all_cards.extend(cards)

    # 按花色分组（王不参与花色分组，单独提取到kings）
    color_cards: dict[str, list[Card]] = {}
    for card in all_cards:
        if card.is_joker:
            continue
        if card.color not in color_cards:
            color_cards[card.color] = []
        color_cards[card.color].append(card)

    for color, cards in color_cards.items():
        cards_sorted = sorted(cards, key=lambda c: c.rank)

        # 按rank分组
        rank_groups: dict[int, list[Card]] = {}
        for card in cards_sorted:
            if card.rank not in rank_groups:
                rank_groups[card.rank] = []
            rank_groups[card.rank].append(card)

        # 提取对子
        dui_list: list[Card] = []
        for rank, group in sorted(rank_groups.items()):
            while len(group) >= 2:
                dui_list.append(group[0])
                dui_list.append(group[1])
                group = group[2:]
                rank_groups[rank] = group
            dan_list_for_color = dan_dict.get(color, [])
            for c in group:
                dan_list_for_color.append(c)
            dan_dict[color] = dan_list_for_color

        dui_dict[color] = sorted(dui_list, key=lambda c: c.rank)

        # 提取连对
        liandui_list: list[list[Card]] = []
        sorted_dui = sorted(dui_list, key=lambda c: c.rank)

        if len(sorted_dui) >= 4:
            i = 0
            while i < len(sorted_dui):
                chain = [sorted_dui[i], sorted_dui[i + 1]] if i + 1 < len(sorted_dui) else []
                if len(chain) == 2 and chain[0].rank + 1 == chain[1].rank:
                    # 连对起点
                    j = i + 2
                    while j + 1 < len(sorted_dui) and sorted_dui[j].rank == sorted_dui[j - 2].rank + 1:
                        chain.append(sorted_dui[j])
                        chain.append(sorted_dui[j + 1])
                        j += 2
                    if len(chain) >= 4:
                        liandui_list.append(chain)
                    i = j
                else:
                    i += 2

        if liandui_list:
            liandui_dict[color] = liandui_list

    kings = []
    for card in all_cards:
        if card.is_joker:
            kings.append(card)

    return dan_dict, kings, dui_dict, liandui_dict


def card_type_analyze(dan_dict, kings, dui_dict, liandui_dict,
                      now_level: str, now_color: Optional[str] = None):
    """分析手牌类型，返回各种牌型列表"""

    zhudan = []
    zhudui = []
    zhuliandui = []
    fudan = {}
    fudui = {}
    fuliandui = {}

    # 分类副牌
    for color, cards in dan_dict.items():
        fu_cards = []
        for card in cards:
            if card.is_zhu(now_level, now_color):
                zhudan.append(card)
            else:
                fu_cards.append(card)
        if fu_cards:
            fudan[color] = sorted(fu_cards, key=lambda c: c.rank)

    for color, cards in dui_dict.items():
        fu_cards = []
        zhu_cards = []
        for card in cards:
            if card.is_zhu(now_level, now_color):
                zhu_cards.append(card)
            else:
                fu_cards.append(card)
        if fu_cards:
            fudui[color] = sorted(fu_cards, key=lambda c: c.rank)
        zhudui.extend(zhu_cards)

    for color, chains in liandui_dict.items():
        fu_chains = []
        zhu_chains = []
        for chain in chains:
            if any(c.is_zhu(now_level, now_color) for c in chain):
                zhu_chains.append(chain)
            else:
                fu_chains.append(chain)
        if fu_chains:
            fuliandui[color] = fu_chains
        zhuliandui.extend(zhu_chains)

    # 分类王：同card_type≥2的成对放入zhudui，单张放入zhudan
    kings_grouped: dict[str, list[Card]] = {}
    for card in kings:
        kings_grouped.setdefault(card.card_type, []).append(card)
    for card_type, group in kings_grouped.items():
        while len(group) >= 2:
            zhudui.extend(group[:2])
            group = group[2:]
        zhudan.extend(group)
    zhudan.sort(key=lambda c: c.rank)
    zhudui.sort(key=lambda c: c.rank)

    # 计算副牌最大单和主牌最大单
    fumax_dan = {}
    for color, cards in fudan.items():
        if cards:
            fumax_dan[color] = cards[-1]

    # 分离主连对
    zhuliandui_list = []
    for chain in zhuliandui:
        zhuliandui_list.append(chain)

    return (zhudan, zhudui, zhuliandui_list, fudan, fudui, fuliandui,
            len(zhudan) + len(zhudui), len(zhudan), fumax_dan)


def _compare_outcards_legacy(cards1: list[Card], cards2: list[Card],
                             now_level: str, now_color: Optional[str] = None) -> bool:
    """比较两组出牌大小，cards1 >= cards2 返回 True

    核心原则：
    1. 出牌数量必须相等（空牌=弃权，永远不赢）
    2. 牌型优先级：连对 > 多对 > 少对+散牌 > 全散牌
       对子越多越强：连对(全对+连续) > 2对(非连) > 1对+散牌 > 全散牌
    3. 同牌型层级内：主牌 > 副牌，然后再比大小
    """
    # 空牌 = 弃权/没手牌，永远不赢
    if not cards1 and not cards2:
        return True
    if not cards2:
        return True
    if not cards1:
        return False

    len1, len2 = len(cards1), len(cards2)

    # 数量不等时：数量多的赢（正常情况不应该出现，因为规则强制等量出牌）
    if len1 != len2:
        return len1 > len2

    # 数量相等，用牌型判断
    type1 = determine_play_type(cards1, now_level, now_color)
    type2 = determine_play_type(cards2, now_level, now_color)

    # 计算牌型层级：对子数决定层级，连对额外加分
    def type_priority(t):
        if t is None:
            return (0, 0)  # 未知牌型
        if t in ('zhudan', 'fudan'):
            return (1, 1)   # 单牌
        if t in ('zhusan', 'fusan'):
            return (1, 0)   # 2张散牌
        if t in ('zhudui', 'fudui'):
            return (2, 1)   # 1对
        if t.startswith('fulian') or t.startswith('zhulian'):
            # 连对：对子数从类型名提取，如fulian3=3连对
            try:
                pair_count = int(t.replace('fulian', '').replace('zhulian', ''))
            except ValueError:
                pair_count = 2
            return (100 + pair_count, 0)  # 连对层级高于任何非连对，3连对>2连对
        if '_san' in t:
            # 部分对子+散牌：如 fudui1_san, zhudui2_san
            base = t.split('dui')[1].split('_san')[0]
            pair_count = int(base)
            return (10 + pair_count, 1)  # 有pair_count对+散牌
        # 纯多对：如 fudui2, zhudui3
        if 'dui' in t:
            base = t.split('dui')[1]
            if base.isdigit():
                pair_count = int(base)
                return (20 + pair_count, 0)  # 纯多对
        # 全散牌：如 fusan4, zhusan6
        if 'san' in t:
            return (5, 0)
        return (0, 0)

    pri1 = type_priority(type1)
    pri2 = type_priority(type2)

    # 不同牌型层级：层级高的赢
    if pri1 != pri2:
        return pri1 > pri2

    # 同层级内比较
    all_zhu1 = all(c.is_zhu(now_level, now_color) for c in cards1)
    all_zhu2 = all(c.is_zhu(now_level, now_color) for c in cards2)

    # 主牌 vs 副牌
    if all_zhu1 and not all_zhu2:
        return True
    if not all_zhu1 and all_zhu2:
        return False

    # 都是主牌
    if all_zhu1 and all_zhu2:
        r1 = max(get_zhu_rank(c, now_level, now_color) for c in cards1)
        r2 = max(get_zhu_rank(c, now_level, now_color) for c in cards2)
        if r1 != r2:
            return r1 > r2
        return compare_zhu_cards(cards1[-1], cards2[-1], now_level, now_color)

    # 都是副牌
    # 升级规则：非首出花色的副牌只能垫牌，不能赢首出花色的副牌
    # cards2是首出(或当前最大)，cards1是后出
    if cards1[0].color != cards2[0].color:
        return False  # 异花色副牌不能赢首出花色副牌
    if cards1[0].color != cards2[0].color:
        return False
    return max(c.rank for c in cards1) > max(c.rank for c in cards2)


def compare_zhu_cards(c1: Card, c2: Card, now_level: str, now_color: Optional[str]) -> bool:
    """比较两张主牌大小"""
    r1 = get_zhu_rank(c1, now_level, now_color)
    r2 = get_zhu_rank(c2, now_level, now_color)
    if r1 != r2:
        return r1 > r2
    # 同rank，亮主花色的大
    if now_color:
        if c1.color == now_color and c2.color != now_color:
            return True
        if c1.color != now_color and c2.color == now_color:
            return False
    return False


def get_zhu_rank(card: Card, now_level: str, now_color: Optional[str]) -> int:
    """获取主牌的主牌排序rank（越大越强）"""
    if card.is_big_joker:
        return 100
    if card.is_small_joker:
        return 99
    if card.name == now_level:
        # 级牌：主花色级牌90，其他花色级牌80，确保不超过小王(99)
        # 不再加card.rank，避免A(rank=13)时103超过大王100
        return 80 + (10 if card.color == now_color else 0)
    if card.name == '5':
        return 60 + (10 if card.color == now_color else 0)
    if card.name == '3':
        return 40 + (10 if card.color == now_color else 0)
    if card.name == '2':
        return 20 + (10 if card.color == now_color else 0)
    if now_color and card.color == now_color:
        return card.rank
    return 0


def _count_pairs(cards: list[Card]) -> int:
    """Count exact pairs from two decks."""
    name_count: dict[str, list[Card]] = {}
    for c in cards:
        name_count.setdefault(c.card_type, []).append(c)
    pairs = 0
    for name, group in name_count.items():
        pairs += len(group) // 2
    return pairs


def _is_consecutive_pairs(cards: list[Card], now_level: str, now_color: str) -> bool:
    """判断一组牌是否构成连对（同花色+对子之间rank连续）
    
    修正6：首出多连对需要同花色
    """
    if len(cards) < 4 or len(cards) % 2 != 0:
        return False
    
    # 检查同花色（连对必须同花色）
    colors = set(c.color for c in cards if not c.is_joker)
    joker_count = sum(1 for c in cards if c.is_joker)
    if joker_count == 0 and len(colors) > 1:
        return False
    
    name_count: dict[str, list[Card]] = {}
    for c in cards:
        name_count.setdefault(c.name, []).append(c)
    # 每个name必须恰好2张才构成对子
    pairs_ranks = []
    for name, group in name_count.items():
        if len(group) != 2:
            return False
        pairs_ranks.append(group[0].rank)
    pairs_ranks.sort()
    # 检查连续性
    for i in range(1, len(pairs_ranks)):
        if pairs_ranks[i] != pairs_ranks[i - 1] + 1:
            return False
    return True


def determine_play_type(cards: list[Card], now_level: str,
                        now_color: Optional[str] = None) -> Optional[str]:
    """判断出牌的牌型

    牌型层级（从强到弱）：
    连对(liandui) > 多对(duodui) > 一对+散牌(yidui_sanpai) > 全散牌(sanpai)
    同层级内：主牌 > 副牌

    具体规则（以4张为例）：
    - 连对（如3344） → fulian/zhulian
    - 2对非连（如3355） → fudui2/zhudui2
    - 1对+2散牌（如33 5 7） → fudui1_san/zhudui1_san
    - 0对全散牌（如3 5 7 9） → fusan4/zhusan4

    6张/8张类似：连对 > 多对 > 少对+散牌 > 全散牌
    """
    if not cards:
        return None

    n = len(cards)
    all_zhu = all(c.is_zhu(now_level, now_color) for c in cards)
    prefix = 'zhu' if all_zhu else 'fu'

    if n == 1:
        return 'zhudan' if all_zhu else 'fudan'

    if n == 2:
        if cards[0].name == cards[1].name and cards[0].color == cards[1].color:
            return 'zhudui' if all_zhu else 'fudui'
        # 2张牌但不是对子 → 散牌组合
        return 'zhusan' if all_zhu else 'fusan'

    if n >= 4 and n % 2 == 0:
        # 检查连对
        if _is_consecutive_pairs(cards, now_level, now_color):
            pair_count = n // 2
            return f'{prefix}lian{pair_count}'  # 修正6：连对带对子数，如fulian3=3连对

        # 统计对子数
        pair_count = _count_pairs(cards)
        expected_pairs = n // 2  # 连对/全对子时的对子数

        if pair_count == expected_pairs:
            # 多对（非连对），如4张2对、6张3对
            return f'{prefix}dui{pair_count}'
        elif pair_count > 0:
            # 部分对子+散牌，如1对+2散牌
            return f'{prefix}dui{pair_count}_san'
        else:
            # 全散牌
            return f'{prefix}san{n}'

    # 奇数张牌（不常见但允许）
    if n >= 3:
        pair_count = _count_pairs(cards)
        if pair_count > 0:
            return f'{prefix}dui{pair_count}_san'
        return f'{prefix}san{n}'

    return None


def compare_outcards(cards1: list[Card], cards2: list[Card],
                     now_level: str, now_color: Optional[str] = None) -> bool:
    """Return True only when cards1 strictly beats cards2."""
    if not cards1 and not cards2:
        return False
    if not cards2:
        return True
    if not cards1:
        return False

    if len(cards1) != len(cards2):
        return len(cards1) > len(cards2)

    type1 = determine_play_type(cards1, now_level, now_color)
    type2 = determine_play_type(cards2, now_level, now_color)

    def type_priority(play_type):
        if play_type is None:
            return (0, 0)
        if play_type in ("zhudan", "fudan"):
            return (1, 1)
        if play_type in ("zhusan", "fusan"):
            return (1, 0)
        if play_type in ("zhudui", "fudui"):
            return (2, 1)
        if play_type.startswith("fulian") or play_type.startswith("zhulian"):
            try:
                pair_count = int(play_type.replace("fulian", "").replace("zhulian", ""))
            except ValueError:
                pair_count = 2
            return (100 + pair_count, 0)
        if "_san" in play_type:
            base = play_type.split("dui")[1].split("_san")[0]
            return (10 + int(base), 1)
        if "dui" in play_type:
            base = play_type.split("dui")[1]
            if base.isdigit():
                return (20 + int(base), 0)
        if "san" in play_type:
            return (5, 0)
        return (0, 0)

    pri1 = type_priority(type1)
    pri2 = type_priority(type2)
    if pri1 != pri2:
        return pri1 > pri2

    all_zhu1 = all(c.is_zhu(now_level, now_color) for c in cards1)
    all_zhu2 = all(c.is_zhu(now_level, now_color) for c in cards2)
    has_zhu1 = any(c.is_zhu(now_level, now_color) for c in cards1)
    has_zhu2 = any(c.is_zhu(now_level, now_color) for c in cards2)

    # 同牌型下，有主牌的一方胜过纯副牌（含混合牌）
    if all_zhu1 and not all_zhu2:
        return True
    if not all_zhu1 and all_zhu2:
        return False
    if has_zhu1 and not has_zhu2:
        return True
    if not has_zhu1 and has_zhu2:
        return False

    if all_zhu1 and all_zhu2:
        rank1 = max(get_zhu_rank(c, now_level, now_color) for c in cards1)
        rank2 = max(get_zhu_rank(c, now_level, now_color) for c in cards2)
        if rank1 != rank2:
            return rank1 > rank2
        return compare_zhu_cards(cards1[-1], cards2[-1], now_level, now_color)

    if cards1[0].color != cards2[0].color:
        return False
    return max(c.rank for c in cards1) > max(c.rank for c in cards2)
