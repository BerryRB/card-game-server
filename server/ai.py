# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - AI出牌策略（v9真人高手版）

v9核心改进——基于100局数据分析，向真人高手思路对齐：

核心数据发现（v8的100局统计）：
1. 先手出分牌66.4%被对手收走 → 不主动出分牌，分牌留给跟牌时贴
2. 先手出副对82.5%被对手收走 → 副对几乎必被主对毙，不出副对先手
3. 先手出主单赢率仅29.6% → 主单先手不安全，主对/主连对才可靠
4. 清短门出大牌浪费 → 大牌留着跟牌管牌更有价值

真人高手核心原则：
1. 【不送分】先手绝不主动出分牌（5/10/K），分牌留给队友大时贴
2. 【副对慎出】副对先手几乎必被毙，只有AK连对/A对这种顶级副对才先手出
3. 【大牌跟牌】大牌（A/K/Q）留着跟牌时管住对手，比先手出更有效
4. 【清门出小】清短门出最小牌，保留大牌后续跟牌用
5. 【主牌控场】主对/主连对才是可靠的先手牌型
6. 【对子策略】出副对要出就出最大的，小对留着跟牌更安全
"""

from __future__ import annotations
from typing import Optional
from server.card import Card
from server.rules import (
    get_dui_and_liandui, card_type_analyze, compare_outcards,
    determine_play_type, get_zhu_rank
)
from server.constants import SCORE_CARDS, FIXED_ZHU_NAMES


class AI:
    """AI出牌策略（v9真人高手版）"""

    def __init__(self, player, now_level: str, now_color: Optional[str] = None,
                 score_koupai: int = 0):
        self.player = player
        self.now_level = now_level
        self.now_color = now_color
        self.score_koupai = score_koupai
        self._cache = None

    def _get_analysis(self):
        """获取并缓存手牌分析结果"""
        if self._cache is None:
            dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
                self.player.cards_in_hand)
            result = card_type_analyze(
                dan_dict, kings, dui_dict, liandui_dict,
                self.now_level, self.now_color)
            zhudan, zhudui, zhuliandui, fudan, fudui, fuliandui, zhu_count, zhu_dan_count, fumax_dan = result
            self._cache = {
                'dan_dict': dan_dict,
                'kings': kings,
                'dui_dict': dui_dict,
                'liandui_dict': liandui_dict,
                'zhudan': zhudan,
                'zhudui': zhudui,
                'zhuliandui': zhuliandui,
                'fudan': fudan,
                'fudui': fudui,
                'fuliandui': fuliandui,
                'zhu_count': zhu_count,
                'zhu_dan_count': zhu_dan_count,
                'fumax_dan': fumax_dan,
            }
        return self._cache

    # ========== 工具方法 ==========

    def _is_zhu(self, card: Card) -> bool:
        return card.is_zhu(self.now_level, self.now_color)

    def _is_endgame(self) -> bool:
        """判断是否进入尾局（剩余手牌≤6张）"""
        total = sum(len(cards) for cards in self.player.cards_in_hand.values())
        return total <= 6

    def _is_last_trick(self) -> bool:
        """判断是否是最后一轮（剩余手牌≤2张）"""
        total = sum(len(cards) for cards in self.player.cards_in_hand.values())
        return total <= 2

    def _is_near_end(self) -> bool:
        """判断是否进入尾局（剩余手牌≤6张，约最后3轮）"""
        total = sum(len(cards) for cards in self.player.cards_in_hand.values())
        return total <= 6

    def _is_team_winning(self, epoch_cards, epoch_players) -> bool:
        """判断队友是否在赢"""
        if not epoch_cards:
            return False
        valid_indices = [i for i, cards in enumerate(epoch_cards) if cards]
        if not valid_indices:
            return False
        max_idx = valid_indices[0]
        for i in valid_indices[1:]:
            if compare_outcards(epoch_cards[i], epoch_cards[max_idx],
                                self.now_level, self.now_color):
                max_idx = i
        if max_idx < len(epoch_players):
            winner = epoch_players[max_idx]
            return (winner.player_id % 2) == (self.player.player_id % 2)
        return False

    def _get_current_winning_card(self, epoch_cards) -> Optional[list[Card]]:
        """获取当前最大的出牌"""
        if not epoch_cards:
            return None
        valid_indices = [i for i, cards in enumerate(epoch_cards) if cards]
        if not valid_indices:
            return None
        max_idx = valid_indices[0]
        for i in valid_indices[1:]:
            if compare_outcards(epoch_cards[i], epoch_cards[max_idx],
                                self.now_level, self.now_color):
                max_idx = i
        return epoch_cards[max_idx]

    def _is_banker_team(self) -> bool:
        return self.player.is_banker

    def _get_color_card_count(self, color: str) -> int:
        return len(self.player.cards_in_hand.get(color, []))

    def _has_pair_in_color(self, color: str) -> bool:
        cards = self.player.cards_in_hand.get(color, [])
        from collections import Counter
        counts = Counter(c.name for c in cards)
        return any(v >= 2 for v in counts.values())

    def _score_of_cards(self, cards: list[Card]) -> int:
        return sum(SCORE_CARDS.get(c.name, 0) for c in cards)

    def _count_zhu_in_hand(self) -> int:
        count = 0
        for cards in self.player.cards_in_hand.values():
            for c in cards:
                if self._is_zhu(c):
                    count += 1
        return count

    def _count_juemen(self) -> int:
        """统计当前绝门数（不含主花色的花色中手牌为0的门数）"""
        count = 0
        for color in ['a', 'b', 'c', 'd']:
            if color != self.now_color:
                cards = self.player.cards_in_hand.get(color, [])
                if len(cards) == 0:
                    count += 1
        return count

    def _get_fudan_sorted(self) -> list[Card]:
        a = self._get_analysis()
        result = []
        for color in sorted(a['fudan'].keys()):
            result.extend(a['fudan'][color])
        return result

    def _get_score_fudan(self) -> list[Card]:
        a = self._get_analysis()
        result = []
        for color, cards in a['fudan'].items():
            for c in cards:
                if c.has_score:
                    result.append(c)
        return sorted(result, key=lambda c: c.rank)

    def _get_non_score_fudan(self) -> list[Card]:
        a = self._get_analysis()
        result = []
        for color, cards in a['fudan'].items():
            for c in cards:
                if not c.has_score:
                    result.append(c)
        return sorted(result, key=lambda c: c.rank)

    def _is_void_in_color(self, color: str) -> bool:
        a = self._get_analysis()
        return color not in a['fudan'] and color not in a['fudui'] and color not in a['fuliandui']

    def _count_colors_with_cards(self) -> int:
        a = self._get_analysis()
        colors = set()
        for d in (a['fudan'], a['fudui'], a['fuliandui']):
            for color in d:
                if d[color]:
                    colors.add(color)
        return len(colors)

    def _count_epoch_score(self, epoch_cards) -> int:
        total = 0
        for cards in epoch_cards:
            if cards:
                total += sum(SCORE_CARDS.get(c.name, 0) for c in cards)
        return total

    def _is_strong_pair(self, cards: list[Card]) -> bool:
        """判断一个副对是否足够强可以安全先手出
        A对/K对是强对（对手要用主对才能管住），Q对以下容易被大副对管
        """
        if not cards:
            return False
        top_rank = max(c.rank for c in cards)
        return top_rank >= 11  # K以上（K=11, A=12）

    def _is_strong_liandui(self, chain: list[Card]) -> bool:
        """判断副连对是否足够强可以安全先手出
        包含K或A的连对是强连对
        """
        return any(c.rank >= 11 for c in chain)

    # ========== 亮主策略 ==========

    def evaluate_hand(self) -> float:
        """评估手牌质量"""
        a = self._get_analysis()
        total = 0.0

        for chain in a['zhuliandui']:
            total += 60 * (len(chain) / 2)
        for i in range(0, len(a['zhudui']), 2):
            total += 25
        for card in a['zhudan']:
            if card.is_big_joker:
                total += 20
            elif card.is_small_joker:
                total += 15
            elif card.name == self.now_level and card.color == self.now_color:
                total += 12
            elif card.name == self.now_level:
                total += 8
            elif card.name in FIXED_ZHU_NAMES:
                total += 5

        for color, cards in a['fudan'].items():
            for c in cards:
                if c.has_score:
                    total += 3
                elif c.rank >= 10:
                    total += 1
        for color, cards in a['fudui'].items():
            for i in range(0, len(cards), 2):
                total += 5
        for color, chains in a['fuliandui'].items():
            for chain in chains:
                total += 15 * (len(chain) / 2)

        return total

    def decide_liangzhu(self, chipai_mode: bool = False) -> list[Card]:
        """决定是否亮主及亮什么牌
        
        chipai_mode: 吃牌返牌阶段，对手已亮主，异队需要更积极亮主来争夺
        """
        dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
            self.player.cards_in_hand)

        candidates = []

        for color, chains in liandui_dict.items():
            for chain in chains:
                if len(chain) >= 6:
                    fu_cards = [c for c in chain if not self._is_zhu(c)]
                    if fu_cards:
                        strength = self._evaluate_color_strength(color, dui_dict, liandui_dict, dan_dict)
                        candidates.append(('duolian', fu_cards[:6], 1, strength))

        if len(kings) >= 1:
            for color, chains in liandui_dict.items():
                for chain in chains:
                    if len(chain) >= 4:
                        fu_cards = [c for c in chain if not self._is_zhu(c)]
                        if fu_cards:
                            strength = self._evaluate_color_strength(color, dui_dict, liandui_dict, dan_dict)
                            candidates.append(('shuanglian', fu_cards[:4] + [kings[0]], 3, strength))

        if len(kings) >= 1:
            for color, chains in liandui_dict.items():
                for chain in chains:
                    fu_cards = [c for c in chain if not self._is_zhu(c)]
                    if fu_cards and len(fu_cards) >= 2:
                        strength = self._evaluate_color_strength(color, dui_dict, liandui_dict, dan_dict)
                        candidates.append(('danlian', fu_cards[:2] + [kings[0]], 4, strength))

        if len(kings) >= 1:
            for color, cards in dui_dict.items():
                for i in range(0, len(cards), 2):
                    if i + 1 < len(cards) and cards[i].name == cards[i + 1].name:
                        if not self._is_zhu(cards[i]):
                            strength = self._evaluate_color_strength(color, dui_dict, liandui_dict, dan_dict)
                            candidates.append(('danlian', [cards[i], cards[i + 1], kings[0]], 4, strength))

        if not candidates:
            return []

        hand_score = self.evaluate_hand()

        # v9.12: 亮主收益评估（修正过高bonus和过低风险）
        # 1. 换牌收益：实际换牌率55.8%，offer均2.3张，收益没原来估计那么高
        huanpai_bonus = 4.0  # v9.12: 8→4，更贴近实际

        # 2. 坐庄收益：做庄有扣底优势，但被吃牌风险大
        banker_bonus = 3.0   # v9.12: 5→3，被吃后庄家得额外牌抵消优势

        # 3. 被吃风险：danlian(4)最弱，几乎一定被吃（吃牌方100%是庄家队）
        #    v9.12: 大幅提高风险权重，弱牌亮主=给庄家送牌
        best_type_rank = min(c[2] for c in candidates)
        chipai_risk = best_type_rank * 5.0  # v9.12: 2→5，弱牌被吃风险巨大

        # 综合评估：手牌+收益-风险
        effective_score = hand_score + huanpai_bonus + banker_bonus - chipai_risk

        # v9.12: 提高亮主门槛——弱牌不亮，减少被吃送牌
        # chipai_mode：对手已亮主，不亮就被动，但仍需基本实力
        threshold = 30 if not chipai_mode else 18
        if effective_score < threshold:
            return []

        best = min(candidates, key=lambda x: (x[2], -x[3]))
        return best[1]

    def _evaluate_color_strength(self, color: str, dui_dict: dict,
                                  liandui_dict: dict, dan_dict: dict) -> float:
        strength = 0.0
        if color in dui_dict:
            cards = dui_dict[color]
            pair_count = len(cards) // 2
            strength += pair_count * 10
            for i in range(0, len(cards), 2):
                if i + 1 < len(cards) and cards[i].name == cards[i + 1].name:
                    if cards[i].rank >= 10:
                        strength += 5
        if color in liandui_dict:
            for chain in liandui_dict[color]:
                chain_len = len(chain) // 2
                strength += chain_len * 15
        if color in dan_dict:
            for c in dan_dict[color]:
                if c.rank >= 10:
                    strength += 3
        total_in_color = self._get_color_card_count(color)
        strength += total_in_color * 1.5
        return strength

    # ========== 换牌策略 ==========

    def decide_shipai(self, huanpai_offer: list[Card], now_color: str) -> tuple[list[str], list[str]]:
        """决定拾牌+还牌

        规则8：扣底完成后，亮主者可以选择拾起底牌中和亮主花色相同的牌。
        拾起后必须还同rank的牌（不要求同花色）。可选择全拾或部分拾。

        策略：基本总是拾起（多拿主牌有利），选还同rank最弱的牌。
        返回: (pick_strs, return_strs) — pick_strs为空表示不拾
        """
        if not huanpai_offer:
            return [], []

        # 评估offer牌的价值
        offer_value = 0
        for c in huanpai_offer:
            if c.has_score:
                offer_value += 5
            elif c.rank >= 10:
                offer_value += 3
            else:
                offer_value += 1

        # 如果offer总价值太低，不值得拾
        if offer_value < len(huanpai_offer) * 0.5:
            return [], []

        # 选择拾起的牌（全拾）
        pick_strs = [c.card_type for c in huanpai_offer]

        # 选择还牌：必须和拾的牌rank一致（不要求同花色）
        offer_ranks = set(c.rank for c in huanpai_offer)

        all_cards = []
        for cards in self.player.cards_in_hand.values():
            all_cards.extend(cards)

        # 按rank分组，每个rank需要还的牌数=offer中该rank的牌数
        from collections import Counter
        offer_rank_count = Counter(c.rank for c in huanpai_offer)
        selected = []
        for rank, count in offer_rank_count.items():
            # 找同rank的牌（不要求同花色），优先还弱的
            rank_cards = sorted([c for c in all_cards if c.rank == rank],
                               key=lambda c: (c.has_score, -c.rank))
            if len(rank_cards) < count:
                # 某个rank不够还，不拾
                return [], []
            selected.extend(rank_cards[:count])

        return pick_strs, [c.card_type for c in selected]

    def decide_huanpai(self, huanpai_offer: list[Card]) -> tuple[bool, list[str]]:
        """决定是否接受换牌（捡主返牌）

        亮主玩家看到扣牌中亮主花色的牌，决定是否捡起。
        如果接受，需返出同数量的非亮主花色牌。

        策略：基本总是接受（多拿主牌有利），除非offer牌太弱。
        返牌优先级：返副牌中最弱的（跟扣牌策略一致）。
        """
        if not huanpai_offer:
            return False, []

        # 评估offer中主牌的价值
        offer_value = 0
        for c in huanpai_offer:
            if c.has_score:
                offer_value += 5
            elif c.rank >= 10:
                offer_value += 3
            else:
                offer_value += 1

        # 如果offer总价值太低（都是小牌），不值得换
        # 但大多数情况下主牌总比副牌好，阈值设低
        if offer_value < len(huanpai_offer) * 0.5:
            return False, []

        # 选择返牌：必须和offer牌rank一致，且非亮主花色、非王牌
        liang_color = self.now_color
        offer_ranks = set(c.rank for c in huanpai_offer)

        # 按rank分组：找到手牌中每个offer rank对应的可返牌
        all_cards = []
        for cards in self.player.cards_in_hand.values():
            all_cards.extend(cards)

        # 可返的牌：非亮主花色、非王牌、rank在offer_ranks中
        candidates = [c for c in all_cards
                       if not c.is_joker and c.color != liang_color and c.rank in offer_ranks]

        # 按rank分组，每个rank需要返的牌数=offer中该rank的牌数
        from collections import Counter
        offer_rank_count = Counter(c.rank for c in huanpai_offer)
        selected = []
        for rank, count in offer_rank_count.items():
            rank_cards = sorted([c for c in candidates if c.rank == rank],
                               key=lambda c: (c.has_score, c.rank))
            if len(rank_cards) < count:
                # 某个rank不够返，不接受
                return False, []
            selected.extend(rank_cards[:count])

        return True, [c.card_type for c in selected]

    # ========== 扣牌策略 ==========

    def decide_koupai(self, hole_cards: list[Card]) -> list[Card]:
        """决定扣牌（v9: 延续v8少藏分策略，增加手牌结构优化）

        真人高手扣牌思路：
        1. 优先扣弱花色无分小牌（减少底牌分风险）
        2. 扣掉"断门"花色——让某花色完全绝门，方便毙牌
        3. 不拆对子和连对
        """
        all_cards = []
        for cards in self.player.cards_in_hand.values():
            all_cards.extend(cards)

        _, _, dui_dict, liandui_dict = get_dui_and_liandui(
            self.player.cards_in_hand)
        protected = set()
        for cards in dui_dict.values():
            for c in cards:
                protected.add(c)
        for chains in liandui_dict.values():
            for chain in chains:
                for c in chain:
                    protected.add(c)

        from collections import defaultdict
        color_strength = defaultdict(int)
        for color, cards in dui_dict.items():
            color_strength[color] += len(cards)
        for color, chains in liandui_dict.items():
            color_strength[color] += sum(len(c) for c in chains)

        # v9.9: 优先扣能形成绝门的花色——绝门战术价值远大于保留副对
        void_candidates = []
        for color in list(self.player.cards_in_hand.keys()):
            color_cards = [c for c in self.player.cards_in_hand[color] if not self._is_zhu(c)]
            # v9.9: 看所有副牌（含对子），如果≤4张就全扣实现绝门
            if 0 < len(color_cards) <= 4:
                # 优先扣无分牌
                nonscore = [c for c in color_cards if not c.has_score]
                score_cards = sorted([c for c in color_cards if c.has_score], 
                                    key=lambda c: (c.score, -c.rank))
                void_candidates.append((color, nonscore, score_cards, len(color_cards)))
        
        # v9.12: 优先双绝门——2个绝门比1个绝门战术价值更高
        # 排序：总牌数少的优先（2个2张花色 < 1个3张花色 < 1个4张花色）
        void_candidates.sort(key=lambda x: (x[3], -len(x[1])))
        partial_result = []  # 绝门扣底的部分结果（可能<4张）
        voided_colors = set()  # 已绝门花色
        for color, nonscore, score_cards, total in void_candidates:
            # v9.12: 全扣实现绝门，优先双绝门
            all_cards_in_color = nonscore + score_cards
            if len(all_cards_in_color) + len(partial_result) <= 4:
                # 全扣该花色实现绝门
                partial_result.extend(all_cards_in_color)
                voided_colors.add(color)
                if len(partial_result) >= 4:
                    return partial_result[:4]
                continue  # 继续找下一个绝门花色
            elif len(all_cards_in_color) <= 4 - len(partial_result):
                # 该花色能全扣但不够4张，扣部分
                partial_result.extend(all_cards_in_color)
                voided_colors.add(color)
                return partial_result[:4]
            else:
                # >4张不能全扣，优先扣无分
                if len(nonscore) >= 4 - len(partial_result):
                    partial_result.extend(nonscore[:4 - len(partial_result)])
                    return partial_result[:4]
                elif nonscore:
                    partial_result.extend(nonscore)
                    remaining = sorted(score_cards, key=lambda c: (c.score, -c.rank))
                    while len(partial_result) < 4 and remaining:
                        partial_result.append(remaining.pop(0))
                    if len(partial_result) >= 4:
                        return partial_result[:4]

        # 兜底：弱花色无分小牌（加上partial_result已有的牌）
        # 先排除partial_result已有的牌
        used_cards = set(id(c) for c in partial_result)
        nonscore_candidates = []
        for card in all_cards:
            if id(card) in used_cards:
                continue
            if card in protected:
                continue
            if self._is_zhu(card):
                continue
            if not card.has_score:
                nonscore_candidates.append(card)
        nonscore_candidates.sort(key=lambda c: (color_strength.get(c.color, 0), c.rank))

        if len(partial_result) + len(nonscore_candidates) >= 4:
            need = 4 - len(partial_result)
            partial_result.extend(nonscore_candidates[:need])
            return partial_result[:4]

        candidates = partial_result + nonscore_candidates[:]
        score_candidates = []
        for card in all_cards:
            if id(card) in used_cards or id(card) in set(id(c) for c in candidates):
                continue
            if card in protected:
                continue
            if self._is_zhu(card):
                continue
            if card.has_score:
                score_candidates.append(card)
        score_candidates.sort(key=lambda c: (color_strength.get(c.color, 0), c.score, -c.rank))
        while len(candidates) < 4 and score_candidates:
            c = score_candidates.pop(0)
            if id(c) not in set(id(x) for x in candidates):
                candidates.append(c)

        if len(candidates) >= 4:
            return candidates[:4]

        zhu_candidates = []
        for card in all_cards:
            if card in protected:
                continue
            if not self._is_zhu(card):
                continue
            if card not in candidates:
                zhu_candidates.append(card)
        zhu_candidates.sort(key=lambda c: c.rank)
        while len(candidates) < 4 and zhu_candidates:
            c = zhu_candidates.pop(0)
            if c not in candidates:
                candidates.append(c)

        remaining = [c for c in all_cards if c not in candidates]
        remaining.sort(key=lambda c: c.rank)
        while len(candidates) < 4 and remaining:
            candidates.append(remaining.pop(0))

        return candidates[:4]

    # ========== 出牌策略 ==========

    def decide_play(self, epoch_cards: list[list[Card]], epoch_players: list,
                    is_first: bool, now_scores: int) -> list[Card]:
        """决定出牌"""
        a = self._get_analysis()
        zhudan = a['zhudan']
        zhudui = a['zhudui']
        zhuliandui = a['zhuliandui']
        fudan = a['fudan']
        fudui = a['fudui']
        fuliandui = a['fuliandui']

        if is_first:
            return self._first_play(zhudan, zhudui, zhuliandui,
                                    fudan, fudui, fuliandui, now_scores)
        else:
            return self._follow_play(epoch_cards, epoch_players,
                                     zhudan, zhudui, zhuliandui,
                                     fudan, fudui, fuliandui, now_scores)

    # ========== 先手出牌策略 ==========

    def _first_play(self, zhudan, zhudui, zhuliandui,
                    fudan, fudui, fuliandui, now_scores) -> list[Card]:
        """首位出牌策略（v9真人高手版）

        核心原则（基于数据分析）：
        1. 不主动出分牌——66%会被对手收走
        2. 副对先手赢率极低(17%)，只有强对(K对/A对)才先手出
        3. 副单出无分小牌试探，大牌留着跟牌
        4. 清短门出小牌（不是大牌！）
        5. 主连对/主对是可靠的先手牌型
        """
        is_banker = self._is_banker_team()

        if is_banker:
            return self._first_play_banker(zhudan, zhudui, zhuliandui,
                                           fudan, fudui, fuliandui, now_scores)
        else:
            return self._first_play_defender(zhudan, zhudui, zhuliandui,
                                             fudan, fudui, fuliandui, now_scores)

    def _first_play_banker(self, zhudan, zhudui, zhuliandui,
                           fudan, fudui, fuliandui, now_scores) -> list[Card]:
        """庄家先手出牌策略（v9）

        庄家目标：控场、拿分、保过庄
        真人高手庄家思路：
        1. 主连对/主对先手——最可靠的赢轮方式
        2. 强副连对(AK连对等)——对手只能用主连对管
        3. 清短门出小牌——绝门后可以毙牌
        4. 出无分小副单试探——不送分
        """
        # v9.2: 庄家尾局策略——不管底牌分多少，都应出主牌争赢最后一轮
        if self._is_endgame():
            for chain in zhuliandui:
                return chain
            if len(zhudui) >= 2:
                return zhudui[-2:]
            if zhudan:
                return [zhudan[-1]]

        # 1. 主连对（最可靠的先手赢轮）
        for chain in zhuliandui:
            return chain

        # v9.8: 移除强副连对先手（从未触发）

        # 3. 主对（可靠赢轮）
        if len(zhudui) >= 2:
            return zhudui[-2:]

        # v9.8: 移除强副对先手（赢率25%太低，不如主单49.6%）
        # 强副对等跟牌时再出

        # v9.9: 先手主单策略——优先出级牌(赢率76%)，否则出最小主单(赢率35%但省牌)
        if zhudan:
            # 优先出级牌（赢率76%，远高于普通主牌35%）
            level_card = None
            for c in zhudan:
                if c.name == str(self.now_level):
                    level_card = c
                    break
            if level_card:
                return [level_card]
            return [zhudan[0]]  # 无级牌出最小主单

        # 5. 清短门——v9.10: 更精细的清门策略
        weak_colors = []
        for color in sorted(fudan.keys()):
            card_count = self._get_color_card_count(color)
            fu_dan_count = len(fudan.get(color, []))
            # v9.7: 放宽到5张以下
            if card_count <= 5 and fu_dan_count > 0:
                has_nonscore = any(not c.has_score for c in fudan[color])
                weak_colors.append((color, card_count, fu_dan_count, has_nonscore))
        if weak_colors:
            # v9.10: 优先清有无分牌的花色（出无分牌不送分）
            # 如果只剩分牌，跳过该花色（清门送分不划算，等绝门后毙牌）
            weak_colors.sort(key=lambda x: (not x[3], x[1]))  # 有无分牌优先，牌少的优先
            for color, _, _, has_nonscore in weak_colors:
                if has_nonscore:
                    nonscore = [c for c in fudan[color] if not c.has_score]
                    if nonscore:
                        return [nonscore[0]]  # 出最小无分牌
                # v9.10: 只剩分牌时也出（绝门价值>送5分，但不送10/K）
                score_cards = [c for c in fudan[color] if c.has_score]
                if score_cards and score_cards[0].score <= 5:
                    return [score_cards[0]]  # 出5分牌（只送5分换绝门）
                # K/10分牌不清门，太贵
            # 所有弱花色都只有高分牌，不清门，走后续逻辑

        # 6. 出无对子花色的小单
        for color in sorted(fudan.keys()):
            if fudan[color] and not self._has_pair_in_color(color):
                nonscore = [c for c in fudan[color] if not c.has_score]
                if nonscore:
                    return [nonscore[0]]
                return [fudan[color][0]]

        # 7. 出副小单
        for color in sorted(fudan.keys()):
            if fudan[color]:
                nonscore = [c for c in fudan[color] if not c.has_score]
                if nonscore:
                    return [nonscore[0]]
                return [fudan[color][0]]

        # 8. 弱副连对（虽然赢率不高，但比出副单强）
        for color in sorted(fuliandui.keys()):
            if fuliandui[color]:
                return fuliandui[color][0]

        # 9. 弱副对（最后才出）
        small_dui_color = None
        small_dui_rank = 999
        for color in sorted(fudui.keys()):
            if len(fudui[color]) >= 2:
                top_rank = fudui[color][0].rank
                if top_rank < small_dui_rank:
                    small_dui_rank = top_rank
                    small_dui_color = color
        if small_dui_color:
            return fudui[small_dui_color][:2]

        # 10. 主单（出最大的，争取赢轮）
        if zhudan:
            return [zhudan[-1]]

        return []

    def _first_play_defender(self, zhudan, zhudui, zhuliandui,
                             fudan, fudui, fuliandui, now_scores) -> list[Card]:
        """闲家先手出牌策略（v9）

        闲家目标：抢分、创造毙牌机会
        真人高手闲家思路：
        1. 清短门出小牌创造绝门→毙牌拿分
        2. 主连对/主对先手赢轮拿分
        3. 强副连对直接拿分
        4. 绝不出分牌先手——等队友大时贴分
        """
        zhu_count = self._count_zhu_in_hand()

        # 尾局策略：v9.2——闲家尾局也必须出主牌争最后一轮
        # 真人高手尾局思路：赢最后一轮=控制扣底，不管底牌分多少
        if self._is_endgame():
            for chain in zhuliandui:
                return chain
            if len(zhudui) >= 2:
                return zhudui[-2:]
            if zhudan:
                return [zhudan[-1]]

        # 1. 主连对/主对先手（赢率55%+，优先出）
        for chain in zhuliandui:
            return chain
        if len(zhudui) >= 2:
            return zhudui[-2:]

        # 2. ★清短门创造绝门（v9.11: 保留但微调）
        short_colors = []
        for color in sorted(fudan.keys()):
            card_count = self._get_color_card_count(color)
            fu_dan_count = len(fudan.get(color, []))
            fu_dui_count = len(fudui.get(color, []))
            if fu_dan_count > 0 and fu_dui_count == 0:
                if card_count <= 3:
                    short_colors.append((color, card_count, 0))
                elif card_count <= 5 and zhu_count >= 5:
                    short_colors.append((color, card_count, 1))
            elif fu_dan_count > 0 and fu_dui_count > 0 and card_count <= 4 and zhu_count >= 5:
                short_colors.append((color, card_count, 2))

        if short_colors:
            short_colors.sort(key=lambda x: (x[2], x[1]))
            for color, _, _ in short_colors:
                nonscore = [c for c in fudan[color] if not c.has_score]
                if nonscore:
                    return [nonscore[0]]
            for color, _, _ in short_colors:
                score_cards = [c for c in fudan[color] if c.has_score]
                if score_cards and score_cards[0].score <= 5:
                    return [score_cards[0]]

        # 3. 主单先手（v9.9: 优先级牌赢率76%，否则出最小）
        if zhudan:
            level_card = None
            for c in zhudan:
                if c.name == str(self.now_level):
                    level_card = c
                    break
            if level_card:
                return [level_card]
            return [zhudan[0]]

        # 4. 出副小单（最后手段）
        for color in sorted(fudan.keys()):
            if fudan[color]:
                nonscore = [c for c in fudan[color] if not c.has_score]
                if nonscore:
                    return [nonscore[0]]
                return [fudan[color][0]]

        return []

    # ========== 跟牌策略 ==========

    def _follow_play(self, epoch_cards, epoch_players,
                     zhudan, zhudui, zhuliandui,
                     fudan, fudui, fuliandui, now_scores) -> list[Card]:
        """跟牌策略（v9真人高手版）

        核心改进：
        1. 队友大时→积极贴分牌（这是分牌最安全的去处）
        2. 对手大时→用最小能管住的牌管（省大牌）
        3. 管不住→绝不出分牌（v8问题：管不住时出最小牌可能是5分）
        4. 绝门毙牌→出最小主牌（省大牌用于后续关键轮）
        5. v5: 返回牌数必须=首出牌数（手牌足够时）
        6. v5: 首出主牌时必须跟主牌（手中无主牌才能出副牌）
        """
        if not epoch_cards:
            return []

        first_cards = epoch_cards[0]
        first_type = determine_play_type(first_cards, self.now_level, self.now_color)
        n = len(first_cards)  # 需要跟的牌数

        if first_type == 'fudan':
            result = self._follow_fudan(first_cards[0], epoch_cards, epoch_players,
                                      zhudan, fudan, now_scores)
        elif first_type == 'fudui':
            result = self._follow_fudui(first_cards[0], epoch_cards, epoch_players,
                                      zhudui, fudui, zhudan, now_scores)
        elif first_type.startswith('fulian'):
            result = self._follow_fulian(first_cards, epoch_cards, epoch_players,
                                       zhuliandui, fuliandui, now_scores)
        elif first_type == 'zhudan':
            result = self._follow_zhudan(first_cards[0], epoch_cards, epoch_players,
                                       zhudan, now_scores)
        elif first_type == 'zhudui':
            result = self._follow_zhudui(first_cards, epoch_cards, epoch_players,
                                       zhudui, zhudan, now_scores)
        elif first_type.startswith('zhulian'):
            result = self._follow_zhulian(first_cards, epoch_cards, epoch_players,
                                       zhuliandui, now_scores)
        elif first_type in ('fusan', 'zhusan'):
            # v5: 散牌牌型（2+张不同名片），走generic
            result = self._follow_generic(first_cards, first_type, epoch_cards, epoch_players,
                                    zhudan, zhudui, fudan, fudui, now_scores)
        else:
            # 新牌型：多对/散牌/混合牌型的兜底处理
            result = self._follow_generic(first_cards, first_type, epoch_cards, epoch_players,
                                    zhudan, zhudui, fudan, fudui, now_scores)

        # v5后处理：确保返回牌数=首出牌数
        if len(result) != n:
            result = self._ensure_card_count(result or [], n, first_cards, first_type)

        # v5后处理：确保跟牌合法（修复AI选了同花色主牌而非副牌的bug）
        # is_first时不做验证（首出不受跟牌规则限制）
        if result and first_cards:  # first_cards非空说明是跟牌
            result = self._validate_and_fix_follow(result, first_cards, n)

        return result

    def _validate_and_fix_follow(self, result: list[Card], first_cards: list[Card], n: int) -> list[Card]:
        """v5后处理：验证跟牌合法性，修复AI选牌不符合引擎规则的问题

        常见问题：AI选了同花色主牌而非副牌（引擎要求优先出同花色副牌）
        """
        first_is_zhu = all(c.is_zhu(self.now_level, self.now_color) for c in first_cards)
        hand = self.player

        if not first_is_zhu and first_cards:
            # 首出副牌：检查是否出了足够的同花色副牌
            first_color = first_cards[0].color
            hand_color_cards = hand.cards_in_hand.get(first_color, [])
            hand_color_fu = [c for c in hand_color_cards
                            if not c.is_zhu(self.now_level, self.now_color)]

            # 计算需要出多少张同花色副牌
            must_play_color = min(len(hand_color_fu), n)

            # 计算实际出了多少张同花色副牌
            played_color_fu = [c for c in result
                              if c.color == first_color
                              and not c.is_zhu(self.now_level, self.now_color)]

            if len(played_color_fu) < must_play_color:
                # 不够！需要替换主牌为同花色副牌
                # 找出result中的同花色主牌（应该替换的）
                played_color_zhu = [c for c in result
                                   if c.color == first_color
                                   and c.is_zhu(self.now_level, self.now_color)]
                # 找出手牌中还未选的同花色副牌
                result_set = set(id(c) for c in result)
                available_color_fu = [c for c in hand_color_fu
                                     if id(c) not in result_set]

                # 替换：用同花色副牌替换同花色主牌
                new_result = list(result)
                for zhu_card in played_color_zhu:
                    if available_color_fu:
                        fu_card = available_color_fu.pop(0)
                        idx = new_result.index(zhu_card)
                        new_result[idx] = fu_card
                    else:
                        break
                return new_result

        elif first_is_zhu:
            # 首出主牌：检查是否出了足够的主牌
            hand_zhu_count = hand.count_zhu(self.now_level, self.now_color)
            must_play_zhu = min(hand_zhu_count, n)

            played_zhu = [c for c in result if c.is_zhu(self.now_level, self.now_color)]

            if len(played_zhu) < must_play_zhu:
                # 不够！需要替换副牌为主牌
                result_set = set(id(c) for c in result)
                available_zhu = []
                for cards in hand.cards_in_hand.values():
                    for c in cards:
                        if c.is_zhu(self.now_level, self.now_color) and id(c) not in result_set:
                            available_zhu.append(c)

                played_fu = [c for c in result if not c.is_zhu(self.now_level, self.now_color)]
                new_result = list(result)
                for fu_card in played_fu:
                    if available_zhu and len([c for c in new_result if c.is_zhu(self.now_level, self.now_color)]) < must_play_zhu:
                        zhu_card = available_zhu.pop(0)
                        idx = new_result.index(fu_card)
                        new_result[idx] = zhu_card
                    else:
                        break
                return new_result

        return result

    def _ensure_card_count(self, result: list[Card], n: int,
                           first_cards: list[Card], first_type: str) -> list[Card]:
        """v5后处理：确保跟牌数量=首出牌数

        规则：
        - 手牌足够时，必须出n张
        - 首出主牌时，优先补主牌，不够再补副牌
        - 首出副牌时，优先补同花色副牌，不够补主牌，再不够补其他副牌
        - 手牌不够时，出全部手牌
        """
        hand = self.player
        total_hand = hand.card_count

        if len(result) == n:
            return result

        # 需要补牌
        if len(result) < n:
            need = n - len(result)
            # 可用手牌（排除已选的）
            result_set = set(id(c) for c in result)

            first_is_zhu = all(c.is_zhu(self.now_level, self.now_color) for c in first_cards)

            if first_is_zhu:
                # 首出主牌：优先补主牌，再补副牌
                candidates = []
                for cards in hand.cards_in_hand.values():
                    for c in cards:
                        if id(c) not in result_set:
                            candidates.append(c)
                # 主牌排前面
                candidates.sort(key=lambda c: (not c.is_zhu(self.now_level, self.now_color), c.rank))
            else:
                # 首出副牌：优先补同花色副牌，再补主牌，再补其他
                first_color = first_cards[0].color
                candidates = []
                for cards in hand.cards_in_hand.values():
                    for c in cards:
                        if id(c) not in result_set:
                            candidates.append(c)
                # 同花色副牌 > 主牌 > 其他副牌
                def sort_key(c):
                    is_same_color_fu = (c.color == first_color and not c.is_zhu(self.now_level, self.now_color))
                    is_zhu = c.is_zhu(self.now_level, self.now_color)
                    return (not is_same_color_fu, not is_zhu, c.rank)
                candidates.sort(key=sort_key)

            # 最多补到手牌上限
            can_add = min(need, len(candidates), total_hand - len(result))
            result = list(result) + candidates[:can_add]

        elif len(result) > n:
            # 多了，截取前n张
            result = result[:n]

        return result

    def _follow_generic(self, first_cards, first_type, epoch_cards, epoch_players,
                        zhudan, zhudui, fudan, fudui, now_scores) -> list[Card]:
        """兜底跟牌策略：处理多对/散牌/混合牌型

        核心思路：
        - 对于新牌型（fudui2, fusan4, fudui1_san等），按首出花色和数量尽量跟牌
        - 如果是副牌牌型，找同花色等量牌跟
        - 如果是主牌牌型，找等量主牌跟
        - 实在凑不出，绝门时出最小主牌垫牌
        """
        n = len(first_cards)
        is_zhu_type = first_type and first_type.startswith('zhu')
        first_color = first_cards[0].color if first_cards else None

        # 确定是否需要跟同花色
        need_same_color = not is_zhu_type and first_color

        if need_same_color:
            # v5：副牌牌型，必须优先出同花色副牌
            color_cards = self.player.cards_in_hand.get(first_color, [])
            # 排除同花色中的主牌（引擎要求优先出同花色副牌）
            color_fu_cards = [c for c in color_cards
                             if not c.is_zhu(self.now_level, self.now_color)]
            
            if len(color_fu_cards) >= n:
                # 有足够同花色副牌
                sorted_cards = sorted(color_fu_cards, key=lambda c: c.rank)
                return sorted_cards[:n]
            elif color_fu_cards:
                # 同花色副牌不够n张：出所有同花色副牌 + 补主牌或副牌
                sorted_color = sorted(color_fu_cards, key=lambda c: c.rank)
                remaining = n - len(sorted_color)
                # 先补主牌
                zhu_cards = [c for c in zhudan]
                if len(zhu_cards) >= remaining:
                    return sorted_color + zhu_cards[:remaining]
                # 主牌不够，补其他副牌
                result_set = set(id(c) for c in sorted_color + zhu_cards)
                all_cards = []
                for cards in self.player.cards_in_hand.values():
                    for c in cards:
                        if id(c) not in result_set:
                            all_cards.append(c)
                other_cards = sorted(all_cards, key=lambda c: c.rank)
                need_more = n - len(sorted_color) - len(zhu_cards)
                return sorted_color + zhu_cards + other_cards[:need_more]
            else:
                # 无同花色副牌：绝门，出主牌+其他副牌
                all_cards = []
                for cards in self.player.cards_in_hand.values():
                    all_cards.extend(cards)
                sorted_all = sorted(all_cards, key=lambda c: (not c.is_zhu(self.now_level, self.now_color), c.rank))
                return sorted_all[:min(n, len(sorted_all))]

        if is_zhu_type:
            # v5：主牌牌型，必须先出主牌
            all_zhu = list(zhudan)
            for i in range(0, len(zhudui), 2):
                all_zhu.extend(zhudui[i:i+2])
            if len(all_zhu) >= n:
                sorted_zhu = sorted(all_zhu, key=lambda c: c.rank)
                return sorted_zhu[:n]
            elif all_zhu:
                # 主牌不够n张，出所有主牌+最小副牌补齐
                sorted_zhu = sorted(all_zhu, key=lambda c: c.rank)
                remaining = n - len(sorted_zhu)
                all_cards = []
                for cards in self.player.cards_in_hand.values():
                    for c in cards:
                        if id(c) not in set(id(x) for x in sorted_zhu):
                            all_cards.append(c)
                fu_cards = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)]
                fu_cards.sort(key=lambda c: c.rank)
                can_add = min(remaining, len(fu_cards))
                return sorted_zhu + fu_cards[:can_add]
            # 完全无主牌，出最小副牌
            all_cards = []
            for cards in self.player.cards_in_hand.values():
                all_cards.extend(cards)
            if all_cards:
                sorted_all = sorted(all_cards, key=lambda c: c.rank)
                return sorted_all[:min(n, len(sorted_all))]
            return []

        # 副牌绝门/凑不出：出最小牌垫牌
        all_cards = []
        for cards in self.player.cards_in_hand.values():
            all_cards.extend(cards)
        if all_cards:
            sorted_all = sorted(all_cards, key=lambda c: (c.is_zhu(self.now_level, self.now_color), c.rank))
            return sorted_all[:min(n, len(sorted_all))]

        return []

    def _has_color_cards(self, color: str) -> bool:
        """检查是否有指定花色的牌（含该花色的主牌）

        注意：引擎的player.has_color()检查的是cards_in_hand中该花色键下的所有牌，
        包括该花色的主牌（如级牌、2/3/5等）。当首出副牌时，引擎要求：
        有该花色时必须出同花色或主牌。所以即使该花色只有主牌（无副牌），
        引擎也认为"有同花色"，AI必须出该花色的主牌来跟牌。
        """
        cards = self.player.cards_in_hand.get(color, [])
        return len(cards) > 0

    def _get_color_single_cards(self, color: str, fudan, fudui, fuliandui, zhudan) -> list[Card]:
        """获取指定花色可用于跟副单的牌

        v9.13: 直接从cards_in_hand获取该花色所有牌（引擎验证也是基于cards_in_hand），
        这样无论牌被分类为副单/副对/主牌，都能正确返回。
        优先返回副单，然后副对拆出的单牌，最后主牌。
        """
        # 直接从手牌获取该花色所有牌
        all_color = self.player.cards_in_hand.get(color, [])
        if not all_color:
            return []
        return sorted(all_color, key=lambda c: (c.is_zhu(self.now_level, self.now_color), c.rank))

    def _follow_fudan(self, first_card, epoch_cards, epoch_players,
                      zhudan, fudan, now_scores) -> list[Card]:
        """跟副单（v9.13）

        关键改进：
        - v9.13: 修复同花色判断——当该花色只有对子/连对时，也视为"有同花色"，
          需要拆对出单牌，不能误判为绝门
        - 队友大→优先贴分牌（分牌的最佳去处！）
        - 对手大+有同花色→用最小能管住的牌管，管不住出最小无分牌
        - 绝门+对手大→毙牌（出最小主牌）
        - 绝门+队友大→贴分牌（不是副小牌）
        """
        color = first_card.color
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)
        is_banker = self._is_banker_team()

        # v9.13: 用_has_color_cards判断是否有同花色（含对子/连对/主牌）
        # 同时获取可用于跟单牌的散牌列表（含拆对出的牌+该花色主牌）
        a = self._get_analysis()
        has_color = self._has_color_cards(color)
        color_singles = self._get_color_single_cards(color, fudan, a['fudui'], a['fuliandui'], zhudan)

        if has_color and color_singles:
            # 有同花色（v9.13: color_singles包含从对子/连对拆出的散牌）
            if my_team_winning:
                # 队友大→v9: 积极贴分牌（分牌最安全的去处）
                score_cards = [c for c in color_singles if c.has_score]
                if score_cards:
                    return [score_cards[0]]  # 出最小分牌
                return [color_singles[0]]  # 没分牌出最小
            else:
                # 对手大→出最小能管住的牌
                current_best = self._get_current_winning_card(epoch_cards)
                epoch_score = self._count_epoch_score(epoch_cards)

                if current_best:
                    # v9: 不管什么身份，有分就认真管（≥5分）
                    should_play_big = epoch_score >= 5
                    search_order = reversed(color_singles) if should_play_big else color_singles
                    for card in search_order:
                        if compare_outcards([card], current_best,
                                            self.now_level, self.now_color):
                            return [card]
                else:
                    for card in color_singles:
                        if compare_outcards([card], [first_card],
                                            self.now_level, self.now_color):
                            return [card]
                # 管不住→v9: 出最小无分牌，绝不送分！
                non_score = [c for c in color_singles if not c.has_score]
                if non_score:
                    return [non_score[0]]
                # 全是分牌没办法
                return [color_singles[0]]

        # 绝门
        if my_team_winning:
            # v9.4: 只有最后一轮+底牌有分时不贴分（防扣底）
            # 之前最后3轮都不贴分太保守了
            if self._is_last_trick() and self.score_koupai > 0:
                non_score_fudan = self._get_non_score_fudan()
                if non_score_fudan:
                    return [non_score_fudan[0]]
                for c in sorted(fudan.keys()):
                    if fudan[c]:
                        return [fudan[c][0]]
            # v9.11: 队友大→优先贴短门花色的分牌（创造更多绝门）
            # 短门花色出完=绝门+1，后续毙牌机会大增
            score_fudan = self._get_score_fudan()
            if score_fudan:
                # 按花色牌数排序，短门优先
                scored_by_color = {}
                for c in score_fudan:
                    color = c.color
                    if color not in scored_by_color:
                        scored_by_color[color] = []
                    scored_by_color[color].append(c)
                
                # 短门优先：该花色总牌数少的先贴
                colors_by_len = sorted(scored_by_color.keys(),
                    key=lambda cl: self._get_color_card_count(cl))
                # 短门花色中出最大分牌
                for cl in colors_by_len:
                    cards = scored_by_color[cl]
                    return [max(cards, key=lambda c: c.score)]
            
            non_score_fudan = self._get_non_score_fudan()
            if non_score_fudan:
                # v9.11: 无分牌也按短门优先
                nonscore_by_color = {}
                for c in non_score_fudan:
                    cl = c.color
                    if cl not in nonscore_by_color:
                        nonscore_by_color[cl] = []
                    nonscore_by_color[cl].append(c)
                colors_by_len = sorted(nonscore_by_color.keys(),
                    key=lambda cl: self._get_color_card_count(cl))
                return [nonscore_by_color[colors_by_len[0]][0]]
            if zhudan:
                return [zhudan[0]]
        else:
            # 对手大→毙牌
            epoch_score = self._count_epoch_score(epoch_cards)
            should_trump = False
            # v9.11: 双方5分毙牌阈值（3分让队1偏强，保持对称）
            if self._is_last_trick() and self.score_koupai > 0:
                should_trump = True
            elif epoch_score >= 5:
                should_trump = True

            if should_trump and zhudan:
                # v9.2: 毙牌出牌选择——有分时用大主牌确保赢，小分时用小主牌省牌
                if epoch_score >= 10 and len(zhudan) >= 2:
                    # 高分轮→用较大主牌确保赢（避免被反毙丢分）
                    return [zhudan[-1]]  # 出最大主牌
                elif epoch_score >= 5:
                    # 中分轮→用中间主牌
                    mid = len(zhudan) // 2
                    return [zhudan[mid]]
                else:
                    # 低分轮→出最小主牌（省大牌）
                    return [zhudan[0]]

            # 不毙牌→出最小无分牌
            non_score_fudan = self._get_non_score_fudan()
            if non_score_fudan:
                return [non_score_fudan[0]]
            # v9.2: 无副牌可贴→出无分主牌（不送分）
            if zhudan:
                nonscore_zhu = [c for c in zhudan if not c.has_score]
                if nonscore_zhu:
                    return [nonscore_zhu[0]]
                return [zhudan[0]]

        return []

    def _follow_fudui(self, first_card, epoch_cards, epoch_players,
                      zhudui, fudui, zhudan, now_scores) -> list[Card]:
        """跟副对（v9.13）

        改进：
        - v9.13: 修复同花色判断——有同花色散牌但无对子时，也要凑2张同花色出，
          不能误判为绝门
        - 队友大→贴分对
        - 对手大→管不住时出无分对，不出分对送对手
        - 闲家≥5分就用大牌管（同步v8单牌阈值）
        """
        color = first_card.color
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)
        is_banker = self._is_banker_team()

        # v9.13: 检查是否有同花色（含散牌、对子、连对）
        has_color = self._has_color_cards(color)

        if color in fudui and len(fudui[color]) >= 2:
            if my_team_winning:
                # 队友大→贴分对
                score_pairs = []
                for i in range(0, len(fudui[color]), 2):
                    if i + 1 < len(fudui[color]) and fudui[color][i].name == fudui[color][i + 1].name:
                        if fudui[color][i].has_score:
                            score_pairs.append(fudui[color][i:i + 2])
                if score_pairs:
                    return score_pairs[0]
                return fudui[color][-2:]
            else:
                # 对手大→出能管住的最小对
                current_best = self._get_current_winning_card(epoch_cards)
                target = current_best if current_best else [first_card, first_card]
                # v9: ≥5分就用大牌管（与单牌一致）
                epoch_score = self._count_epoch_score(epoch_cards)
                should_play_big = epoch_score >= 5
                order = range(len(fudui[color]) - 1, -1, -2) if should_play_big else range(0, len(fudui[color]), 2)
                for i in order:
                    if i + 1 < len(fudui[color]) and fudui[color][i].name == fudui[color][i + 1].name:
                        if compare_outcards(fudui[color][i:i + 2], target,
                                            self.now_level, self.now_color):
                            return fudui[color][i:i + 2]
                # 管不住→v9: 出无分对，分对留着自己先手出
                non_score_pairs = []
                for i in range(0, len(fudui[color]), 2):
                    if i + 1 < len(fudui[color]) and fudui[color][i].name == fudui[color][i + 1].name:
                        if not fudui[color][i].has_score:
                            non_score_pairs.append(fudui[color][i:i + 2])
                if non_score_pairs:
                    return non_score_pairs[0]
                return fudui[color][:2]

        # v9.13: 有同花色但凑不到对子（散牌或单张+对子拆出，含主牌）
        if has_color:
            all_color = self.player.cards_in_hand.get(color, [])
            # v9.13: 不过滤主牌，引擎允许出同花色主牌跟牌
            if len(all_color) >= 2:
                # 优先出无分小牌
                nonscore = sorted([c for c in all_color if not c.has_score], key=lambda c: c.rank)
                if len(nonscore) >= 2:
                    return nonscore[:2]
                return sorted(all_color, key=lambda c: (c.has_score, c.rank))[:2]
            elif len(all_color) == 1:
                # 只有1张同花色：用其他副牌补齐，不够时用主牌
                other_fu = [c for cs in self.player.cards_in_hand.values() for c in cs
                           if c != all_color[0] and not c.is_zhu(self.now_level, self.now_color)]
                if other_fu:
                    return [all_color[0], min(other_fu, key=lambda c: (c.has_score, c.rank))]
                # 没有其他副牌，用主牌补
                zhu_cards = [c for c in zhudan if c not in all_color]
                if zhu_cards:
                    return [all_color[0], zhu_cards[0]]
                # 没有主牌，补最小副牌
                all_cards = []
                for cards in self.player.cards_in_hand.values():
                    all_cards.extend(cards)
                other = [c for c in all_cards if c != all_color[0]]
                if other:
                    return [all_color[0], min(other, key=lambda c: c.rank)]
                return [all_color[0]]

        # 绝门
        epoch_score = self._count_epoch_score(epoch_cards)

        if my_team_winning:
            # 贴其他花色分对
            for other_color in sorted(fudui.keys()):
                if other_color != color and len(fudui[other_color]) >= 2:
                    score_pairs = []
                    for i in range(0, len(fudui[other_color]), 2):
                        if i + 1 < len(fudui[other_color]):
                            if fudui[other_color][i].has_score:
                                score_pairs.append(fudui[other_color][i:i + 2])
                    if score_pairs:
                        return score_pairs[0]
            for other_color in sorted(fudui.keys()):
                if other_color != color and len(fudui[other_color]) >= 2:
                    return fudui[other_color][:2]
            # 贴最大的分牌给队友
            score_fudan = self._get_score_fudan()
            if len(score_fudan) >= 2:
                return sorted(score_fudan, key=lambda c: -c.score)[:2]
            if len(score_fudan) >= 1:
                non_score_fudan = self._get_non_score_fudan()
                if non_score_fudan:
                    return [score_fudan[-1] if score_fudan[-1].score >= score_fudan[0].score else score_fudan[0], non_score_fudan[0]]
            non_score_fudan = self._get_non_score_fudan()
            if len(non_score_fudan) >= 2:
                return non_score_fudan[:2]
        else:
            # 对手大→绝门时可灵活选择
            # 场上有分>10时用主对毙牌赢分，否则优先出无分副散牌保牌力
            should_trump = False
            if self._is_last_trick() and self.score_koupai > 0:
                should_trump = True
            elif epoch_score > 10:
                should_trump = True

            if should_trump and len(zhudui) >= 2:
                if epoch_score >= 15 and len(zhudui) >= 4:
                    return zhudui[-2:]  # 高分出大主对确保赢
                return zhudui[:2]  # 出最小主对

            # 不毙牌→优先出无分副散牌（保主牌力）
            a = self._get_analysis()
            non_score_fudan = self._get_non_score_fudan()
            if len(non_score_fudan) >= 2:
                return non_score_fudan[:2]
            if len(non_score_fudan) >= 1:
                for c in sorted(a['fudan'].keys()):
                    if c != color and a['fudan'][c]:
                        nonscore2 = [x for x in a['fudan'][c] if not x.has_score]
                        if nonscore2:
                            return [non_score_fudan[0], nonscore2[0]]
            # 没有足够无分副牌→出主牌（不给对手送分）
            if a['zhudan']:
                return [a['zhudan'][0]]
            all_cards = []
            for cards in self.player.cards_in_hand.values():
                all_cards.extend(cards)
            if len(all_cards) >= 2:
                return sorted(all_cards, key=lambda c: (c.has_score, c.rank))[:2]

        return []

    def _follow_fulian(self, first_cards, epoch_cards, epoch_players,
                       zhuliandui, fuliandui, now_scores) -> list[Card]:
        """跟副连对（v9.13）

        v9.13: 修复同花色判断——有同花色但无足够连对时，也要凑同花色出
        """
        color = first_cards[0].color
        n = len(first_cards)
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)
        is_banker = self._is_banker_team()

        # v9.13: 检查是否有同花色
        has_color = self._has_color_cards(color)

        if color in fuliandui:
            for chain in fuliandui[color]:
                if len(chain) == n:
                    if my_team_winning:
                        return chain
                    else:
                        current_best = self._get_current_winning_card(epoch_cards)
                        target = current_best if current_best else first_cards
                        if compare_outcards(chain, target,
                                            self.now_level, self.now_color):
                            return chain
            for chain in fuliandui[color]:
                if len(chain) >= n:
                    return chain[:n]

        # v9.13: 有同花色但无足够连对，凑同花色散牌/对子出
        if has_color:
            all_color = self.player.cards_in_hand.get(color, [])
            # v9.13: 不过滤主牌，引擎允许出同花色主牌跟牌
            if len(all_color) >= n:
                # 凑n张同花色牌（优先出无分小牌）
                nonscore = sorted([c for c in all_color if not c.has_score], key=lambda c: c.rank)
                result = nonscore[:n]
                if len(result) < n:
                    scored = sorted([c for c in all_color if c.has_score], key=lambda c: c.rank)
                    result += scored[:n - len(result)]
                if len(result) == n:
                    return result

        # 绝门
        if my_team_winning:
            # 队友大→优先贴分牌
            score_fudan = self._get_score_fudan()
            if len(score_fudan) >= n:
                return sorted(score_fudan, key=lambda c: -c.score)[:n]
            # 贴副连对
            for other_color in sorted(fuliandui.keys()):
                for chain in fuliandui[other_color]:
                    if len(chain) >= n:
                        return chain[:n]
            # 贴副对+分牌散牌组合
            a = self._get_analysis()
            for other_color in sorted(a['fudui'].keys()):
                if len(a['fudui'][other_color]) >= 2:
                    pair = a['fudui'][other_color][:2]
                    if n <= 2:
                        return pair
                    if score_fudan:
                        need = n - 2
                        extra = sorted(score_fudan, key=lambda c: -c.score)[:need]
                        if len(extra) == need:
                            return pair + extra
            # 贴分牌散牌
            if score_fudan:
                result = sorted(score_fudan, key=lambda c: -c.score)[:n]
                if len(result) < n:
                    non_score = self._get_non_score_fudan()
                    result += non_score[:n - len(result)]
                if len(result) == n:
                    return result
        else:
            # 对手大→绝门时不主动用主连对毙牌（保牌力），优先出无分副散牌
            # 只有没其他牌可出时才拆主连对
            non_score_fudan = self._get_non_score_fudan()
            if len(non_score_fudan) >= n:
                return non_score_fudan[:n]

            # 用副对凑
            for other_color in sorted(a.get('fudui', {}).keys() if 'a' in dir() else []):
                pass  # a可能未定义，下面重新获取
            a2 = self._get_analysis()
            for other_color in sorted(a2['fudui'].keys()):
                if len(a2['fudui'][other_color]) >= 2:
                    pair = a2['fudui'][other_color][:2]
                    if n <= 2:
                        return pair
                    need = n - 2
                    extra = non_score_fudan[:need]
                    if len(extra) == need:
                        return pair + extra

            # 副散牌不够凑齐→拆主连对（必须出牌）
            all_cards = []
            for cards in self.player.cards_in_hand.values():
                all_cards.extend(cards)
            fu_all = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)]
            if len(fu_all) >= n:
                sorted_fu = sorted(fu_all, key=lambda c: (c.has_score, c.rank))
                return sorted_fu[:n]
            # 副牌不够，拆主连对凑齐
            zhu_all = [c for c in all_cards if c.is_zhu(self.now_level, self.now_color)]
            combined = sorted(fu_all, key=lambda c: (c.has_score, c.rank)) + sorted(zhu_all, key=lambda c: c.rank)
            return combined[:min(n, len(combined))]

        return []

    def _follow_zhudan(self, first_card, epoch_cards, epoch_players,
                       zhudan, now_scores) -> list[Card]:
        """跟主单（v9.2）

        v9.2改进：对手大且管不住时，优先出无分主牌（不送分！）
        """
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)

        if zhudan:
            if my_team_winning:
                # v9.10: 队友大→贴最大分主牌（K/10=10分）
                score_zhudan = [c for c in zhudan if c.has_score]
                if score_zhudan:
                    return [max(score_zhudan, key=lambda c: c.score)]
                return [zhudan[0]]
            else:
                # 对手大→尝试管住
                current_best = self._get_current_winning_card(epoch_cards)
                # v9.7: 高分轮(≥10)双方都用更大主牌确保赢轮
                epoch_score = self._count_epoch_score(epoch_cards)
                should_play_big = epoch_score >= 10
                search_order = reversed(zhudan) if should_play_big else zhudan
                if current_best:
                    for card in search_order:
                        if compare_outcards([card], current_best,
                                            self.now_level, self.now_color):
                            return [card]
                else:
                    for card in search_order:
                        if compare_outcards([card], [first_card],
                                            self.now_level, self.now_color):
                            return [card]
                # v9.2: 管不住→优先出无分主牌，绝不送分！
                nonscore_zhudan = [c for c in zhudan if not c.has_score]
                if nonscore_zhudan:
                    return [nonscore_zhudan[0]]
                # 只剩分主牌没办法
                return [zhudan[0]]

        # 无主牌时出最小副牌
        all_cards = []
        for cards in self.player.cards_in_hand.values():
            all_cards.extend(cards)
        if all_cards:
            return [min(all_cards, key=lambda c: c.rank)]
        return []

    def _follow_zhudui(self, first_cards, epoch_cards, epoch_players,
                       zhudui, zhudan, now_scores) -> list[Card]:
        """跟主对（v9.2）

        v9.2: 管不住时优先出无分主对
        """
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)

        if len(zhudui) >= 2:
            if my_team_winning:
                # v9.10: 队友大→贴最大分主对
                score_pairs = []
                for i in range(0, len(zhudui), 2):
                    if i + 1 < len(zhudui) and zhudui[i].name == zhudui[i + 1].name:
                        if zhudui[i].has_score:
                            score_pairs.append((zhudui[i:i + 2], zhudui[i].score))
                if score_pairs:
                    score_pairs.sort(key=lambda x: -x[1])
                    return score_pairs[0][0]  # 贴最大分的对子
                return zhudui[:2]
            else:
                # 对手大→尝试管住
                current_best = self._get_current_winning_card(epoch_cards)
                target = current_best if current_best else first_cards
                for i in range(0, len(zhudui), 2):
                    if i + 1 < len(zhudui) and zhudui[i].name == zhudui[i + 1].name:
                        if compare_outcards(zhudui[i:i + 2], target,
                                            self.now_level, self.now_color):
                            return zhudui[i:i + 2]
                # v9.2: 管不住→优先出无分主对
                for i in range(0, len(zhudui), 2):
                    if i + 1 < len(zhudui) and zhudui[i].name == zhudui[i + 1].name:
                        if not zhudui[i].has_score:
                            return zhudui[i:i + 2]
                return zhudui[:2]

        # 没有主对时，用主散牌凑对或垫牌
        if len(zhudan) >= 2:
            # 无主对必须出最大的主牌
            return self._top_zhu_cards(zhudan, 2)
        if len(zhudan) == 1:
            # 只有一张主散牌，必须先出这张主牌+最小副牌垫
            all_cards = []
            for cards in self.player.cards_in_hand.values():
                all_cards.extend(cards)
            fu_cards = [c for c in all_cards if not c.is_zhu(self.now_level, self.now_color) and c != zhudan[0]]
            if fu_cards:
                return [zhudan[0], min(fu_cards, key=lambda c: c.rank)]
            return [zhudan[0]]
        # 完全没有主牌，用最小副牌垫
        all_cards = []
        for cards in self.player.cards_in_hand.values():
            all_cards.extend(cards)
        if len(all_cards) >= 2:
            sorted_cards = sorted(all_cards, key=lambda c: c.rank)
            return sorted_cards[:2]
        if all_cards:
            return [all_cards[0]]

        return []

    def _top_zhu_cards(self, cards: list[Card], count: int) -> list[Card]:
        return sorted(
            cards,
            key=lambda c: (get_zhu_rank(c, self.now_level, self.now_color), c.rank),
            reverse=True,
        )[:count]

    def _follow_zhulian(self, first_cards, epoch_cards, epoch_players,
                        zhuliandui, now_scores) -> list[Card]:
        """跟主连对（v9）"""
        n = len(first_cards)
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)

        if my_team_winning:
            for chain in zhuliandui:
                if len(chain) >= n:
                    return chain[:n]
        else:
            current_best = self._get_current_winning_card(epoch_cards)
            target = current_best if current_best else first_cards
            for chain in zhuliandui:
                if len(chain) >= n:
                    if compare_outcards(chain[:n], target,
                                        self.now_level, self.now_color):
                        return chain[:n]
            for chain in zhuliandui:
                if len(chain) >= n:
                    return chain[:n]

        # 无足够长主连对：用主对/主散牌凑齐
        all_cards = []
        for cards in self.player.cards_in_hand.values():
            all_cards.extend(cards)
        zhu_cards = sorted([c for c in all_cards if c.is_zhu(self.now_level, self.now_color)],
                          key=lambda c: (get_zhu_rank(c, self.now_level, self.now_color), c.rank),
                          reverse=True)
        fu_cards = sorted([c for c in all_cards if not c.is_zhu(self.now_level, self.now_color)],
                         key=lambda c: c.rank)

        # 有主牌时必须出所有主牌，不够n张用副牌补齐
        if zhu_cards:
            result = zhu_cards[:min(len(zhu_cards), n)]
            if len(result) < n:
                # 主牌不够，副牌补齐
                result += fu_cards[:n - len(result)]
            return result[:n]
        # 完全无主牌(0张)：用副牌凑
        if fu_cards:
            return fu_cards[:n]
        return []
