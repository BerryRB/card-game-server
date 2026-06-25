# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - AI出牌策略（v10真人经验版）

v10核心改进——基于真人升级出牌策略经验，全面重构：

真人核心经验：
1. 【得分核心理念】先出确定能控场的副牌（副大单→副大对→副对试探→队友绝门→主牌）
2. 【5分牌不无脑贴】5有赢轮潜力时留着自己赢轮跑掉，不是逢贴必出
3. 【保底策略】底牌有分+有能力保底+后期→保留主牌，优先出副牌
4. 【队友绝门利用】出队友绝门对手未绝门的花色→队友毙牌赢轮
5. 【副对谨慎】对10/对K被毙丢20分，游戏前期不出危险分对
6. 【毙牌优化】对手没绝门→亮主花色分牌/小牌毙；对手也绝门→大主牌确保赢
7. 【主牌策略分化】前期出主大单让队友贴分，后期留主大单控场压牌
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
    """AI出牌策略（v10真人经验版）"""

    def __init__(self, player, now_level: str, now_color: Optional[str] = None,
                 score_koupai: int = 0):
        self.player = player
        self.now_level = now_level
        self.now_color = now_color
        self.score_koupai = score_koupai
        self._cache = None
        # v10.1: 从Player对象恢复追踪状态（跨decide_play持久化）
        tracking = getattr(player, 'ai_tracking', None)
        if tracking is not None:
            self._played_tricks = tracking.get('played_tricks', 0)
            self._mate_void_colors = tracking.get('mate_void_colors', set()).copy()
            self._opponent_void_colors = tracking.get('opponent_void_colors', set()).copy()
        else:
            self._played_tricks = 0
            self._mate_void_colors = set()
            self._opponent_void_colors = set()

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

    # ========== v10新增：局势感知辅助方法 ==========

    def _update_trick_state(self, epoch_cards, epoch_players, is_first):
        """v10: 每次decide_play时更新跨轮追踪状态
        
        通过当前轮出牌信息推算绝门信息：
        - 首出副牌时，跟牌出主牌→该花色绝门
        - 非首出时，不更新（信息不完整）
        """
        if not is_first and epoch_cards and len(epoch_cards) >= 2:
            first_cards = epoch_cards[0]
            if first_cards:
                first_type = determine_play_type(first_cards, self.now_level, self.now_color)
                # 只在首出副牌时判断绝门（主牌轮无法判断）
                if first_type and first_type.startswith('fu'):
                    first_color = first_cards[0].color
                    for i, cards in enumerate(epoch_cards[1:], 1):
                        if not cards:
                            continue
                        player_id = epoch_players[i].player_id if i < len(epoch_players) else -1
                        is_mate = (player_id % 2) == (self.player.player_id % 2)
                        # 判断该玩家是否绝门：出了主牌或不同花色副牌
                        played_fu_same_color = any(
                            c.color == first_color and not self._is_zhu(c) for c in cards)
                        if not played_fu_same_color:
                            if is_mate:
                                self._mate_void_colors.add(first_color)
                            else:
                                self._opponent_void_colors.add(first_color)

    def _is_certain_max_fudan(self, card: Card) -> bool:
        """v10.1: 判断一张副单是否确定是（或大概率是）该花色最大牌
        
        真人经验：确定能控场的副牌优先出，让队友贴分。
        扩展判断：
        1. A（非主牌level时）一定是最大
        2. K在中后期（6+轮）大概率最大
        3. Q在后期（10+轮）大概率最大
        4. 该花色fudan中rank最大的牌→当前最大
        """
        if self._is_zhu(card):
            return False
        # A一定是最大副牌（但A是level时变成了主牌，不会到这里）
        if card.name == 'A':
            return True
        # K在游戏中后期（6+轮）大概率是最大
        if card.name == 'K' and self._played_tricks >= 6:
            return True
        # Q在后期（10+轮）大概率最大
        if card.name == 'Q' and self._played_tricks >= 10:
            return True
        # 该花色fudan中rank最大的牌且rank>=Q(11)→大概率最大
        # 不算J及以下的——J太容易被Q/K/A压
        a = self._get_analysis()
        color_fudan = a['fudan'].get(card.color, [])
        if color_fudan:
            max_rank = max(c.rank for c in color_fudan)
            if card.rank == max_rank and card.rank >= 11:  # Q及以上
                return True
        return False

    def _is_likely_max_fudan(self, card: Card) -> bool:
        """v12: 判断一张副单是否大概率是该花色最大牌（放宽版）

        比_is_certain_max_fudan更宽松：
        - J以上且是该花色最大→大概率最大
        - A一定最大，K前期也大概率最大
        """
        if self._is_zhu(card):
            return False
        # A一定是最大副牌
        if card.name == 'A':
            return True
        # K大概率最大（不需要等6轮）
        if card.name == 'K':
            return True
        # Q大概率最大
        if card.name == 'Q':
            return True
        # J在前期(≤3轮)大概率最大
        if card.name == 'J' and self._played_tricks <= 3:
            return True
        # 该花色fudan中rank最大的牌且rank>=J(11)→大概率最大
        a = self._get_analysis()
        color_fudan = a['fudan'].get(card.color, [])
        if color_fudan:
            max_rank = max(c.rank for c in color_fudan)
            if card.rank == max_rank and card.rank >= 11:  # J及以上
                return True
        return False

    def _is_certain_max_fudui(self, cards: list[Card]) -> bool:
        """v10: 判断一副对是否确定是最大副对
        
        A对一定最大；K对在后期大概率最大。
        """
        if not cards or len(cards) < 2:
            return False
        if any(self._is_zhu(c) for c in cards):
            return False
        top_rank = max(c.rank for c in cards)
        if top_rank >= 13:  # A对
            return True
        if top_rank >= 12 and self._played_tricks >= 8:  # K对+后期
            return True
        return False

    def _is_game_midlate(self) -> bool:
        """v10: 判断是否游戏中后期（剩余手牌≤13张，约第7轮起）"""
        total = sum(len(cards) for cards in self.player.cards_in_hand.values())
        return total <= 13

    def _can_protect_koupai(self) -> bool:
        """v10: 判断是否具备保底能力
        
        真人经验：底牌有分时，需要保留主对保底。
        条件：主对数≥2 + 有主大单确保倒数第二轮能赢。
        """
        a = self._get_analysis()
        zhu_dui_count = len(a['zhudui']) // 2  # 主对数量
        has_big_zhu = len(a['zhudan']) >= 2  # 有大主单
        return zhu_dui_count >= 2 and has_big_zhu

    def _should_conserve_zhu_for_koupai(self) -> bool:
        """v12: 是否需要为主牌保底而保留主牌
        
        庄家：底牌有分+具备保底能力+游戏中后期→保留主牌
        闲家：不需要保底（闲家不扣底），始终返回False
        
        v12改进：放宽触发时机——有底分+有能力保底+剩余≤20张（约第5轮起）
        前期不需要保底（牌多时随便打），中后期必须保留主牌
        """
        if not self.player.is_banker:
            return False  # 闲家不保底
        if self.score_koupai <= 0:
            return False  # 底牌没分，不需要保底
        if not self._can_protect_koupai():
            return False  # 没能力保底，正常打
        return self._is_game_midlate()  # 后期才考虑保底

    def _get_mate_jue_for_exploit(self) -> list[str]:
        """v10: 获取队友绝门但对手未绝门的花色列表
        
        真人经验：出队友绝门对手未绝门的牌→队友可以毙牌赢轮。
        """
        result = []
        for color in self._mate_void_colors:
            if color not in self._opponent_void_colors:
                result.append(color)
        return result

    def _is_dangerous_score_pair(self, cards: list[Card]) -> bool:
        """v10: 判断副对是否是危险分对（对10/对K，出时被毙丢20分）
        
        真人经验：对10这种要谨慎判断，被对手毙了一下丢20分。
        """
        if not cards:
            return False
        total_score = sum(c.score for c in cards)
        return total_score >= 15  # 对10=20分，对K=20分，对5=10分

    def _5_has_winning_potential(self, card: Card) -> bool:
        """v10.1: 判断5分牌是否有自己赢轮的潜力
        
        真人经验：5很多时候具备赢轮的能力，可以自己赢轮时顺便跑掉，
        不要无脑贴给队友。
        
        v10.1修正：副5也有潜力——如果该花色5是手中该花色fudan最大的，
        或者该花色只剩1-2张fudan时5有希望领打。
        """
        if card.name != '5':
            return False
        color = card.color
        # 该花色5是否是主牌？主牌5有赢轮能力
        if self._is_zhu(card):
            return True
        # v10.1: 副5的潜力判断
        a = self._get_analysis()
        color_fudan = a['fudan'].get(color, [])
        if not color_fudan:
            return False
        # 如果该花色副单只有1-2张，5大概率要领打→有潜力
        if len(color_fudan) <= 2:
            return True
        # 如果5是该花色fudan中rank最高的（只剩5和小牌），有潜力
        max_rank = max(c.rank for c in color_fudan)
        if card.rank >= max_rank - 3:  # 5接近最大时有潜力（放宽：-3而非-2）
            return True
        return False

    def _best_trump_card_for_kill(self, epoch_score: int) -> Optional[Card]:
        """v10: 选择最优毙牌主牌
        
        真人经验：
        - 对手都没绝门→优先用亮主花色分牌毙牌（10/K级牌慎用），没分用亮主花色小牌
        - 自己和对手都绝门→场上有分用大主牌毙，无分用防10/K跑的主牌
        """
        a = self._get_analysis()
        zhudan = a['zhudan']
        if not zhudan:
            return None

        # 检查对手是否也绝门
        opponent_also_void = len(self._opponent_void_colors) > 0

        if opponent_also_void and epoch_score >= 5:
            # 对手也绝门+场上有分→用大主牌确保赢（级牌/大小王衡量后使用）
            # 优先用级牌或主花色大牌
            for c in reversed(zhudan):
                if c.rank >= 10:  # Q级以上的主牌
                    return c
            return zhudan[-1]  # 没有大牌就用最大

        # 对手没绝门（或场上无分）→优先用亮主花色分牌/小牌
        # 找亮主花色的牌
        liang_color_zhu = [c for c in zhudan if c.color == self.now_color]
        if liang_color_zhu:
            # 优先出亮主花色的分牌
            score_liang = [c for c in liang_color_zhu if c.has_score]
            if score_liang:
                # 10和K是级牌时慎用（级牌太大，留着控场更好）
                level_name = str(self.now_level)
                safe_score = [c for c in score_liang if c.name != level_name]
                if safe_score:
                    return safe_score[0]  # 出亮主花色5分牌
                # 只有级牌分牌，场上高分时才用
                if epoch_score >= 10:
                    return score_liang[0]
            # 没分牌→出亮主花色小牌（保留大牌控场）
            return liang_color_zhu[0]

        # 没有亮主花色主牌→出最小主牌
        return zhudan[0]

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
        # v10: 更新跨轮追踪状态
        self._update_trick_state(epoch_cards, epoch_players, is_first)
        # v10: 首出时递增轮数计数器
        if is_first:
            self._played_tricks += 1

        a = self._get_analysis()
        zhudan = a['zhudan']
        zhudui = a['zhudui']
        zhuliandui = a['zhuliandui']
        fudan = a['fudan']
        fudui = a['fudui']
        fuliandui = a['fuliandui']

        if is_first:
            result = self._first_play(zhudan, zhudui, zhuliandui,
                                    fudan, fudui, fuliandui, now_scores)
        else:
            result = self._follow_play(epoch_cards, epoch_players,
                                     zhudan, zhudui, zhuliandui,
                                     fudan, fudui, fuliandui, now_scores)
        
        # v9.14兜底：AI返回空列表但手中还有牌时，出第一张
        if not result and self.player.card_count > 0:
            for cards in self.player.cards_in_hand.values():
                if cards:
                    result = [cards[0]]
                    break
        
        # 防御性检查：去重（同一Card对象不能出两次）
        if result:
            seen_ids = set()
            deduped = []
            for c in result:
                if id(c) not in seen_ids:
                    seen_ids.add(id(c))
                    deduped.append(c)
            if len(deduped) < len(result):
                # 有重复→用手中其他牌补足
                used_ids = set(id(c) for c in deduped)
                for cards in self.player.cards_in_hand.values():
                    for c in cards:
                        if id(c) not in used_ids and len(deduped) < len(result):
                            deduped.append(c)
                            used_ids.add(id(c))
                result = deduped
        
        # v10.1: 将追踪状态写回Player对象（跨decide_play持久化）
        tracking = getattr(self.player, 'ai_tracking', None)
        if tracking is not None:
            tracking['played_tricks'] = self._played_tricks
            tracking['mate_void_colors'] = self._mate_void_colors
            tracking['opponent_void_colors'] = self._opponent_void_colors
        
        return result

    # ========== 先手出牌策略 ==========

    def _first_play(self, zhudan, zhudui, zhuliandui,
                    fudan, fudui, fuliandui, now_scores) -> list[Card]:
        """v12领打策略——副牌优先+庄闲差异化

        v12核心改进（基于500局数据分析）：
        - 前3轮首出主牌率68%→目标40%：放宽副单出牌条件，先清副牌再出主牌
        - 庄家首出主牌门槛提高：只有确定赢的主大牌才首出
        - 分牌保护：5/K在不确定赢时不出

        优先级：
        1. 确定最大副单→让队友贴分（放宽：J以上也算大概率最大）
        2. 副大牌试探出（rank≥9，不需要确定赢，先出再说）
        3. 确定最大副对→让队友贴分
        4. 副对试探出
        5. 队友绝门利用
        6. 剩余副小单/副小对（v12新增：在出主牌前先清副牌）
        7. 主牌策略（庄家：只出确定赢的大主牌；闲家：自由出主牌）
        8. 兜底
        """
        # 尾局策略：赢最后一轮=控制扣底（庄家更重要）
        if self._is_endgame():
            for chain in zhuliandui:
                return chain
            if len(zhudui) >= 2:
                return zhudui[-2:]
            if zhudan:
                return [zhudan[-1]]

        # ===== 1. 确定最大副单——让队友贴分（v12放宽条件）=====
        for color in sorted(fudan.keys()):
            for card in reversed(fudan[color]):  # 从大到小找
                if self._is_likely_max_fudan(card):
                    # v12: 确定大概率最大的副单就出，不要求无分
                    # 有分的K/A更要出（让队友贴分的同时跑自己的分）
                    if not card.has_score:
                        return [card]  # 无分确定最大副单，最安全
                    # 有分但确定最大也出（A/K的分在确定赢时跑掉最安全）
                    if card.rank >= 12:  # K及以上确定赢
                        return [card]

        # ===== 2. 副大牌试探出（v12新增）=====
        # 真人经验：不确定赢的副大牌也要先出，比出主牌好
        # rank≥9(9/J/Q)的副单先出试探，让队友决定是否贴分
        for color in sorted(fudan.keys()):
            for card in reversed(fudan[color]):  # 从大到小
                if card.rank >= 9 and not card.has_score:
                    return [card]  # 出无分的中大副单试探
        # 有分的中大副单也出（rank≥J，有5分的9/10不出）
        for color in sorted(fudan.keys()):
            for card in reversed(fudan[color]):
                if card.rank >= 11 and card.has_score:
                    # J的5分不出（太小），10的10分在前期谨慎
                    if card.score <= 5 and self._played_tricks <= 3:
                        continue
                    return [card]

        # ===== 3. 确定最大副对——让队友贴分 =====
        for color in sorted(fudui.keys()):
            if len(fudui[color]) >= 2:
                for i in range(len(fudui[color]) - 1, 0, -2):
                    if i >= 1 and fudui[color][i].name == fudui[color][i-1].name:
                        pair = fudui[color][i-1:i+1]
                        if self._is_certain_max_fudui(pair):
                            if not self._is_dangerous_score_pair(pair):
                                return pair

        # ===== 4. 副对试探出 =====
        for color in sorted(fudui.keys()):
            if len(fudui[color]) >= 2:
                for i in range(len(fudui[color]) - 1, 0, -2):
                    if i >= 1 and fudui[color][i].name == fudui[color][i-1].name:
                        pair = fudui[color][i-1:i+1]
                        if not self._is_dangerous_score_pair(pair):
                            return pair
                if self._is_game_midlate():
                    for i in range(0, len(fudui[color]), 2):
                        if i + 1 < len(fudui[color]) and fudui[color][i].name == fudui[color][i+1].name:
                            return fudui[color][i:i+2]

        # ===== 5. 队友绝门对手未绝门的副单——让队友毙牌赢轮 =====
        exploit_colors = self._get_mate_jue_for_exploit()
        if exploit_colors:
            for color in exploit_colors:
                if color in fudan and fudan[color]:
                    nonscore = [c for c in fudan[color] if not c.has_score]
                    if nonscore:
                        return [nonscore[0]]
                    if fudan[color]:
                        return [fudan[color][0]]

        # ===== 6. 副牌清场（v12新增：在出主牌前先清剩余副牌）=====
        # 真人经验：手里有副牌就先出副牌，不要急着出主牌
        # 6a. 清短门副小单
        weak_colors = []
        for color in sorted(fudan.keys()):
            card_count = self._get_color_card_count(color)
            fu_dan_count = len(fudan.get(color, []))
            if fu_dan_count > 0:
                has_nonscore = any(not c.has_score for c in fudan[color])
                weak_colors.append((color, card_count, fu_dan_count, has_nonscore))
        if weak_colors:
            # 短门优先出（创造绝门机会）
            weak_colors.sort(key=lambda x: (not x[3], x[1]))
            for color, _, _, has_nonscore in weak_colors:
                if has_nonscore:
                    nonscore = [c for c in fudan[color] if not c.has_score]
                    if nonscore:
                        return [nonscore[0]]
                # v12: 分牌也出（5分可出，10/K留到后面处理）
                score_cards = [c for c in fudan[color] if c.has_score]
                if score_cards:
                    safe_scores = [c for c in score_cards if c.score <= 5]
                    if safe_scores:
                        return [safe_scores[0]]

        # 6b. 出无对子花色的小单
        for color in sorted(fudan.keys()):
            if fudan[color] and not self._has_pair_in_color(color):
                nonscore = [c for c in fudan[color] if not c.has_score]
                if nonscore:
                    return [nonscore[0]]
                return [fudan[color][0]]

        # 6c. 副小单兜底
        for color in sorted(fudan.keys()):
            if fudan[color]:
                nonscore = [c for c in fudan[color] if not c.has_score]
                if nonscore:
                    return [nonscore[0]]
                return [fudan[color][0]]

        # 6d. 副连对
        for color in sorted(fuliandui.keys()):
            if fuliandui[color]:
                return fuliandui[color][0]

        # ===== 7. 主牌策略（v12：庄家门槛提高）=====
        conserve_zhu = self._should_conserve_zhu_for_koupai()

        if conserve_zhu:
            # 保底策略：有底分+有能力保底+后期→保留主牌
            if len(zhudui) >= 2:
                return zhudui[:2]
            if zhudan:
                return [zhudan[0]]
        else:
            if self.player.is_banker:
                # v12庄家：只有确定能赢的主牌才首出
                # 大主单（top 30%的主单）或主大对
                for chain in zhuliandui:
                    return chain
                if len(zhudui) >= 2:
                    return zhudui[-2:]  # 最大主对
                # v12: 庄家只出大主单（rank在主单中排名前30%）
                if zhudan:
                    big_threshold = max(1, len(zhudan) * 7 // 10)  # 前30%的rank门槛
                    for card in reversed(zhudan):  # 从大到小
                        if card.rank >= zhudan[big_threshold - 1].rank:
                            return [card]
                    # 没有大主单→出最小主单（但这种情况说明主牌很弱，应该先清了副牌）
                    return [zhudan[0]]
            else:
                # 闲家：自由出主牌（和v11一样）
                for chain in zhuliandui:
                    return chain
                if len(zhudui) >= 2:
                    return zhudui[-2:]
                if zhudan:
                    return [zhudan[0]]

        # ===== 8. 主单兜底 =====
        if zhudan:
            return [zhudan[0]]

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
        """检查是否有指定花色的副牌（排除该花色的主牌）

        引擎_validate_follow的逻辑：
        - 首出副牌时，引擎按hand_color_fu（同花色非主牌）数量决定must_play_color
        - 如果同花色只有主牌（级牌/2/3/5），hand_color_fu=0，must_play_color=0
          → 引擎视为绝门，不强制出同花色
        - 所以AI的"有同花色"判断也应只看副牌，否则会误入"有同花色"分支
          虽然后续all_color_fu=0会fallback到绝门，但浪费判断且可能导致策略次优
        
        v10.1: 修复误判——只统计同花色非主牌数量
        """
        cards = self.player.cards_in_hand.get(color, [])
        fu_count = sum(1 for c in cards if not c.is_zhu(self.now_level, self.now_color))
        return fu_count > 0

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
        """跟副单（v10真人经验版）

        v10核心改动（基于真人经验）：
        1. 队友赢+贴分：5分牌不无脑贴（5有赢轮潜力时留着自己赢轮跑掉）
        2. 不确定队友赢：有能控场的副牌就出，防止对手控场
        3. 绝门毙牌：用_best_trump_card_for_kill选最优主牌
        4. 队友赢+绝门：5分主牌也酌情保留（不是无脑贴）
        """
        color = first_card.color
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)
        epoch_score = self._count_epoch_score(epoch_cards)

        # v9.13: 用_has_color_cards判断是否有同花色（含对子/连对/主牌）
        a = self._get_analysis()
        has_color = self._has_color_cards(color)
        color_singles = self._get_color_single_cards(color, fudan, a['fudui'], a['fuliandui'], zhudan)

        if has_color and color_singles:
            # ===== 有同花色 =====
            if my_team_winning:
                # v11: 队友赢→庄闲差异化贴分
                if self.player.is_banker:
                    # 庄家：队友赢→不贴分牌（庄家不靠得分赢，保留分牌）
                    # 出最小无分牌
                    non_score = [c for c in color_singles if not c.has_score]
                    if non_score:
                        return [non_score[0]]
                    # 只有分牌→出最小（保留大牌）
                    return [color_singles[0]]
                else:
                    # 闲家：队友赢→积极贴分牌（得分过80=闲家赢）
                    score_cards = [c for c in color_singles if c.has_score]
                    if score_cards:
                        # 5分牌有赢轮潜力就留着
                        safe_score = [c for c in score_cards if not self._5_has_winning_potential(c)]
                        if safe_score:
                            return [safe_score[0]]
                        return [score_cards[0]]
                    # 没分牌出最小
                    return [color_singles[0]]
            else:
                # ===== 对手赢→尝试管住 =====
                current_best = self._get_current_winning_card(epoch_cards)

                # v10: 不确定队友是否赢轮（自己不是最后出牌）时，如果有能管住的牌就管
                if current_best:
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
                # 管不住→庄家保护5/K，闲家出最小牌
                non_score = [c for c in color_singles if not c.has_score]
                if non_score:
                    return [non_score[0]]
                # v12: 庄家管不住+只有分牌→5/K不出（除非只剩1张）
                if self.player.is_banker:
                    # 优先出5（5分vs K的10分，损失更小）
                    five_cards = [c for c in color_singles if c.score == 5]
                    if five_cards:
                        return [five_cards[0]]
                    # 只有10/K→无奈出最小
                    return [color_singles[0]]
                # 闲家：5分牌有赢轮潜力则保留
                safe = [c for c in color_singles if not self._5_has_winning_potential(c)]
                if safe:
                    return [safe[0]]
                return [color_singles[0]]

        # ===== 绝门 =====
        if my_team_winning:
            # v11: 队友赢+绝门→庄闲差异化
            if self.player.is_banker:
                # 庄家：队友赢+绝门→不贴分牌（保留分牌，庄家不靠得分赢）
                non_score_fudan = self._get_non_score_fudan()
                if non_score_fudan:
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
                    nonscore_zhu = [c for c in zhudan if not c.has_score]
                    if nonscore_zhu:
                        return [nonscore_zhu[0]]
                    return [zhudan[0]]
            else:
                # 闲家：队友赢+绝门→积极贴分牌（得分过80=闲家赢）
                if self._is_last_trick() and self.score_koupai > 0:
                    non_score_fudan = self._get_non_score_fudan()
                    if non_score_fudan:
                        return [non_score_fudan[0]]
                    for c in sorted(fudan.keys()):
                        if fudan[c]:
                            return [fudan[c][0]]

                # 优先贴短门花色的分牌（创造绝门）
                score_fudan = self._get_score_fudan()
                if score_fudan:
                    scored_by_color = {}
                    for c in score_fudan:
                        cl = c.color
                        if cl not in scored_by_color:
                            scored_by_color[cl] = []
                        scored_by_color[cl].append(c)
                    safe_by_color = {}
                    for cl, cards in scored_by_color.items():
                        safe = [c for c in cards if not self._5_has_winning_potential(c)]
                        if safe:
                            safe_by_color[cl] = safe
                    if safe_by_color:
                        colors_by_len = sorted(safe_by_color.keys(),
                            key=lambda cl: self._get_color_card_count(cl))
                        return [max(safe_by_color[colors_by_len[0]], key=lambda c: c.score)]

                # 无可安全贴的分牌→出短门无分牌
                non_score_fudan = self._get_non_score_fudan()
                if non_score_fudan:
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
                    nonscore_zhu = [c for c in zhudan if not c.has_score]
                    if nonscore_zhu:
                        return [nonscore_zhu[0]]
                    return [zhudan[0]]
        else:
            # ===== 对手赢+绝门→毙牌（≥3分门槛）=====
            should_trump = False
            if self._is_last_trick() and self.score_koupai > 0:
                should_trump = True
            elif epoch_score >= 5:
                should_trump = True
            elif epoch_score >= 3 and zhudan:
                # 闲家：≥3分+有主牌→毙
                should_trump = True

            if should_trump and zhudan:
                # v10: 用最优毙牌主牌选择
                best_kill = self._best_trump_card_for_kill(epoch_score)
                if best_kill:
                    return [best_kill]
                # 兜底
                return [zhudan[0]]

            # 不毙牌→出最小无分牌
            non_score_fudan = self._get_non_score_fudan()
            if non_score_fudan:
                return [non_score_fudan[0]]
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
                # v11: 队友大→庄闲差异化贴分对
                if self.player.is_banker:
                    # 庄家：不贴分对（保留分牌）
                    non_score_pairs = []
                    for i in range(0, len(fudui[color]), 2):
                        if i + 1 < len(fudui[color]) and fudui[color][i].name == fudui[color][i + 1].name:
                            if not fudui[color][i].has_score:
                                non_score_pairs.append(fudui[color][i:i + 2])
                    if non_score_pairs:
                        return non_score_pairs[0]
                    return fudui[color][-2:]
                else:
                    # 闲家：积极贴分对
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
                epoch_score = self._count_epoch_score(epoch_cards)
                should_play_big = epoch_score >= 5
                order = range(len(fudui[color]) - 1, -1, -2) if should_play_big else range(0, len(fudui[color]), 2)
                for i in order:
                    if i + 1 < len(fudui[color]) and fudui[color][i].name == fudui[color][i + 1].name:
                        if compare_outcards(fudui[color][i:i + 2], target,
                                            self.now_level, self.now_color):
                            return fudui[color][i:i + 2]
                # 管不住→出无分对，分对留着自己先手出
                non_score_pairs = []
                for i in range(0, len(fudui[color]), 2):
                    if i + 1 < len(fudui[color]) and fudui[color][i].name == fudui[color][i + 1].name:
                        if not fudui[color][i].has_score:
                            non_score_pairs.append(fudui[color][i:i + 2])
                if non_score_pairs:
                    return non_score_pairs[0]
                # 只有分对→庄家出最小分对减少损失
                return fudui[color][:2]

        # v9.14: 有同花色但凑不到对子——必须检查同花色副牌是否有对子
        if has_color:
            all_color = self.player.cards_in_hand.get(color, [])
            all_color_fu = [c for c in all_color
                           if not c.is_zhu(self.now_level, self.now_color)]
            all_color_zhu = [c for c in all_color
                            if c.is_zhu(self.now_level, self.now_color)]
            
            # 检查同花色副牌是否有对子
            fu_rank_groups = {}
            for c in all_color_fu:
                fu_rank_groups.setdefault(c.rank, []).append(c)
            fu_has_pair = any(len(g) >= 2 for g in fu_rank_groups.values())
            
            if fu_has_pair:
                # 必须出对子！找最小无分对子
                for rank in sorted(fu_rank_groups.keys()):
                    g = fu_rank_groups[rank]
                    if len(g) >= 2:
                        pair = sorted(g[:2], key=lambda c: (c.has_score, c.rank))
                        return pair
            
            if len(all_color_fu) >= 2:
                # 有2+张同花色副牌但无对子，出最小2张
                return sorted(all_color_fu, key=lambda c: (c.has_score, c.rank))[:2]
            
            elif len(all_color_fu) == 1:
                # 1张同花色副牌 + 1张其他牌
                must_play_fu = all_color_fu[0]
                if all_color_zhu:
                    return [must_play_fu, all_color_zhu[0]]
                # 补最小其他副牌
                all_cards = []
                for cards in self.player.cards_in_hand.values():
                    all_cards.extend(cards)
                other = [c for c in all_cards if c != must_play_fu]
                if other:
                    return [must_play_fu, min(other, key=lambda c: c.rank)]
                return [must_play_fu]

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
            # v9.10: 贴最大的分牌给队友（K/10=10分 > 5=5分）
            score_fudan = self._get_score_fudan()
            if len(score_fudan) >= 2:
                return sorted(score_fudan, key=lambda c: -c.score)[:2]  # 贴最大的
            if len(score_fudan) >= 1:
                non_score_fudan = self._get_non_score_fudan()
                if non_score_fudan:
                    return [score_fudan[-1] if score_fudan[-1].score >= score_fudan[0].score else score_fudan[0], non_score_fudan[0]]
            non_score_fudan = self._get_non_score_fudan()
            if len(non_score_fudan) >= 2:
                return non_score_fudan[:2]
        else:
            # 对手赢+绝门→毙牌（≥3分门槛）
            epoch_score = self._count_epoch_score(epoch_cards)
            should_trump = False
            if self._is_last_trick() and self.score_koupai > 0:
                should_trump = True
            elif epoch_score >= 5:
                should_trump = True
            elif epoch_score >= 3 and len(zhudui) >= 2:
                should_trump = True

            if should_trump and len(zhudui) >= 2:
                # v10: 用最优毙牌策略
                if len(self._opponent_void_colors) > 0 and epoch_score >= 5:
                    return zhudui[-2:]  # 出最大主对
                return zhudui[:2]

            # 不毙牌→出最小无分牌
            a = self._get_analysis()
            non_score_fudan = self._get_non_score_fudan()
            if len(non_score_fudan) >= 2:
                return non_score_fudan[:2]
            if len(non_score_fudan) >= 1:
                first_card = non_score_fudan[0]
                first_id = id(first_card)
                for c in sorted(a['fudan'].keys()):
                    if c != color and a['fudan'][c]:
                        nonscore2 = [x for x in a['fudan'][c] if not x.has_score and id(x) != first_id]
                        if nonscore2:
                            return [first_card, nonscore2[0]]
            # v9.10f: 没有第二个无分副牌，出主牌（不给对手送分）
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

        # v9.13: 有同花色但无足够连对，凑同花色对子/散牌出
        # v9.14: 修复凑散牌逻辑——跟连对时不能出纯散牌，必须凑对子组合
        if has_color:
            all_color = self.player.cards_in_hand.get(color, [])
            all_color_fu = [c for c in all_color
                           if not c.is_zhu(self.now_level, self.now_color)]
            if len(all_color_fu) >= n:
                # 按rank分组找对子
                rank_groups: dict[int, list] = {}
                for c in all_color_fu:
                    if c.rank not in rank_groups:
                        rank_groups[c.rank] = []
                    rank_groups[c.rank].append(c)
                
                # 收集所有对子（每对2张同rank）
                pairs_available = []
                for rank in sorted(rank_groups.keys()):
                    group = rank_groups[rank]
                    while len(group) >= 2:
                        pairs_available.append([group[0], group[1]])
                        group = group[2:]
                
                # 用对子凑够n张
                result = []
                for pair in pairs_available:
                    result.extend(pair)
                    if len(result) >= n:
                        return result[:n]
                
                # 对子不够n张，用对子+单牌凑
                if result:
                    remaining = n - len(result)
                    used_ids = set(id(c) for c in result)
                    single_cards = sorted([c for c in all_color_fu if id(c) not in used_ids],
                                         key=lambda c: (c.has_score, c.rank))
                    result.extend(single_cards[:remaining])
                    if len(result) == n:
                        return result

            # v9.14: 同花色副牌不够n张时，先出所有同花色副牌再补主牌/其他
            elif len(all_color_fu) > 0:
                # 先凑同花色副牌（优先对子）
                rank_groups2: dict[int, list] = {}
                for c in all_color_fu:
                    rank_groups2.setdefault(c.rank, []).append(c)
                pairs2 = []
                for rank in sorted(rank_groups2.keys()):
                    g = rank_groups2[rank]
                    while len(g) >= 2:
                        pairs2.append([g[0], g[1]])
                        g = g[2:]
                fu_result = []
                for p in pairs2:
                    fu_result.extend(p)
                used_ids2 = set(id(c) for c in fu_result)
                singles = sorted([c for c in all_color_fu if id(c) not in used_ids2],
                                key=lambda c: (c.has_score, c.rank))
                fu_result.extend(singles)
                
                # 补足n张：用主牌补
                remaining = n - len(fu_result)
                if remaining > 0:
                    a = self._get_analysis()
                    zhu_cards = sorted(a['zhudan'], key=lambda c: c.rank)
                    used_zhu_ids = set(id(c) for c in fu_result)
                    zhu_available = [c for c in zhu_cards if id(c) not in used_zhu_ids]
                    fu_result.extend(zhu_available[:remaining])
                
                if len(fu_result) == n:
                    return fu_result

        # 绝门
        if my_team_winning:
            # v11: 队友大→庄闲差异化贴分
            if self.player.is_banker:
                # 庄家：不贴分牌→优先出无分副牌
                non_score_fudan = self._get_non_score_fudan()
                if len(non_score_fudan) >= n:
                    return non_score_fudan[:n]
                # 贴副连对
                for other_color in sorted(fuliandui.keys()):
                    for chain in fuliandui[other_color]:
                        if len(chain) >= n:
                            return chain[:n]
                # 贴副对
                a = self._get_analysis()
                for other_color in sorted(a['fudui'].keys()):
                    if len(a['fudui'][other_color]) >= 2:
                        pair = a['fudui'][other_color][:2]
                        if n <= 2:
                            return pair
                        need = n - 2
                        if non_score_fudan:
                            extra = non_score_fudan[:need]
                            if len(pair) + len(extra) >= n:
                                return (pair + extra)[:n]
                # 出最小无分牌
                if non_score_fudan:
                    return non_score_fudan[:n]
                a2 = self._get_analysis()
                if a2['zhudan']:
                    nonscore_zhu = [c for c in a2['zhudan'] if not c.has_score]
                    if nonscore_zhu:
                        return nonscore_zhu[:n]
                    return a2['zhudan'][:n]
            else:
                # 闲家：积极贴分牌（得分过80=闲家赢）
                score_fudan = self._get_score_fudan()
                if len(score_fudan) >= n:
                    return sorted(score_fudan, key=lambda c: -c.score)[:n]
                for other_color in sorted(fuliandui.keys()):
                    for chain in fuliandui[other_color]:
                        if len(chain) >= n:
                            return chain[:n]
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
                # 贴分牌散牌（不够n张则用无分牌凑）
                if score_fudan:
                    result = sorted(score_fudan, key=lambda c: -c.score)[:n]
                    if len(result) < n:
                        non_score = self._get_non_score_fudan()
                        result += non_score[:n - len(result)]
                    if len(result) == n:
                        return result
        else:
            # 对手赢+绝门→毙牌（≥3分门槛）
            epoch_score = self._count_epoch_score(epoch_cards)
            should_trump = False
            if self._is_last_trick() and self.score_koupai > 0:
                should_trump = True
            elif epoch_score >= 5:
                should_trump = True
            elif epoch_score >= 3 and zhuliandui:
                should_trump = True

            if should_trump:
                for chain in zhuliandui:
                    if len(chain) >= n:
                        return chain[:n]
            if len(self._get_analysis()['zhudui']) >= 2:
                return self._get_analysis()['zhudui'][:2]

        return []

    def _follow_zhudan(self, first_card, epoch_cards, epoch_players,
                       zhudan, now_scores) -> list[Card]:
        """跟主单（v9.2）

        v9.2改进：对手大且管不住时，优先出无分主牌（不送分！）
        """
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)

        if zhudan:
            if my_team_winning:
                # v11: 队友大→庄闲差异化
                if self.player.is_banker:
                    # 庄家：不贴分主牌→出最小无分主牌
                    nonscore_zhu = [c for c in zhudan if not c.has_score]
                    if nonscore_zhu:
                        return [nonscore_zhu[0]]
                    return [zhudan[0]]
                else:
                    # 闲家：积极贴分主牌
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

        return []

    def _follow_zhudui(self, first_cards, epoch_cards, epoch_players,
                       zhudui, zhudan, now_scores) -> list[Card]:
        """跟主对（v9.2）

        v9.2: 管不住时优先出无分主对
        """
        my_team_winning = self._is_team_winning(epoch_cards, epoch_players)

        if len(zhudui) >= 2:
            if my_team_winning:
                # v11: 队友大→庄闲差异化
                if self.player.is_banker:
                    # 庄家：不贴分主对→出最小无分主对
                    nonscore_pairs = []
                    for i in range(0, len(zhudui), 2):
                        if i + 1 < len(zhudui) and zhudui[i].name == zhudui[i + 1].name:
                            if not zhudui[i].has_score:
                                nonscore_pairs.append(zhudui[i:i + 2])
                    if nonscore_pairs:
                        return nonscore_pairs[0]
                    return zhudui[:2]
                else:
                    # 闲家：积极贴分主对
                    score_pairs = []
                    for i in range(0, len(zhudui), 2):
                        if i + 1 < len(zhudui) and zhudui[i].name == zhudui[i + 1].name:
                            if zhudui[i].has_score:
                                score_pairs.append((zhudui[i:i + 2], zhudui[i].score))
                    if score_pairs:
                        score_pairs.sort(key=lambda x: -x[1])
                        return score_pairs[0][0]
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

        return []
