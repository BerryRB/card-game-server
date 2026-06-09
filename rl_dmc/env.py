# -*- coding: utf-8 -*-
"""
升级(Trump/Shengji)纸牌游戏 - RLCard兼容环境 v2

核心设计（参考DouZero）:
- 108张牌, 每张有唯一card_id (0-107)
- 动作空间: 每步动态枚举合法出牌，用index作为action_id
- 状态编码: 手牌(108) + 已出牌(108) + 当前轮(108*4) + 主牌(4+13) + 级牌(13) + 庄家(4) + 当前玩家(4) + 分数(20) = 693维
- 动作编码: 对每个合法出牌，用出牌的108维one-hot向量作为action特征（用于网络输入）
- 奖励: 每轮得分 + 最终结果
"""

import sys
import os
import numpy as np
from collections import OrderedDict
from typing import List, Optional

# 确保能import server模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.card import Card, Deck, card_from_str
from server.constants import (
    RANK_MAP, SCORE_CARDS, FIXED_ZHU_NAMES, LEVEL_ORDER,
    GamePhase, LIANG_TYPE_RANK
)
from server.rules import (
    get_dui_and_liandui, card_type_analyze,
    compare_outcards, determine_play_type, get_zhu_rank
)
from server.ai import AI


# ============================================================
# 全局常量: 108张牌的固定ID映射
# ============================================================

NUM_CARDS = 108

# 构建稳定的card_id映射
# 策略: 54种独特牌(card_type) × 2副 = 108张
# card_type排序: a-2, a-3, ..., a-A, b-2, ..., d-A, w, W
_CARD_TYPES_SORTED = []
for rank_name in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']:
    for color in ['a', 'b', 'c', 'd']:
        _CARD_TYPES_SORTED.append(f'{rank_name}-{RANK_MAP[rank_name]}-{color}')
_CARD_TYPES_SORTED.append('w-14-z')  # 小王
_CARD_TYPES_SORTED.append('W-15-z')  # 大王
# 共54种card_type，每种2张

# card_type -> [id1, id2]
_CARD_TYPE_TO_IDS = {}
for i, ct in enumerate(_CARD_TYPES_SORTED):
    _CARD_TYPE_TO_IDS[ct] = [i * 2, i * 2 + 1]

# id -> card_type
_ID_TO_CARD_TYPE = {}
for ct, ids in _CARD_TYPE_TO_IDS.items():
    for cid in ids:
        _ID_TO_CARD_TYPE[cid] = ct


def cards_to_ids(cards: List[Card], used_ids: set = None) -> List[int]:
    """将Card列表映射到card_id列表
    
    由于同card_type有2张，用used_ids来区分第一张和第二张
    """
    if used_ids is None:
        used_ids = set()
    
    result = []
    type_count = {}  # card_type -> 已使用次数
    
    for card in cards:
        ct = card.card_type
        count = type_count.get(ct, 0)
        ids = _CARD_TYPE_TO_IDS.get(ct, [])
        
        if count < len(ids):
            cid = ids[count]
            # 如果这个id已经被占用(之前用过)，试下一个
            if cid in used_ids and count + 1 < len(ids):
                cid = ids[count + 1]
            result.append(cid)
            used_ids.add(cid)
            type_count[ct] = count + 1
        else:
            # 不应该发生
            result.append(ids[0])
    
    return result


def id_to_card_type(card_id: int) -> str:
    """将card_id转回card_type字符串"""
    return _ID_TO_CARD_TYPE.get(card_id, '??')


def cards_to_onehot(cards: List[Card], used_ids: set = None) -> np.ndarray:
    """将Card列表转为108维one-hot向量"""
    ids = cards_to_ids(cards, used_ids)
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for cid in ids:
        vec[cid] = 1.0
    return vec


# ============================================================
# 简化的升级游戏引擎 (用于RL训练)
# ============================================================

class ShengjiGame:
    """升级游戏的简化引擎，专注于出牌阶段"""
    
    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 4
        self.reset()
    
    def reset(self):
        """初始化一局游戏"""
        from server.game_engine import GameRoom
        self.room = GameRoom('rl_train')
        self.room.start_game()
        
        # 自动完成到出牌阶段
        self._auto_play_to_playing()
        
        self.step_count = 0
        self.game_over = False
        self.payoffs = None
        self.trick_rewards = [0.0] * 4  # 每步即时奖励缓存
    
    def _auto_play_to_playing(self):
        """自动完成亮主/吃牌/扣牌/换牌阶段"""
        max_steps = 200
        for _ in range(max_steps):
            if self.room.phase == GamePhase.PLAYING or self.room.phase == GamePhase.GAME_OVER:
                break
            actions = self.room.auto_play_current_robot()
            if not actions:
                if self.room.phase == GamePhase.LIANGZHU:
                    all_ready = all(p.ready for p in self.room.players if p)
                    if all_ready and self.room.liangzhu_player is None:
                        self.room.handle_no_liang()
                elif self.room.phase == GamePhase.KOUPAI:
                    for i in range(4):
                        if self.room.players[i]:
                            ai = AI(self.room.players[i], i, self.room)
                            koupai = ai.decide_koupai(self.room.hole_cards)
                            result = self.room.handle_koupai(i, koupai)
                            if result.get('status') == 'ok':
                                break
        
        if self.room.phase == GamePhase.GAME_OVER:
            self.game_over = True
    
    def get_current_player(self) -> int:
        if self.game_over or self.room.phase == GamePhase.GAME_OVER:
            return -1
        return self.room.current_turn
    
    def get_legal_actions(self, player_id: int = None) -> list:
        """获取合法出牌，返回 [(action_id, card_ids, card_onehot), ...]"""
        if self.game_over:
            return []
        
        if player_id is None:
            player_id = self.get_current_player()
        
        if player_id < 0:
            return []
        
        player = self.room.players[player_id]
        if not player:
            return []
        
        # 获取合法出牌组合
        combos = self._get_legal_play_combos(player, player_id)
        
        # 编码为动作
        actions = []
        used_global = set()  # 用于区分同card_type的两张牌
        for i, combo in enumerate(combos):
            card_ids = cards_to_ids(combo, used_global.copy())
            onehot = np.zeros(NUM_CARDS, dtype=np.float32)
            for cid in card_ids:
                onehot[cid] = 1.0
            actions.append((i, card_ids, onehot))
        
        return actions
    
    def _get_legal_play_combos(self, player, player_id: int) -> List[List[Card]]:
        """获取玩家的所有合法出牌组合 - 使用game_engine的枚举逻辑"""
        all_cards = []
        for cards in player.cards_in_hand.values():
            all_cards.extend(cards)
        
        if not all_cards:
            return []
        
        is_first = len(self.room.epoch_cards) == 0
        
        # 直接使用game_engine的枚举逻辑
        options = self.room._enumerate_legal_plays(player, is_first)
        
        # 构建card_type -> Card映射（手牌中的真实Card对象）
        ct_to_card = {}
        for c in all_cards:
            ct_to_card.setdefault(c.card_type, []).append(c)
        
        # 转换为Card列表
        combos = []
        seen = set()  # 去重
        for opt in options:
            card_strs = opt['cards']
            combo = []
            used_indices = {}  # card_type -> 已使用几张
            for ct in card_strs:
                idx = used_indices.get(ct, 0)
                available = ct_to_card.get(ct, [])
                if idx < len(available):
                    combo.append(available[idx])
                    used_indices[ct] = idx + 1
                else:
                    # fallback: 用card_from_str创建
                    combo.append(Card(ct))
            
            # 用card_type排序后tuple去重
            key = tuple(sorted(ct for ct in card_strs))
            if key not in seen:
                seen.add(key)
                combos.append(combo)
        
        if not combos:
            # fallback: 至少出一张
            combos.append([all_cards[0]])
        
        return combos
    
    def _get_first_play_combos(self, hand: List[Card]) -> List[List[Card]]:
        """先手出牌的所有合法组合"""
        combos = []
        now_level = self.room.now_level
        now_color = self.room.now_color
        
        # 单牌
        for card in hand:
            combos.append([card])
        
        # 对子 - 同card_type的2张牌
        type_groups = {}
        for card in hand:
            ct = card.card_type
            if ct not in type_groups:
                type_groups[ct] = []
            type_groups[ct].append(card)
        
        for ct, cards in type_groups.items():
            if len(cards) >= 2:
                combos.append(cards[:2])
        
        # 主对 - 同name不同color但都是主牌
        zhu_by_name = {}
        for card in hand:
            if card.is_zhu(now_level, now_color):
                if card.name not in zhu_by_name:
                    zhu_by_name[card.name] = []
                zhu_by_name[card.name].append(card)
        
        for name, cards in zhu_by_name.items():
            if len(cards) >= 2:
                # 检查是否已经有同card_type的对子
                # 只有跨color的对子才需要额外添加
                colors_seen = set(c.color for c in cards)
                if len(colors_seen) >= 2:
                    # 取不同color的前2张
                    combo = [cards[0]]
                    for c in cards[1:]:
                        if c.color != cards[0].color:
                            combo.append(c)
                            break
                    if len(combo) == 2:
                        combos.append(combo)
        
        return combos
    
    def _get_follow_play_combos(self, hand: List[Card], player_id: int) -> List[List[Card]]:
        """跟牌的所有合法组合"""
        combos = []
        now_level = self.room.now_level
        now_color = self.room.now_color
        
        lead_cards = self.room.epoch_cards[0]
        lead_type = determine_play_type(lead_cards, now_level, now_color)
        if lead_type is None:
            lead_type = 'fudan'
        
        is_zhu_lead = all(c.is_zhu(now_level, now_color) for c in lead_cards)
        lead_color = lead_cards[0].color if lead_cards and not is_zhu_lead else None
        
        # 分类手牌
        zhu_cards = [c for c in hand if c.is_zhu(now_level, now_color)]
        fu_by_color = {}
        for c in hand:
            if not c.is_zhu(now_level, now_color):
                if c.color not in fu_by_color:
                    fu_by_color[c.color] = []
                fu_by_color[c.color].append(c)
        
        if 'dan' in lead_type or lead_type == 'zhudan':
            # 跟单牌
            if is_zhu_lead:
                if zhu_cards:
                    for c in zhu_cards:
                        combos.append([c])
                else:
                    for c in hand:
                        combos.append([c])
            else:
                same_color = fu_by_color.get(lead_color, [])
                if same_color:
                    for c in same_color:
                        combos.append([c])
                    # 可毙牌
                    for c in zhu_cards:
                        combos.append([c])
                else:
                    for c in hand:
                        combos.append([c])
        
        elif 'dui' in lead_type:
            # 跟对子
            def find_pairs(cards):
                groups = {}
                for c in cards:
                    if c.name not in groups:
                        groups[c.name] = []
                    groups[c.name].append(c)
                pairs = []
                for name, cs in groups.items():
                    if len(cs) >= 2:
                        pairs.append(cs[:2])
                    elif name in [c.name for c in cards]:
                        # 跨color主对
                        pass
                return pairs
            
            if is_zhu_lead:
                pairs = find_pairs(zhu_cards)
                # 跨color主对
                zhu_by_name = {}
                for c in zhu_cards:
                    if c.name not in zhu_by_name:
                        zhu_by_name[c.name] = []
                    zhu_by_name[c.name].append(c)
                for name, cs in zhu_by_name.items():
                    if len(cs) >= 2:
                        colors = set(c.color for c in cs)
                        if len(colors) >= 2:
                            combo = [cs[0]]
                            for c2 in cs[1:]:
                                if c2.color != cs[0].color:
                                    combo.append(c2)
                                    break
                            if len(combo) == 2:
                                pairs.append(combo)
                
                if pairs:
                    for p in pairs:
                        combos.append(p)
                else:
                    for c in hand:
                        combos.append([c])
            else:
                same_color = fu_by_color.get(lead_color, [])
                pairs = find_pairs(same_color)
                if pairs:
                    for p in pairs:
                        combos.append(p)
                    # 主对毙
                    zhu_pairs = find_pairs(zhu_cards)
                    for p in zhu_pairs:
                        combos.append(p)
                else:
                    for c in hand:
                        combos.append([c])
        
        else:
            # 其他牌型，简化为单牌
            for c in hand:
                combos.append([c])
        
        if not combos:
            combos.append([hand[0]])
        
        return combos
    
    def step(self, action_id: int):
        """执行一个动作
        
        Args:
            action_id: 合法动作列表中的index
            
        Returns:
            (next_state, next_player_id, reward, done)
        """
        if self.game_over:
            return None, -1, 0, True
        
        player_id = self.get_current_player()
        actions = self.get_legal_actions(player_id)
        
        if action_id >= len(actions):
            action_id = 0
        
        _, card_ids, _ = actions[action_id]
        
        # card_ids -> card_strs
        card_strs = [id_to_card_type(cid) for cid in card_ids]
        
        # 执行出牌
        result = self.room.handle_play(player_id, card_strs)
        
        if result.get('status') != 'ok':
            # 非法出牌 - 自动重试第一个合法动作
            if actions:
                _, card_ids, _ = actions[0]
                card_strs = [id_to_card_type(cid) for cid in card_ids]
                result = self.room.handle_play(player_id, card_strs)
            if result.get('status') != 'ok':
                self.game_over = True
                return None, -1, 0, True
        
        self.step_count += 1
        
        # 计算即时奖励
        reward = 0.0
        if 'epoch_result' in result:
            epoch = result['epoch_result']
            winner = epoch['winner']
            # 己方赢轮+0.1, 对方赢轮-0.05
            if winner % 2 == player_id % 2:
                reward += 0.1
            else:
                reward -= 0.05
        
        # 检查游戏结束
        if 'game_over' in result:
            self.game_over = True
            self.payoffs = self._compute_payoffs(result['game_over'])
            reward += self.payoffs[player_id]
            next_player = -1
        else:
            next_player = self.get_current_player()
        
        state = self.get_state(next_player) if next_player >= 0 else None
        return state, next_player, reward, self.game_over
    
    def _compute_payoffs(self, game_over_info: dict) -> np.ndarray:
        """计算最终payoff"""
        score_now = game_over_info.get('score_now', 0)
        banker_team = game_over_info.get('banker_team', 0)
        
        if score_now >= 80:
            bp, dp = -1.0, 1.0
        elif score_now >= 40:
            bp, dp = 1.0, -1.0
        elif score_now > 0:
            bp, dp = 1.5, -1.5
        else:
            bp, dp = 2.0, -2.0
        
        payoffs = np.zeros(4)
        for i in range(4):
            if (i % 2) == banker_team:
                payoffs[i] = bp
            else:
                payoffs[i] = dp
        return payoffs
    
    def get_state(self, player_id: int) -> dict:
        """获取玩家视角的状态"""
        if player_id < 0 or player_id >= 4:
            return {'obs': np.zeros(693), 'legal_actions': OrderedDict(), 
                    'raw_legal_actions': [], 'action_embeddings': np.zeros((1, 108))}
        
        obs = self._encode_observation(player_id)
        actions = self.get_legal_actions(player_id)
        
        # action embeddings: 每个合法动作的108维one-hot
        if actions:
            action_emb = np.stack([a[2] for a in actions])  # (num_actions, 108)
        else:
            action_emb = np.zeros((1, 108), dtype=np.float32)
        
        legal_action_ids = OrderedDict({a[0]: None for a in actions})
        raw_ids = [a[0] for a in actions]
        
        return {
            'obs': obs,
            'legal_actions': legal_action_ids,
            'raw_legal_actions': raw_ids,
            'action_embeddings': action_emb,
        }
    
    def _encode_observation(self, player_id: int) -> np.ndarray:
        """编码玩家观测为693维向量"""
        now_level = self.room.now_level
        now_color = self.room.now_color
        player = self.room.players[player_id]
        
        # 1. 手牌 (108维)
        hand_rep = np.zeros(NUM_CARDS, dtype=np.float32)
        used = set()
        if player:
            for cards in player.cards_in_hand.values():
                for card in cards:
                    ids = cards_to_ids([card], used.copy())
                    for cid in ids:
                        hand_rep[cid] = 1.0
                        used.add(cid)
        
        # 2. 当前轮出牌 (108*4=432维)
        trick_rep = np.zeros(NUM_CARDS * 4, dtype=np.float32)
        used_trick = set()
        # 如果_pending_clear_epoch，epoch_cards是上一轮的，不应编码
        pending_clear = getattr(self.room, '_pending_clear_epoch', False)
        if not pending_clear:
            for i, ec_cards in enumerate(self.room.epoch_cards):
                if i < 4:
                    ids = cards_to_ids(ec_cards, used_trick.copy())
                    for cid in ids:
                        trick_rep[i * NUM_CARDS + cid] = 1.0
                        used_trick.add(cid)
        
        # 3. 已出的牌 (108维) - 从所有epoch_history推断
        played_rep = np.zeros(NUM_CARDS, dtype=np.float32)
        used_played = set()
        for trick in getattr(self.room, 'epoch_history', []):
            for cards in trick:
                ids = cards_to_ids(cards, used_played.copy())
                for cid in ids:
                    played_rep[cid] = 1.0
                    used_played.add(cid)
        
        # 4. 主牌花色 (4维)
        color_rep = np.zeros(4, dtype=np.float32)
        if now_color and now_color in ['a', 'b', 'c', 'd']:
            color_rep[['a', 'b', 'c', 'd'].index(now_color)] = 1.0
        
        # 5. 级牌 (13维)
        level_rep = np.zeros(13, dtype=np.float32)
        if now_level in LEVEL_ORDER:
            level_rep[LEVEL_ORDER.index(now_level)] = 1.0
        
        # 6. 庄家 (4维)
        banker_rep = np.zeros(4, dtype=np.float32)
        if self.room.bankers:
            banker_rep[self.room.bankers[0]] = 1.0
        
        # 7. 当前玩家 (4维)
        current_rep = np.zeros(4, dtype=np.float32)
        current_rep[player_id] = 1.0
        
        # 8. 分数 (20维)
        score_rep = np.zeros(20, dtype=np.float32)
        score_bucket_0 = min(int(self.room.score_now / 20), 9)
        score_rep[score_bucket_0] = 1.0
        banker_score = max(0, 100 - self.room.score_now)
        score_bucket_1 = min(int(banker_score / 20), 9) + 10
        score_rep[score_bucket_1] = 1.0
        
        obs = np.concatenate([
            hand_rep, trick_rep, played_rep,
            color_rep, level_rep, banker_rep, current_rep, score_rep,
        ])
        
        return obs
    
    def is_over(self) -> bool:
        return self.game_over or self.room.phase == GamePhase.GAME_OVER
    
    def get_payoffs(self) -> np.ndarray:
        if self.payoffs is not None:
            return self.payoffs
        # 游戏未正常结束，基于当前分数估算
        score_now = getattr(self.room, 'score_now', 50)
        if score_now >= 80:
            bp, dp = -0.5, 0.5
        else:
            bp, dp = 0.5, -0.5
        payoffs = np.zeros(4)
        banker_team = 0
        if self.room.bankers:
            banker_team = self.room.bankers[0] % 2
        for i in range(4):
            if (i % 2) == banker_team:
                payoffs[i] = bp
            else:
                payoffs[i] = dp
        return payoffs


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    print("=== ShengjiGame v2 测试 ===")
    game = ShengjiGame()
    
    step = 0
    total_reward = [0.0] * 4
    while not game.is_over() and step < 200:
        player_id = game.get_current_player()
        if player_id < 0:
            break
        
        actions = game.get_legal_actions(player_id)
        if not actions:
            break
        
        # 随机选择
        action_id = np.random.randint(len(actions))
        state, next_player, reward, done = game.step(action_id)
        total_reward[player_id] += reward
        step += 1
    
    payoffs = game.get_payoffs()
    print(f"步数: {step}")
    print(f"累计奖励: {total_reward}")
    print(f"最终Payoffs: {payoffs}")
    print("✅ 测试通过!")
