# -*- coding: utf-8 -*-
"""
DMC V9 AI - 用训练好的DMC模型替代规则AI做出牌决策

架构:
- 加载DMC V9 checkpoint (QNetworkV2, dueling=False)
- 庄家/闲家分别使用banker_net/xianjia_net
- 观测编码: 764维 = 693(base) + 71(v11 extra)
- 动作编码: 108维one-hot
- 出牌: 遍历合法动作，选Q值最高的

非出牌决策(亮主/换牌/扣牌)仍使用规则AI
"""

import sys
import os
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_dmc.q_network_v2 import QNetworkV2, DMCNetworkSetV2
from rl_dmc.env import (
    NUM_CARDS, cards_to_ids, cards_to_onehot,
    _CARD_TYPE_TO_IDS, _ID_TO_CARD_TYPE
)
from rl_shengji.obs_v11 import encode_obs_v11, OBS_DIM_V11
from rl_shengji.env import ShengjiGame

logger = logging.getLogger(__name__)


class DMCAI:
    """DMC V9 AI推理引擎
    
    用法:
        ai = DMCAI()  # 自动加载V9 best模型
        card_strs = ai.decide_play(room, player_id)
    """
    
    _instance = None  # 单例，避免重复加载模型
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_path=None, device='cpu'):
        if self._initialized:
            return
        
        self.device = device
        
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'models_v9', 'dmc_v9_best.pt')
        
        self.model_path = model_path
        
        self.network_set = DMCNetworkSetV2(
            obs_dim=OBS_DIM_V11,
            num_cards=NUM_CARDS,
            hidden_dim=512,
            action_embed_dim=128,
            dueling=False,
            device=device,
        )
        
        if os.path.exists(model_path):
            self.network_set.load(model_path)
            self.network_set.banker_net.eval()
            self.network_set.xianjia_net.eval()
            logger.info(f"DMC V9 模型加载成功: {model_path}")
        else:
            logger.error(f"DMC V9 模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self._initialized = True
    
    def decide_play(self, room, player_id):
        """DMC出牌决策
        
        Args:
            room: GameRoom实例
            player_id: 玩家座位号(0-3)
            
        Returns:
            list[str]: 出牌的card_type列表，如 ['10-a-10', '10-b-10']
            如果推理失败返回None
        """
        try:
            # 1. 构建764维观测
            obs = self._build_obs(room, player_id)
            
            # 2. 枚举合法动作
            legal_actions = self._get_legal_actions(room, player_id)
            
            if not legal_actions:
                return None
            
            if len(legal_actions) == 1:
                return legal_actions[0]['card_strs']
            
            # 3. 构建动作onehot张量
            action_onehots = torch.tensor(
                np.stack([a['onehot'] for a in legal_actions]),
                dtype=torch.float32,
                device=self.device,
            )
            
            # 4. 选择网络(庄家/闲家)
            is_banker = player_id in (room.bankers or [0, 2])
            net = self.network_set.get_net(is_banker)
            
            # 5. 推理
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                q_values = net.evaluate_actions(obs_tensor, action_onehots)
            
            # 6. 选Q值最高的
            best_idx = q_values.argmax().item()
            best_action = legal_actions[best_idx]
            
            logger.debug(f"DMC seat={player_id} banker={is_banker} "
                        f"legal={len(legal_actions)} best_idx={best_idx} "
                        f"q_max={q_values.max().item():.2f} q_min={q_values.min().item():.2f}")
            
            return best_action['card_strs']
            
        except Exception as e:
            logger.error(f"DMC推理失败 seat={player_id}: {e}", exc_info=True)
            return None
    
    def _build_obs(self, room, player_id):
        """构建764维观测向量
        
        复用obs_v11的encode_obs_v11，需要一个game包装器
        """
        game = _GameWrapper(room)
        obs = encode_obs_v11(game, player_id)
        return obs
    
    def _get_legal_actions(self, room, player_id):
        """枚举合法出牌并编码为108维onehot"""
        player = room.players[player_id]
        if not player:
            return []
        
        is_first = len(room.epoch_cards) == 0
        # 处理pending_clear
        pending_clear = getattr(room, '_pending_clear_epoch', False)
        if pending_clear:
            is_first = True
        
        options = room._enumerate_legal_plays(player, is_first)
        
        # 构建card_type -> Card映射
        all_cards = []
        for cards in player.cards_in_hand.values():
            all_cards.extend(cards)
        
        ct_to_card = {}
        for c in all_cards:
            ct_to_card.setdefault(c.card_type, []).append(c)
        
        # 转换并编码
        actions = []
        seen = set()
        used_global = set()
        
        for opt in options:
            card_strs = opt['cards']
            
            # 构建Card对象列表
            combo = []
            used_indices = {}
            for ct in card_strs:
                idx = used_indices.get(ct, 0)
                available = ct_to_card.get(ct, [])
                if idx < len(available):
                    combo.append(available[idx])
                    used_indices[ct] = idx + 1
                else:
                    from server.card import Card
                    combo.append(Card(ct))
            
            # 去重
            key = tuple(sorted(ct for ct in card_strs))
            if key in seen:
                continue
            seen.add(key)
            
            # 编码为108维onehot
            onehot = cards_to_onehot(combo, used_global.copy())
            for cid in cards_to_ids(combo, used_global.copy()):
                used_global.add(cid)
            
            actions.append({
                'card_strs': card_strs,
                'onehot': onehot,
            })
        
        if not actions and all_cards:
            onehot = cards_to_onehot([all_cards[0]])
            actions.append({
                'card_strs': [all_cards[0].card_type],
                'onehot': onehot,
            })
        
        return actions


class _GameWrapper:
    """轻量级GameRoom包装器
    
    obs_v11.py的encode_obs_v11(game, player_id)需要:
    - game.room: GameRoom实例 (直接访问)
    - game.get_state(player_id): 返回{'obs': ndarray(693,)} (base观测)
    
    base观测直接从room构建（与env.py _encode_observation一致）
    """
    
    def __init__(self, room):
        self.room = room
    
    def get_state(self, player_id):
        """返回base 693维obs"""
        from server.constants import LEVEL_ORDER
        from rl_shengji.env import NUM_CARDS, cards_to_ids
        
        room = self.room
        now_level = room.now_level
        now_color = room.now_color
        player = room.players[player_id]
        
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
        
        # 2. 当前轮出牌 (432维)
        trick_rep = np.zeros(NUM_CARDS * 4, dtype=np.float32)
        used_trick = set()
        pending_clear = getattr(room, '_pending_clear_epoch', False)
        if not pending_clear:
            for i, ec_cards in enumerate(room.epoch_cards):
                if i < 4:
                    ids = cards_to_ids(ec_cards, used_trick.copy())
                    for cid in ids:
                        trick_rep[i * NUM_CARDS + cid] = 1.0
                        used_trick.add(cid)
        
        # 3. 已出的牌 (108维)
        played_rep = np.zeros(NUM_CARDS, dtype=np.float32)
        used_played = set()
        for trick in getattr(room, 'epoch_history', []):
            for cards in trick['cards']:
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
        if room.bankers:
            banker_rep[room.bankers[0]] = 1.0
        
        # 7. 当前玩家 (4维)
        current_rep = np.zeros(4, dtype=np.float32)
        current_rep[player_id] = 1.0
        
        # 8. 分数 (20维)
        score_rep = np.zeros(20, dtype=np.float32)
        score_bucket_0 = min(int(room.score_now / 20), 9)
        score_rep[score_bucket_0] = 1.0
        banker_score_val = max(0, 100 - room.score_now)
        score_bucket_1 = min(int(banker_score_val / 20), 9) + 10
        score_rep[score_bucket_1] = 1.0
        
        obs = np.concatenate([
            hand_rep, trick_rep, played_rep,
            color_rep, level_rep, banker_rep, current_rep, score_rep,
        ])
        
        return {'obs': obs}


# ============================================================
# 全局接口
# ============================================================

_dmc_ai = None
_dmc_stats = {'dmc_calls': 0, 'dmc_success': 0, 'fallback_calls': 0}


def get_dmc_ai():
    """获取全局DMC AI实例（懒加载）"""
    global _dmc_ai
    if _dmc_ai is None:
        _dmc_ai = DMCAI()
    return _dmc_ai


def dmc_decide_play(room, player_id):
    """DMC出牌决策接口
    
    在game_engine.py的auto_play_current_robot()中，
    替换 AI(p,...).decide_play() 调用
    
    Args:
        room: GameRoom实例
        player_id: 玩家座位号
        
    Returns:
        list[str]: 出牌的card_type列表，或None（fallback到规则AI）
    """
    global _dmc_stats
    try:
        ai = get_dmc_ai()
        _dmc_stats['dmc_calls'] += 1
        result = ai.decide_play(room, player_id)
        if result is not None:
            _dmc_stats['dmc_success'] += 1
        return result
    except Exception as e:
        logger.error(f"DMC决策异常: {e}", exc_info=True)
        _dmc_stats['fallback_calls'] += 1
        return None


def get_dmc_stats():
    """获取DMC调用统计"""
    return _dmc_stats.copy()
