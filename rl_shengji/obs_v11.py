# -*- coding: utf-8 -*-
"""
v11 观测编码 - 精确信息 + 绝门推断 + 底牌详情 + 剩余总分

修复的bug：
1. epoch_history现已修复，base 693维中"已出的牌"108维不再全0
2. 分牌追踪从epoch_history精确统计，不再用粗略估算
3. 已出主牌从epoch_history精确统计
4. 已出分牌不再重复计数（epoch_history与epoch_cards互斥）

新增71维增强（替换v7/v8的32维buggy增强）：
1.  各玩家剩余手牌数 (4维)
2.  自己各花色牌数 (4维)
3.  自己主牌数 (1维)
4.  首出花色 (5维)
5.  上轮赢家 (4维)
6.  上轮闲家得分 (1维)
7.  上轮得分量 (1维)
8.  已出5分牌 (1维)
9.  已出10分牌 (1维)
10. 已出K牌 (1维)
11. 已出主牌数 (1维)
12. 场上剩余总分 (1维)
13. 闲家累计得分 (1维)
14. 庄家累计得分 (1维)
15. 底牌分数 (1维)
16. 底牌花色分布 (5维) — ♠♥♣♦+主 各几张/4
17. 底牌分牌分布 (3维) — 底牌中5/10/K各几张/4
18. 本轮分牌价值 (1维)
19. 出牌位置 (1维)
20. 队友已出 (1维)
21. 剩余轮数 (1维)
22. 角色 (1维)
23. 队友绝门确认 (5维)
24. 队友绝门推测 (5维)
25. 对手0绝门确认 (5维)
26. 对手0绝门推测 (5维)
27. 对手1绝门确认 (5维)
28. 对手1绝门推测 (5维)
= 4+4+1+5+4+1+1+1+1+1+1+1+1+1+1+5+3+1+1+1+1+1+5+5+5+5+5+5 = 71

总计: 693(base) + 71(v11) = 764维
"""

import numpy as np
from server.constants import SCORE_CARDS
from rl_shengji.env import NUM_CARDS

OBS_DIM_V11 = 764  # 693 base + 71 v11 extra

# 2副牌总分
TOTAL_SCORE = 200  # 4花色 × (5+10+K)×2副 = 4×25×2


def encode_obs_v11(game, player_id):
    """v11观测编码：693维base + 71维精确增强
    
    绝门推断规则（用户5/22纠正后）：
    - 确认绝门：首出某花色时，该玩家跟牌未出同花色(出了主牌或其他花色)，且手牌>0
      - 首出主牌时跟牌者出任意副牌不构成绝门判断
      - 首出者行为是自愿选择，不能判断绝门
    - 推测绝门：
      - 从扣底牌推断：扣底牌中某花色≥3张 → 推测该花色0.7
      - 多轮未出某花色(有出牌机会) → 0.3×min(次数/3, 1)
      - 如果后续观察到该玩家出了该花色，推测清零
    """
    room = game.room
    bankers = room.bankers or [0, 2]
    now_color = room.now_color
    now_level = room.now_level
    player = room.players[player_id]
    score_now = getattr(room, 'score_now', 0)
    score_koupai = getattr(room, 'score_koupai', 0)
    epoch_log = getattr(room, '_epoch_log', [])
    epoch_history = getattr(room, 'epoch_history', [])
    koupai_cards = getattr(room, 'koupai_cards', [])
    teammate_id = (player_id + 2) % 4
    opponents = [i for i in range(4) if i != player_id and i != teammate_id]
    
    # ====== 1. 各玩家剩余手牌数 (4维) ======
    hand_counts = np.zeros(4, dtype=np.float32)
    for i in range(4):
        if room.players[i]:
            hand_counts[i] = room.players[i].card_count / 26.0
    
    # ====== 2. 自己各花色牌数 (4维) ======
    my_suit_counts = np.zeros(4, dtype=np.float32)
    if player:
        for ci, color in enumerate(['a', 'b', 'c', 'd']):
            if color in player.cards_in_hand:
                my_suit_counts[ci] = len(player.cards_in_hand[color]) / 13.0
    
    # ====== 3. 自己主牌数 (1维) ======
    my_trump_count = np.zeros(1, dtype=np.float32)
    if player:
        tc = 0
        for cards in player.cards_in_hand.values():
            for card in cards:
                if card.is_zhu(now_level, now_color):
                    tc += 1
        my_trump_count[0] = tc / 27.0  # 1人最多约27张主牌(54/2)
    
    # ====== 4. 首出花色 (5维) ======
    lead_suit = np.zeros(5, dtype=np.float32)
    # 处理_pending_clear_epoch：如果待清空，epoch_cards还是上一轮的，不应编码
    pending_clear = getattr(room, '_pending_clear_epoch', False)
    if not pending_clear and room.epoch_cards and room.epoch_cards[0]:
        first_card = room.epoch_cards[0][0]
        if first_card.is_zhu(now_level, now_color):
            lead_suit[4] = 1.0
        elif first_card.color in ['a', 'b', 'c', 'd']:
            lead_suit[['a', 'b', 'c', 'd'].index(first_card.color)] = 1.0
    
    # ====== 5. 上轮赢家 (4维) ======
    last_winner = np.zeros(4, dtype=np.float32)
    lw = getattr(room, 'last_epoch_winner', -1)
    if 0 <= lw < 4:
        last_winner[lw] = 1.0
    
    # ====== 6. 上轮闲家得分 (1维) ======
    last_xianjia_won = np.zeros(1, dtype=np.float32)
    if epoch_log:
        last_xianjia_won[0] = 1.0 if epoch_log[-1].get('xianjia_won', False) else 0.0
    
    # ====== 7. 上轮得分量 (1维) ======
    last_score = np.zeros(1, dtype=np.float32)
    if epoch_log:
        last_score[0] = min(epoch_log[-1].get('score_earned', 0) / 20.0, 1.0)
    
    # ====== 8-11. 已出分牌精确统计 (4维) ======
    # 只统计epoch_history（已结算的轮次），不含当前轮（避免与base的当前轮出牌重复）
    # 当前轮的出牌已经在base 432维(当前轮出牌)中编码了，不需要重复
    played_5 = np.zeros(1, dtype=np.float32)
    played_10 = np.zeros(1, dtype=np.float32)
    played_K = np.zeros(1, dtype=np.float32)
    played_trump = np.zeros(1, dtype=np.float32)
    
    for trick in epoch_history:
        trick_cards = trick['cards'] if isinstance(trick, dict) else trick
        for card_list in trick_cards:
            for card in card_list:
                if hasattr(card, 'name'):
                    if card.name == '5':
                        played_5[0] += 1.0
                    elif card.name == '10':
                        played_10[0] += 1.0
                    elif card.name == 'K':
                        played_K[0] += 1.0
                if hasattr(card, 'is_zhu') and card.is_zhu(now_level, now_color):
                    played_trump[0] += 1.0
    
    played_5[0] /= 8.0       # 2副×4花色=8张5
    played_10[0] /= 8.0      # 8张10
    played_K[0] /= 8.0       # 8张K
    played_trump[0] /= 54.0  # 2副主牌: 将花色26+级牌6+固定主(2,3,5)18+王4=54
    
    # ====== 12. 场上剩余总分 (1维) ======
    # 用score_now精确计算: 闲家已得分=score_now
    # 已结算总分 = score_now + 庄家已结算分
    # 但庄家已结算分无法直接获取。用epoch_log累加更可靠：
    settled_score = 0.0
    for entry in epoch_log:
        settled_score += entry.get('score_earned', 0)
    # 当前轮未结算分牌：只统计epoch_cards中当前轮的牌
    # 关键：epoch_cards长度 <= epoch_log长度 的差值 = 当前未结算轮
    # 更安全：直接统计epoch_cards所有牌减去已在epoch_log中结算的
    # 简化方案：用epoch_cards长度 vs epoch_log长度判断
    current_epoch_count = len(room.epoch_cards) if not pending_clear else 0
    settled_epoch_count = len(epoch_log)
    current_trick_score = 0.0
    if current_epoch_count > settled_epoch_count:
        # 有未结算的轮（epoch_cards包含已结算+未结算的）
        # 未结算部分：epoch_cards[settled_epoch_count:]
        for card_list in room.epoch_cards[settled_epoch_count:]:
            for card in card_list:
                if hasattr(card, 'name'):
                    current_trick_score += SCORE_CARDS.get(card.name, 0)
    total_captured = settled_score + current_trick_score
    remaining_score = np.array([max(0, TOTAL_SCORE - total_captured) / TOTAL_SCORE], 
                               dtype=np.float32)
    
    # ====== 13. 闲家累计得分 (1维) ======
    xianjia_score = np.array([min(score_now / TOTAL_SCORE, 1.0)], dtype=np.float32)
    
    # ====== 14. 庄家累计得分 (1维) ======
    # 精确计算：从epoch_log中累加庄家赢轮的分数
    banker_earned = 0.0
    for entry in epoch_log:
        if not entry.get('xianjia_won', False):
            banker_earned += entry.get('score_earned', 0)
    banker_score = np.array([min(banker_earned / TOTAL_SCORE, 1.0)], dtype=np.float32)
    
    # ====== 15. 底牌分数 (1维) ======
    koupai_score = np.array([min(score_koupai / 40.0, 1.0)], dtype=np.float32)
    
    # ====== 16. 底牌花色分布 (5维) ======
    # ♠♥♣♦+主 各几张/4（最多4张底牌）
    koupai_suit_dist = np.zeros(5, dtype=np.float32)
    for card in koupai_cards:
        if hasattr(card, 'is_zhu') and card.is_zhu(now_level, now_color):
            koupai_suit_dist[4] += 1.0
        elif hasattr(card, 'color') and card.color in ['a', 'b', 'c', 'd']:
            koupai_suit_dist[['a', 'b', 'c', 'd'].index(card.color)] += 1.0
    koupai_suit_dist /= 4.0
    
    # ====== 17. 底牌分牌分布 (3维) ======
    # 底牌中5/10/K各几张/4
    koupai_score_dist = np.zeros(3, dtype=np.float32)
    for card in koupai_cards:
        if hasattr(card, 'name'):
            if card.name == '5':
                koupai_score_dist[0] += 1.0
            elif card.name == '10':
                koupai_score_dist[1] += 1.0
            elif card.name == 'K':
                koupai_score_dist[2] += 1.0
    koupai_score_dist /= 4.0
    
    # ====== 18. 本轮分牌价值 (1维) ======
    # 同样只统计未结算轮的分牌
    current_trick_score_for_norm = 0.0
    if not pending_clear:
        cur_ec = len(room.epoch_cards)
        set_ec = len(epoch_log)
        if cur_ec > set_ec:
            for card_list in room.epoch_cards[set_ec:]:
                for card in card_list:
                    if hasattr(card, 'name'):
                        current_trick_score_for_norm += SCORE_CARDS.get(card.name, 0)
    current_trick_score_norm = np.array([min(current_trick_score_for_norm / 30.0, 1.0)],
                                         dtype=np.float32)
    
    # ====== 19. 出牌位置 (1维) ======
    ec_count = len(room.epoch_cards) if not pending_clear else 0
    my_pos = np.array([ec_count / 4.0], dtype=np.float32)
    
    # ====== 20. 队友已出 (1维) ======
    teammate_played = np.zeros(1, dtype=np.float32)
    if not pending_clear and teammate_id in room.epoch_players:
        teammate_played[0] = 1.0
    
    # ====== 21. 剩余轮数 (1维) ======
    remaining_tricks = np.zeros(1, dtype=np.float32)
    if player:
        remaining_tricks[0] = player.card_count / 26.0
    
    # ====== 22. 角色 (1维) ======
    is_banker = np.array([1.0 if player_id in bankers else 0.0], dtype=np.float32)
    
    # ====== 23-28. 绝门推断 (30维) ======
    # 每个其他玩家10维 = 5花色(♠♥♣♦+主) × 2(确认/推测)
    void_teammate_confirmed = np.zeros(5, dtype=np.float32)
    void_teammate_suspected = np.zeros(5, dtype=np.float32)
    void_opp0_confirmed = np.zeros(5, dtype=np.float32)
    void_opp0_suspected = np.zeros(5, dtype=np.float32)
    void_opp1_confirmed = np.zeros(5, dtype=np.float32)
    void_opp1_suspected = np.zeros(5, dtype=np.float32)
    
    void_map = {
        teammate_id: (void_teammate_confirmed, void_teammate_suspected),
        opponents[0]: (void_opp0_confirmed, void_opp0_suspected),
        opponents[1]: (void_opp1_confirmed, void_opp1_suspected) if len(opponents) > 1 
                       else (void_opp1_confirmed, void_opp1_suspected),
    }
    
    # --- 确认绝门：从last_epoch + current_epoch推断 ---
    for card_lists, players in [
        (room.epoch_cards if not pending_clear else [], 
         room.epoch_players if not pending_clear else []),
        (getattr(room, 'last_epoch_cards', []),
         getattr(room, 'last_epoch_players', [])),
    ]:
        if not card_lists or not players or len(card_lists) < 2:
            continue
        
        first_cards = card_lists[0]
        if not first_cards:
            continue
        first_card = first_cards[0]
        
        # 首出主牌→不能从跟牌推断绝门
        if first_card.is_zhu(now_level, now_color):
            continue
        
        lead_color = first_card.color
        if lead_color not in ['a', 'b', 'c', 'd']:
            continue
        lead_suit_idx = ['a', 'b', 'c', 'd'].index(lead_color)
        
        for ci, (cards, pid) in enumerate(zip(card_lists, players)):
            if ci == 0:  # 首出者跳过
                continue
            if pid not in void_map:
                continue
            if not cards:
                continue
            # 对手剩余0张手牌不算绝门
            if room.players[pid] and room.players[pid].card_count == 0:
                continue
            
            # 检查是否跟了首出花色
            followed = any(
                not c.is_zhu(now_level, now_color) and c.color == lead_color
                for c in cards
            )
            if not followed:
                conf, susp = void_map[pid]
                conf[lead_suit_idx] = 1.0
    
    # --- 推测绝门：从扣底牌推断 ---
    # 扣底牌是公开信息，扣了≥3张同花色 → 推测扣底者该花色绝门
    koupai_player = getattr(room, 'koupai_picker_id', None)
    if koupai_player is None:
        koupai_player = getattr(room, 'liangzhu_player', None)
    if koupai_player is not None and koupai_player in void_map:
        suit_count = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        for card in koupai_cards:
            if hasattr(card, 'color') and card.color in suit_count and not card.is_zhu(now_level, now_color):
                suit_count[card.color] += 1
        conf, susp = void_map[koupai_player]
        for si, color in enumerate(['a', 'b', 'c', 'd']):
            if suit_count[color] >= 3:
                susp[si] = max(susp[si], 0.7)
            elif suit_count[color] >= 2:
                susp[si] = max(susp[si], 0.4)
    
    # ====== 组合 (4+4+1+5+4+1+1+1+1+1+1+1+1+1+1+5+3+1+1+1+1+1+5+5+5+5+5+5 = 71) ======
    extra = np.concatenate([
        hand_counts,              # 4
        my_suit_counts,           # 4
        my_trump_count,           # 1
        lead_suit,                # 5
        last_winner,              # 4
        last_xianjia_won,         # 1
        last_score,               # 1
        played_5,                 # 1
        played_10,                # 1
        played_K,                 # 1
        played_trump,             # 1
        remaining_score,          # 1
        xianjia_score,            # 1
        banker_score,             # 1
        koupai_score,             # 1
        koupai_suit_dist,         # 5
        koupai_score_dist,        # 3
        current_trick_score_norm, # 1
        my_pos,                   # 1
        teammate_played,          # 1
        remaining_tricks,         # 1
        is_banker,                # 1
        void_teammate_confirmed,  # 5
        void_teammate_suspected,  # 5
        void_opp0_confirmed,      # 5
        void_opp0_suspected,      # 5
        void_opp1_confirmed,      # 5
        void_opp1_suspected,      # 5
    ])
    
    assert extra.shape[0] == 71, f"extra维度错误: {extra.shape[0]} != 71"
    
    # 获取base obs (693维)
    base_obs = game.get_state(player_id)['obs']
    
    result = np.concatenate([base_obs, extra])
    assert result.shape[0] == OBS_DIM_V11, f"obs总维度错误: {result.shape[0]} != {OBS_DIM_V11}"
    return result
