#!/usr/bin/env python3
"""
Pre-play 决策网络：亮主 / 吃牌 / 扣底

三个独立小网络，各自有独立的 obs / action / buffer / training。
与主出牌网络 (DMC V11) 完全解耦。

中间奖励设计：
  亮主 r_lz = w1*被吃惩罚 + w2*底牌捡主奖励 + w3*手牌质量变化
  吃牌 r_cp = w1*吃牌结果 + w2*底牌捡主奖励 + w3*手牌质量变化
  扣底 r_kp = w1*手牌质量变化 + w2*队友捡主 + w3*对手捡主
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import random
import copy

# ============================================================
# 手牌质量评估（独立函数，可脱离AI类使用）
# ============================================================

def compute_hand_quality(player, now_level, now_color):
    """评估手牌质量 (越高越好)，逻辑同 AI.evaluate_hand + 绝门加成
    
    Args:
        player: Player 对象
        now_level: 当前级牌名 (str, 如 '7')
        now_color: 主花色 (str, 如 'a')
    Returns:
        float: 手牌质量分数 (典型 0~200)
    """
    from server.ai import get_dui_and_liandui, card_type_analyze, FIXED_ZHU_NAMES
    
    dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
        player.cards_in_hand)
    result = card_type_analyze(
        dan_dict, kings, dui_dict, liandui_dict,
        now_level, now_color)
    zhudan, zhudui, zhuliandui, fudan, fudui, fuliandui, zhu_count, zhu_dan_count, fumax_dan = result

    total = 0.0
    # 主连对
    for chain in zhuliandui:
        total += 60 * (len(chain) / 2)
    # 主对
    for i in range(0, len(zhudui), 2):
        total += 25
    # 主单
    for card in zhudan:
        if card.is_big_joker:
            total += 20
        elif card.is_small_joker:
            total += 15
        elif card.name == now_level and card.color == now_color:
            total += 12
        elif card.name == now_level:
            total += 8
        elif card.name in FIXED_ZHU_NAMES:
            total += 5
    # 副单
    for color, cards in fudan.items():
        for c in cards:
            if c.has_score:
                total += 3
            elif c.rank >= 10:
                total += 1
    # 副对
    for color, cards in fudui.items():
        for i in range(0, len(cards), 2):
            total += 5
    # 副连对
    for color, chains in fuliandui.items():
        for chain in chains:
            total += 15 * (len(chain) / 2)
    
    # 绝门加成：每个绝门花色 +8 (战术价值)
    for color in ['a', 'b', 'c', 'd']:
        fu_cards = [c for c in player.cards_in_hand.get(color, [])
                    if not c.is_zhu(now_level, now_color)]
        if len(fu_cards) == 0:
            total += 8  # 绝门
    
    return total


# ============================================================
# 对手推断函数（不完全信息博弈）
# ============================================================

def _count_seen_cards(player, room=None):
    """统计已知的牌（自己的手牌 + 底牌），推断对手可能持有的牌
    
    Returns:
        dict: {name: 剩余未见面数}，如 {'5': 3, 'A': 2, ...}
    """
    seen = {}
    # 标准牌分布: 4花色 × 13name + 2王 = 54张
    for name in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']:
        seen[name] = 4
    seen['small_joker'] = 1
    seen['big_joker'] = 1
    
    # 减去自己手牌
    for color_cards in player.cards_in_hand.values():
        for c in color_cards:
            name = c.name
            if name in seen:
                seen[name] = max(0, seen[name] - 1)
    
    # 减去底牌（如果可见）
    if room and hasattr(room, 'hole_cards') and room.hole_cards:
        for c in room.hole_cards:
            name = c.name
            if name in seen:
                seen[name] = max(0, seen[name] - 1)
    
    return seen


def _infer_opponent_liangzhu_capabilities(player, now_level):
    """推断对手能亮主的能力
    
    亮主阶段，从自己的手牌推断对手可能持有什么亮牌型。
    
    Returns:
        dict: {
            'opp_has_joker_prob': float,      # 对手有王的概率
            'opp_has_duolian_prob': float,     # 对手有多连对的概率
            'opp_has_shuanglian_prob': float,  # 对手有双连对的概率
            'opp_has_danlian_prob': float,     # 对手有单连对的概率
            'my_best_lz_priority': float,      # 我最强亮牌型优先级(越小越强)
            'my_lz_type_count': int,           # 我有几种亮牌型可选
        }
    """
    from server.ai import get_dui_and_liandui
    
    seen = _count_seen_cards(player)
    
    # 推断对手持王概率
    my_jokers = sum(1 for c in sum(player.cards_in_hand.values(), [])
                    if c.is_joker)
    total_jokers = 2
    opp_jokers = total_jokers - my_jokers  # 对手队伍可能有的王
    # 2个对手共持有 opp_jokers 个王，概率 = opp_jokers/2
    opp_has_joker_prob = opp_jokers / 2.0
    
    # 推断对手有连对的概率
    # 简化模型：统计我手中的对子，推断剩余对子的分布
    dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
        player.cards_in_hand)
    
    # 我有各花色对子数
    my_pair_count = sum(len(v) for v in dui_dict.values()) // 2
    # 总对子数上限：13name × 2对/name = 26对(理论上)
    # 但每花色最多6-7对(26张/2)，用简化模型
    total_possible_pairs = 26  # 上限
    remaining_pairs = max(0, total_possible_pairs - my_pair_count)
    # 2个对手共52-my_hand张牌，含remaining_pairs个对子
    opp_pair_prob = min(1.0, remaining_pairs / 20.0)  # 简化
    
    # 对手有连对的概率（需要2+连续对同花色，概率更低）
    opp_has_duolian_prob = opp_pair_prob * 0.15   # 多连对: ~15%条件概率
    opp_has_shuanglian_prob = opp_pair_prob * 0.3  # 双连对: ~30%条件概率  
    opp_has_danlian_prob = opp_pair_prob * 0.8     # 单连对: ~80%条件概率
    
    # 我的最强亮牌型
    my_lz_types = []  # (priority, type_name)
    for color, chains in liandui_dict.items():
        for chain in chains:
            chain_len = len(chain) // 2  # 连对数
            if chain_len >= 3:
                my_lz_types.append((1, 'duolian'))     # 多连对 prio=1
            elif chain_len >= 2 and my_jokers > 0:
                my_lz_types.append((2, 'shuanglian'))   # 双连对 prio=2
    
    # 单连对(对子) + 王
    for color, pairs in dui_dict.items():
        if len(pairs) >= 2 and my_jokers > 0:
            my_lz_types.append((2, 'shuanglian'))
        if len(pairs) >= 2:
            my_lz_types.append((3, 'danlian'))
    
    # 王连对
    if my_jokers >= 2:
        my_lz_types.append((0, 'wanglian'))  # 王对 prio=0 最强
    
    # 三王
    if my_jokers >= 2:
        my_lz_types.append((0, 'wanglian'))
    
    if my_lz_types:
        my_best_priority = min(p for p, _ in my_lz_types)
        my_type_count = len(set(t for _, t in my_lz_types))
    else:
        my_best_priority = 4.0  # 没有亮牌型，最弱
        my_type_count = 0
    
    return {
        'opp_has_joker_prob': opp_has_joker_prob,
        'opp_has_duolian_prob': opp_has_duolian_prob,
        'opp_has_shuanglian_prob': opp_has_shuanglian_prob,
        'opp_has_danlian_prob': opp_has_danlian_prob,
        'my_best_lz_priority': my_best_priority / 4.0,  # 归一化
        'my_lz_type_count': min(my_type_count, 4) / 4.0,
    }


def _infer_chipai_risk_reward(player, now_level, now_color, liangzhu_type):
    """推断吃牌的风险和收益
    
    Returns:
        dict: {
            'chipai_success_prob': float,   # 吃牌成功概率估计
            'chipai_fail_cost': float,      # 吃牌失败代价(被返牌的损失)
            'my_best_cp_priority': float,   # 我最强吃牌型优先级
            'liangzhu_type_rank': float,    # 对手亮牌型的强度(0~1)
            'return_card_risk': float,      # 返牌风险(手牌中弱牌占比)
        }
    """
    from server.ai import get_dui_and_liandui
    
    # 亮牌型强度排序: duolian=1(最强) > shuanglian=2 > danlian=3(最弱)
    type_priority = {'duolian': 1, 'shuanglian': 2, 'danlian': 3, 'wanglian': 0}
    opp_priority = type_priority.get(liangzhu_type, 4)
    
    # 对手亮牌型强度(归一化到0~1, 越小越强)
    liangzhu_type_rank = opp_priority / 4.0
    
    # 我能压过对手的概率估计
    dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
        player.cards_in_hand)
    
    my_jokers = sum(1 for c in kings if c.is_joker)
    
    # 我有哪些更强的牌型
    can_beat = False
    my_best_cp_priority = 4.0
    
    # 王连对(最强)
    if my_jokers >= 2 and opp_priority > 0:
        can_beat = True
        my_best_cp_priority = 0
    
    # 多连对
    for color, chains in liandui_dict.items():
        for chain in chains:
            chain_len = len(chain) // 2
            if chain_len >= 3 and opp_priority > 1:
                can_beat = True
                my_best_cp_priority = min(my_best_cp_priority, 1)
    
    # 双连对(王+2连对)
    if my_jokers >= 1:
        for color, chains in liandui_dict.items():
            for chain in chains:
                if len(chain) // 2 >= 2 and opp_priority > 2:
                    can_beat = True
                    my_best_cp_priority = min(my_best_cp_priority, 2)
    
    # 吃牌成功概率: 能压=0.7(可能还有更强对手), 不能=0.0
    chipai_success_prob = 0.7 if can_beat else 0.0
    
    # 吃牌失败代价: 返牌损失估计
    # 返牌时吃牌者要还给被吃者等量的牌，优先还弱牌
    hand_size = sum(len(cards) for cards in player.cards_in_hand.values())
    weak_cards = sum(1 for color_cards in player.cards_in_hand.values()
                     for c in color_cards
                     if not c.is_zhu(now_level, now_color) and not c.has_score)
    return_card_risk = 1.0 - (weak_cards / max(hand_size, 1))  # 弱牌少=返牌风险大
    
    # 吃牌失败代价(如果弱牌多则代价小)
    chipai_fail_cost = return_card_risk * 0.5  # 归一化
    
    return {
        'chipai_success_prob': chipai_success_prob,
        'chipai_fail_cost': chipai_fail_cost,
        'my_best_cp_priority': my_best_cp_priority / 4.0,
        'liangzhu_type_rank': liangzhu_type_rank,
        'return_card_risk': return_card_risk,
    }


# ============================================================
# Obs 编码器
# ============================================================

def encode_lz_obs(player, now_level):
    """亮主阶段 obs (80维)
    
    基础60维(手牌结构) + 20维博弈特征(对手推断+风险评估)。
    此时主花色未知，只编码手牌结构信息。
    """
    from server.card import Card
    from server.ai import get_dui_and_liandui
    
    obs = np.zeros(80, dtype=np.float32)
    
    # === 基础特征: 60维 ===
    
    # 1. 手牌 one-hot: 4花色 × 13点数 = 52维 (obs[0:52])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        for card in player.cards_in_hand.get(color, []):
            rank_idx = min(card.rank - 2, 12)  # 2→0, ..., A→12
            obs[color_idx * 13 + rank_idx] = 1.0
    
    # 2. 王: 2维 (obs[52:54])
    dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
        player.cards_in_hand)
    obs[52] = 1.0 if any(c.is_big_joker for c in kings) else 0.0
    obs[53] = 1.0 if any(c.is_small_joker for c in kings) else 0.0
    
    # 3. 级牌数量: 1维 (obs[54])
    level_count = sum(1 for color_cards in player.cards_in_hand.values()
                      for c in color_cards if c.name == now_level)
    obs[54] = level_count / 4.0  # 归一化 0~1
    
    # 4. 手牌总分: 1维 (obs[55])
    score_count = sum(1 for color_cards in player.cards_in_hand.values()
                      for c in color_cards if c.has_score)
    obs[55] = score_count / 8.0  # 归一化 0~1
    
    # 5. 各花色对子数: 4维 (obs[56:60])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        pairs = len(dui_dict.get(color, [])) // 2
        obs[56 + color_idx] = pairs / 4.0  # 归一化 0~1
    
    # === 博弈特征: 20维 (obs[60:80]) ===
    
    infer = _infer_opponent_liangzhu_capabilities(player, now_level)
    
    # 6. 对手持王概率: 1维 (obs[60])
    obs[60] = infer['opp_has_joker_prob']
    
    # 7. 对手有多连对概率: 1维 (obs[61])
    obs[61] = infer['opp_has_duolian_prob']
    
    # 8. 对手有双连对概率: 1维 (obs[62])
    obs[62] = infer['opp_has_shuanglian_prob']
    
    # 9. 对手有单连对概率: 1维 (obs[63])
    obs[63] = infer['opp_has_danlian_prob']
    
    # 10. 我最强亮牌型优先级: 1维 (obs[64])
    #     0=王连对(最强), 1=多连对, 2=双连对, 3=单连对, 4=无
    obs[64] = infer['my_best_lz_priority']
    
    # 11. 我有几种亮牌型: 1维 (obs[65])
    obs[65] = infer['my_lz_type_count']
    
    # 12. 各花色连对数: 4维 (obs[66:70])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        chains = liandui_dict.get(color, [])
        chain_count = sum(len(chain) // 2 for chain in chains)
        obs[66 + color_idx] = min(chain_count / 4.0, 1.0)
    
    # 13. 手牌中王的总数: 1维 (obs[70])
    my_joker_count = sum(1 for c in sum(player.cards_in_hand.values(), [])
                         if c.is_joker)
    obs[70] = my_joker_count / 2.0
    
    # 14. 被吃风险估计: 1维 (obs[71])
    # 如果我有弱亮牌型(单连对)，被吃风险高
    # 风险 = 对手有更强牌型的概率 × 我亮弱牌的概率
    if infer['my_best_lz_priority'] >= 0.75:  # 只有单连对或无
        risk = infer['opp_has_duolian_prob'] * 0.8 + infer['opp_has_shuanglian_prob'] * 0.4
    elif infer['my_best_lz_priority'] >= 0.5:  # 双连对
        risk = infer['opp_has_duolian_prob'] * 0.6
    else:  # 多连对或王连对
        risk = 0.0
    obs[71] = min(risk, 1.0)
    
    # 15. 各花色非主牌数: 4维 (obs[72:76]) — 绝门可能性
    # 注意：亮主阶段主花色未定，用花色牌数代替
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        fu_count = len(player.cards_in_hand.get(color, []))
        obs[72 + color_idx] = fu_count / 13.0
    
    # 16. 手牌中分牌在各花色的分布: 4维 (obs[76:80])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        score_in_color = sum(1 for c in player.cards_in_hand.get(color, [])
                             if c.has_score)
        obs[76 + color_idx] = score_in_color / 4.0
    
    return obs


def encode_cp_obs(player, now_level, now_color, liangzhu_player_id,
                  liangzhu_type, liangzhu_color, player_id):
    """吃牌阶段 obs (110维)
    
    基础60维(手牌) + 20维亮牌信息 + 30维博弈特征(风险收益推断)。
    """
    from server.card import Card
    from server.ai import get_dui_and_liandui, FIXED_ZHU_NAMES, card_type_analyze
    
    obs = np.zeros(110, dtype=np.float32)
    
    # === 基础特征: 60维 (obs[0:60]) — 同 LZ ===
    
    # 1. 手牌 one-hot: 4花色 × 13点数 = 52维 (obs[0:52])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        for card in player.cards_in_hand.get(color, []):
            rank_idx = min(card.rank - 2, 12)
            obs[color_idx * 13 + rank_idx] = 1.0
    
    # 2. 王: 2维 (obs[52:54])
    dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
        player.cards_in_hand)
    obs[52] = 1.0 if any(c.is_big_joker for c in kings) else 0.0
    obs[53] = 1.0 if any(c.is_small_joker for c in kings) else 0.0
    
    # 3. 级牌数量: 1维 (obs[54])
    level_count = sum(1 for color_cards in player.cards_in_hand.values()
                      for c in color_cards if c.name == now_level)
    obs[54] = level_count / 4.0
    
    # 4. 手牌总分: 1维 (obs[55])
    score_count = sum(1 for color_cards in player.cards_in_hand.values()
                      for c in color_cards if c.has_score)
    obs[55] = score_count / 8.0
    
    # 5. 各花色对子数: 4维 (obs[56:60])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        pairs = len(dui_dict.get(color, [])) // 2
        obs[56 + color_idx] = pairs / 4.0
    
    # === 亮牌信息: 20维 (obs[60:80]) ===
    
    # 6. 主花色已知: 4维 one-hot (obs[60:64])
    color_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    if now_color and now_color in color_map:
        obs[60 + color_map[now_color]] = 1.0
    
    # 7. 主牌数量: 1维 (obs[64])
    zhu_count = sum(1 for color_cards in player.cards_in_hand.values()
                    for c in color_cards if c.is_zhu(now_level, now_color))
    obs[64] = zhu_count / 10.0
    
    # 8. 对手亮牌类型: 4维 (obs[65:69])
    #    duolian=0, shuanglian=1, danlian=2, wanglian=3
    type_map = {'duolian': 0, 'shuanglian': 1, 'danlian': 2, 'wanglian': 3}
    if liangzhu_type and liangzhu_type in type_map:
        obs[65 + type_map[liangzhu_type]] = 1.0
    
    # 9. 对手亮牌花色: 4维 (obs[69:73])
    if liangzhu_color and liangzhu_color in color_map:
        obs[69 + color_map[liangzhu_color]] = 1.0
    
    # 10. 队友身份: 1维 (obs[73]) — 亮主者是否是队友
    is_teammate = 1.0 if (liangzhu_player_id % 2) == (player_id % 2) else 0.0
    obs[73] = is_teammate
    
    # 11. 主牌对子数: 1维 (obs[74])
    result = card_type_analyze(dan_dict, kings, dui_dict, liandui_dict,
                                now_level, now_color)
    zhudui = result[1]
    obs[74] = (len(zhudui) // 2) / 4.0
    
    # 12. 主连对数: 1维 (obs[75])
    zhuliandui = result[2]
    obs[75] = len(zhuliandui) / 3.0
    
    # 13. 主牌总数: 1维 (obs[76])
    obs[76] = result[6] / 15.0
    
    # 14. 亮主者是否是庄家: 1维 (obs[77])
    # 吃牌时不知道谁是庄家，留0（后续可传入）
    obs[77] = 0.0
    
    # 15. 我的非亮主队身份: 1维 (obs[78])
    # 1=我是非亮主队(可吃), 0=我是亮主队(不可吃)
    is_opponent_team = 1.0 if (player_id % 2) != (liangzhu_player_id % 2) else 0.0
    obs[78] = is_opponent_team
    
    # 16. 队友是否已经pass: 1维 (obs[79])
    # 需要外部传入，暂留0
    obs[79] = 0.0
    
    # === 博弈特征: 30维 (obs[80:110]) ===
    
    infer = _infer_chipai_risk_reward(player, now_level, now_color, liangzhu_type)
    
    # 17. 吃牌成功概率: 1维 (obs[80])
    obs[80] = infer['chipai_success_prob']
    
    # 18. 吃牌失败代价: 1维 (obs[81])
    obs[81] = infer['chipai_fail_cost']
    
    # 19. 我最强吃牌型优先级: 1维 (obs[82])
    obs[82] = infer['my_best_cp_priority']
    
    # 20. 对手亮牌型强度: 1维 (obs[83])
    obs[83] = infer['liangzhu_type_rank']
    
    # 21. 返牌风险: 1维 (obs[84])
    obs[84] = infer['return_card_risk']
    
    # 22. 各花色弱牌(非主非分)数: 4维 (obs[85:89])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        weak = sum(1 for c in player.cards_in_hand.get(color, [])
                   if not c.is_zhu(now_level, now_color) and not c.has_score)
        obs[85 + color_idx] = weak / 8.0
    
    # 23. 各花色分牌数: 4维 (obs[89:93])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        sc = sum(1 for c in player.cards_in_hand.get(color, [])
                 if c.has_score)
        obs[89 + color_idx] = sc / 4.0
    
    # 24. 各花色绝门状态: 4维 (obs[93:97])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        fu_count = sum(1 for c in player.cards_in_hand.get(color, [])
                       if not c.is_zhu(now_level, now_color))
        obs[93 + color_idx] = 1.0 if fu_count == 0 else 0.0
    
    # 25. 不吃牌的隐含代价: 1维 (obs[97])
    # 对手亮主后确定主花色，如果我们不吃，主花色由对手选定
    # 代价取决于我们的手牌和主花色的匹配度
    if now_color:
        my_zhu_in_color = sum(1 for c in player.cards_in_hand.get(now_color, [])
                              if c.is_zhu(now_level, now_color))
        obs[97] = 1.0 - (my_zhu_in_color / max(zhu_count, 1))
    else:
        obs[97] = 0.5  # 未知
    
    # 26. 吃牌后预估手牌质量提升: 1维 (obs[98])
    # 简化：如果我能吃成功，我获得对手的亮牌+返牌，对手削弱
    # 预估 = 吃牌成功概率 × 0.5
    obs[98] = infer['chipai_success_prob'] * 0.5
    
    # 27. 对手亮牌中已知分牌数推断: 1维 (obs[99])
    # 亮主牌型中可能含有5/10/K，不知道具体但可推断概率
    # 对手亮连对时，分牌概率 ~3/13
    if liangzhu_type == 'duolian':
        obs[99] = 0.6  # 多连对含分牌概率高
    elif liangzhu_type == 'shuanglian':
        obs[99] = 0.4  # 双连对
    elif liangzhu_type == 'danlian':
        obs[99] = 0.3  # 单连对
    else:
        obs[99] = 0.0  # 王连对无分
    
    # 28. 我手牌中可用于返牌的弱牌比例: 1维 (obs[100])
    hand_size = sum(len(cards) for cards in player.cards_in_hand.values())
    weak_total = sum(1 for color_cards in player.cards_in_hand.values()
                     for c in color_cards
                     if not c.is_zhu(now_level, now_color) and not c.has_score)
    obs[100] = weak_total / max(hand_size, 1)
    
    # 29. 手牌量差: 1维 (obs[101]) — 我vs对手手牌数差异
    # 吃牌时双方手牌数相同(发牌阶段)，但返牌后会变化
    obs[101] = 0.0  # 发牌阶段手牌数相同
    
    # 30. 亮牌张数: 1维 (obs[102])
    # 对手亮了几张牌 (多连对6+, 双连对5+, 单连对2)
    type_card_count = {'duolian': 6, 'shuanglian': 5, 'danlian': 2, 'wanglian': 2}
    card_count = type_card_count.get(liangzhu_type, 0)
    obs[102] = card_count / 8.0
    
    # 31. padding: 7维 (obs[103:110]) — 预留给未来博弈特征
    # (如: 历史对局中该对手吃牌习惯、当前轮次等)
    
    return obs


def encode_kp_obs(player, now_level, now_color, hole_cards,
                  liangzhu_player_id, player_id):
    """扣底阶段 obs (120维)
    
    基础60维(手牌) + 20维亮牌+底牌信息 + 20维主牌分析 + 20维博弈特征。
    扣底是庄家的关键决策——扣什么牌给对手捡，留什么牌给自己。
    """
    from server.card import Card
    from server.ai import get_dui_and_liandui, card_type_analyze, FIXED_ZHU_NAMES
    
    obs = np.zeros(130, dtype=np.float32)
    
    # === 基础特征: 60维 (obs[0:60]) ===
    
    # 1. 手牌 one-hot: 4花色 × 13点数 = 52维 (obs[0:52])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        for card in player.cards_in_hand.get(color, []):
            rank_idx = min(card.rank - 2, 12)
            obs[color_idx * 13 + rank_idx] = 1.0
    
    # 2. 王: 2维 (obs[52:54])
    dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
        player.cards_in_hand)
    obs[52] = 1.0 if any(c.is_big_joker for c in kings) else 0.0
    obs[53] = 1.0 if any(c.is_small_joker for c in kings) else 0.0
    
    # 3. 级牌数量: 1维 (obs[54])
    level_count = sum(1 for color_cards in player.cards_in_hand.values()
                      for c in color_cards if c.name == now_level)
    obs[54] = level_count / 4.0
    
    # 4. 手牌总分: 1维 (obs[55])
    score_count = sum(1 for color_cards in player.cards_in_hand.values()
                      for c in color_cards if c.has_score)
    obs[55] = score_count / 8.0
    
    # 5. 各花色对子数: 4维 (obs[56:60])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        pairs = len(dui_dict.get(color, [])) // 2
        obs[56 + color_idx] = pairs / 4.0
    
    # === 亮牌+底牌信息: 20维 (obs[60:80]) ===
    
    # 6. 主花色: 4维 (obs[60:64])
    color_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    if now_color and now_color in color_map:
        obs[60 + color_map[now_color]] = 1.0
    
    # 7. 主牌数量: 1维 (obs[64])
    zhu_count = sum(1 for color_cards in player.cards_in_hand.values()
                    for c in color_cards if c.is_zhu(now_level, now_color))
    obs[64] = zhu_count / 10.0
    
    # 8. 主牌分析 (obs[65:68])
    result = card_type_analyze(dan_dict, kings, dui_dict, liandui_dict,
                                now_level, now_color)
    zhudui, zhuliandui = result[1], result[2]
    obs[65] = (len(zhudui) // 2) / 4.0   # 主对数
    obs[66] = len(zhuliandui) / 3.0       # 主连对数
    obs[67] = result[6] / 15.0            # 主牌总数 / 15
    
    # 9. 底牌信息 (obs[68:76])
    # 底牌 one-hot (4张×2维 = 8维: 花色+分值)
    for i, card in enumerate(hole_cards[:4]):
        if card.color in color_map:
            obs[68 + i * 2] = color_map[card.color] / 3.0  # 花色
        obs[68 + i * 2 + 1] = 1.0 if card.has_score else 0.0  # 有分
    
    # 10. 底牌分值: 1维 (obs[76])
    from server.constants import SCORE_CARDS
    hole_score = sum(SCORE_CARDS.get(c.name, 0) for c in hole_cards)
    obs[76] = hole_score / 20.0
    
    # 11. 底牌中主花色牌数: 1维 (obs[77])
    hole_zhu = sum(1 for c in hole_cards if c.is_zhu(now_level, now_color))
    obs[77] = hole_zhu / 4.0
    
    # 12. 亮主者身份: 2维 (obs[78:80])
    if liangzhu_player_id == player_id:
        obs[78] = 1.0  # 自己是亮主者
    elif (liangzhu_player_id % 2) == (player_id % 2):
        obs[79] = 1.0  # 队友是亮主者
    
    # === 绝门+副牌分析: 20维 (obs[80:100]) ===
    
    # 13. 各花色非主牌数: 4维 (obs[80:84])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        fu_count = sum(1 for c in player.cards_in_hand.get(color, [])
                       if not c.is_zhu(now_level, now_color))
        obs[80 + color_idx] = fu_count / 8.0
    
    # 14. 各花色绝门状态: 4维 (obs[84:88])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        fu_count = sum(1 for c in player.cards_in_hand.get(color, [])
                       if not c.is_zhu(now_level, now_color))
        obs[84 + color_idx] = 1.0 if fu_count == 0 else 0.0
    
    # 15. 各花色分牌数: 4维 (obs[88:92])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        sc = sum(1 for c in player.cards_in_hand.get(color, []) if c.has_score)
        obs[88 + color_idx] = sc / 4.0
    
    # 16. 各花色弱牌(非主非分)数: 4维 (obs[92:96])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        weak = sum(1 for c in player.cards_in_hand.get(color, [])
                   if not c.is_zhu(now_level, now_color) and not c.has_score)
        obs[92 + color_idx] = weak / 8.0
    
    # 17. 各花色副对数: 4维 (obs[96:100])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        fu_pairs = 0
        for i in range(0, len(dui_dict.get(color, [])), 2):
            c1 = dui_dict[color][i]
            if not c1.is_zhu(now_level, now_color):
                fu_pairs += 1
        obs[96 + color_idx] = fu_pairs / 4.0
    
    # === 博弈特征: 30维 (obs[100:130]) ===
    
    # 18. 自己主牌强度评估: 1维 (obs[100])
    # 综合：主牌数 + 主对 + 主连对 + 王 → 越强越敢扣分
    zhu_strength = (zhu_count / 10.0 * 0.3 +
                    (len(zhudui) // 2) / 4.0 * 0.3 +
                    len(zhuliandui) / 3.0 * 0.25 +
                    (obs[52] + obs[53]) * 0.15)
    obs[100] = zhu_strength
    
    # 19. 自己主牌强度分级: 3维 (obs[101:104])
    # 弱(<0.3) / 中(0.3~0.6) / 强(>0.6) one-hot
    if zhu_strength < 0.3:
        obs[101] = 1.0  # 弱主
    elif zhu_strength < 0.6:
        obs[102] = 1.0  # 中主
    else:
        obs[103] = 1.0  # 强主
    
    # 20. 队友吃牌结果: 2维 (obs[104:106])
    # 需要外部传入，默认[0,0]
    # obs[104]=1 → 队友吃牌成功(队友主牌强，能帮忙守底)
    # obs[105]=1 → 我方被对手吃牌(我方主牌被削弱)
    obs[104] = 0.0  # 队友chipai_success
    obs[105] = 0.0  # 被对手吃牌
    
    # 21. 队友主牌强度推测: 1维 (obs[106])
    # 推断依据：
    #   - 队友吃牌成功 → 队友主牌强 (0.8+)
    #   - 队友未吃/无吃牌机会 → 中性 (0.5)
    #   - 对手吃牌成功 → 队友主牌可能弱 (0.3)
    # 默认0.5，train.py中根据实际吃牌结果更新
    obs[106] = 0.5  # teammate_zhu_strength_estimate
    
    # 22. 综合守底能力评估: 1维 (obs[107])
    # = 自己主牌强度 × 0.6 + 队友主牌推测 × 0.4
    # 越高越敢扣分到底牌
    defend_ability = zhu_strength * 0.6 + obs[106] * 0.4
    obs[107] = defend_ability
    
    # 23. 对手捡主概率推断: 1维 (obs[108])
    obs[108] = 0.5  # 先验概率
    
    # 24. 队友捡主概率推断: 1维 (obs[109])
    obs[109] = 0.5
    
    # 25. 底牌被对手捡到时的分值损失: 1维 (obs[110])
    obs[110] = hole_score / 40.0 * 0.5  # 期望损失
    
    # 26. 手牌中可安全扣的弱牌数: 1维 (obs[111])
    safe_cards = sum(1 for color_cards in player.cards_in_hand.values()
                     for c in color_cards
                     if not c.is_zhu(now_level, now_color) 
                     and not c.has_score 
                     and c.rank < 8)  # 低牌
    obs[111] = safe_cards / 10.0
    
    # 27. 手牌中危险分牌数(必须留住的): 1维 (obs[112])
    danger_score = sum(1 for color_cards in player.cards_in_hand.values()
                       for c in color_cards
                       if c.has_score and c.is_zhu(now_level, now_color))
    obs[112] = danger_score / 4.0
    
    # 28. 扣底后手牌质量预估: 1维 (obs[113])
    current_hq = compute_hand_quality(player, now_level, now_color)
    obs[113] = min(current_hq / 200.0, 1.0)
    
    # 29. 底牌中各花色非主牌数: 4维 (obs[114:118])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        fu_hole = sum(1 for c in hole_cards 
                      if c.color == color and not c.is_zhu(now_level, now_color))
        obs[114 + color_idx] = fu_hole / 4.0
    
    # 30. 底牌中各花色分牌数: 4维 (obs[118:122])
    for color_idx, color in enumerate(['a', 'b', 'c', 'd']):
        sc_hole = sum(1 for c in hole_cards if c.color == color and c.has_score)
        obs[118 + color_idx] = sc_hole / 2.0
    
    # 31. 绝门花色数: 1维 (obs[122])
    jue_count = sum(1 for color in ['a', 'b', 'c', 'd']
                    if sum(1 for c in player.cards_in_hand.get(color, [])
                           if not c.is_zhu(now_level, now_color)) == 0)
    obs[122] = jue_count / 4.0
    
    # 32. 绝门花色中底牌分布: 1维 (obs[123])
    jue_hole = sum(1 for c in hole_cards 
                   for color in ['a', 'b', 'c', 'd']
                   if sum(1 for pc in player.cards_in_hand.get(color, [])
                          if not pc.is_zhu(now_level, now_color)) == 0
                   and c.color == color)
    obs[123] = jue_hole / 4.0
    
    # 33. 建议扣分力度: 1维 (obs[124])
    # 基于守底能力：强→敢扣分(1.0)，弱→不扣分(0.0)
    # 给网络一个先验建议，但网络可以override
    score_suggestion = min(defend_ability * 1.5, 1.0)  # 守底能力映射到扣分建议
    obs[124] = score_suggestion
    
    # 34. 建议扣分牌数: 1维 (obs[125])
    # 根据守底能力推荐扣几张分牌 (0~4)
    # 强→可扣3-4张分, 中→扣1-2张, 弱→0张
    if defend_ability > 0.6:
        suggested_score_cards = min(4, sum(1 for c in hole_cards if c.has_score))
    elif defend_ability > 0.35:
        suggested_score_cards = min(2, sum(1 for c in hole_cards if c.has_score))
    else:
        suggested_score_cards = 0
    obs[125] = suggested_score_cards / 4.0
    
    # 35. 主牌控制力: 1维 (obs[126])
    # 主对+主连对数量 → 控制出牌权的能力
    zhu_control = (len(zhudui) // 2) / 4.0 * 0.5 + len(zhuliandui) / 3.0 * 0.5
    obs[126] = zhu_control
    
    # 36. 主牌长度(最长连对): 1维 (obs[127])
    # 主连对越长，控制力越强
    max_zhu_lian = 0
    for lian in zhuliandui:
        l = len(lian)
        if l > max_zhu_lian:
            max_zhu_lian = l
    obs[127] = max_zhu_lian / 12.0
    
    # 37. padding: 2维 (obs[128:130])
    
    return obs


# ============================================================
# 动作候选生成器
# ============================================================

def get_lz_actions(player, now_level):
    """亮主阶段合法动作列表
    
    Returns:
        list of (action_id, action_info)
        action_info = {
            'type': 'pass' | 'duolian' | 'shuanglian' | 'danlian' | 'wanglian',
            'cards': list[Card],  # 亮出的牌
            'color': str,         # 主花色
        }
    """
    from server.ai import get_dui_and_liandui
    from server.card import Card
    
    actions = []
    
    # action 0: 不亮
    actions.append((0, {'type': 'pass', 'cards': [], 'color': None}))
    
    dan_dict, kings, dui_dict, liandui_dict = get_dui_and_liandui(
        player.cards_in_hand)
    
    action_id = 1
    
    # 多连对(≥6张, 无需王) → 最强亮牌
    for color, chains in liandui_dict.items():
        for chain in chains:
            if len(chain) >= 6:
                fu_cards = [c for c in chain if not c.is_zhu(now_level, None)]
                if fu_cards:
                    actions.append((action_id, {
                        'type': 'duolian',
                        'cards': fu_cards[:6],
                        'color': color,
                    }))
                    action_id += 1
    
    # 双连对(4张+1王)
    if len(kings) >= 1:
        for color, chains in liandui_dict.items():
            for chain in chains:
                if len(chain) >= 4:
                    fu_cards = [c for c in chain if not c.is_zhu(now_level, None)]
                    if fu_cards:
                        actions.append((action_id, {
                            'type': 'shuanglian',
                            'cards': fu_cards[:4] + [kings[0]],
                            'color': color,
                        }))
                        action_id += 1
    
    # 单连对(2张+1王) 或 单对(2张+1王)
    if len(kings) >= 1:
        for color, chains in liandui_dict.items():
            for chain in chains:
                fu_cards = [c for c in chain if not c.is_zhu(now_level, None)]
                if fu_cards and len(fu_cards) >= 2:
                    actions.append((action_id, {
                        'type': 'danlian',
                        'cards': fu_cards[:2] + [kings[0]],
                        'color': color,
                    }))
                    action_id += 1
        
        for color, cards in dui_dict.items():
            for i in range(0, len(cards), 2):
                if i + 1 < len(cards) and cards[i].name == cards[i + 1].name:
                    if not cards[i].is_zhu(now_level, None):
                        actions.append((action_id, {
                            'type': 'danlian',
                            'cards': [cards[i], cards[i + 1], kings[0]],
                            'color': color,
                        }))
                        action_id += 1
    
    # 三王(2王+对子定花色) — 罕见
    if len(kings) >= 2:
        for color, cards in dui_dict.items():
            for i in range(0, len(cards), 2):
                if i + 1 < len(cards) and cards[i].name == cards[i + 1].name:
                    if not cards[i].is_zhu(now_level, None):
                        actions.append((action_id, {
                            'type': 'wanglian',
                            'cards': [cards[i], cards[i + 1], kings[0], kings[1]],
                            'color': color,
                        }))
                        action_id += 1
    
    return actions


def get_cp_actions(room, player, now_level, now_color):
    """吃牌阶段合法动作列表
    
    使用GameRoom的_find_chipai_liangzhu_candidates获取准确候选。
    
    Returns:
        list of (action_id, action_info)
        action_info = {
            'type': 'pass' | 'claim',
            'cards': list[Card],
            'color': str,
            'claim_type': str,
        }
    """
    actions = []
    
    # action 0: 不吃
    actions.append((0, {'type': 'pass', 'cards': [], 'color': None, 'claim_type': 'pass'}))
    
    # 使用引擎的候选生成
    candidates = room._find_chipai_liangzhu_candidates(player)
    
    action_id = 1
    for cand in candidates:
        actions.append((action_id, {
            'type': 'claim',
            'cards': cand['cards'],
            'color': cand.get('color'),
            'claim_type': cand.get('type', ''),
        }))
        action_id += 1
    
    return actions


def get_kp_candidates(player, now_level, now_color, hole_cards, K=16):
    """扣底阶段候选方案列表
    
    生成K个扣牌方案供DMC选择，覆盖多种策略。
    
    Returns:
        list of (action_id, action_info)
        action_info = {
            'type': 'koupai',
            'cards': list[Card],  # 要扣的4张牌
            'strategy': str,      # 策略名 (用于debug)
        }
    """
    from server.card import Card
    from server.constants import SCORE_CARDS
    
    candidates = []
    seen_card_sets = set()  # 去重
    
    def _add_candidate(cards, strategy):
        card_key = tuple(sorted(c.card_type for c in cards))
        if card_key not in seen_card_sets and len(cards) == 4:
            seen_card_sets.add(card_key)
            candidates.append((len(candidates), {
                'type': 'koupai',
                'cards': cards,
                'strategy': strategy,
            }))
    
    # 所有非主牌
    fu_cards = []
    for color in ['a', 'b', 'c', 'd']:
        for c in player.cards_in_hand.get(color, []):
            if not c.is_zhu(now_level, now_color):
                fu_cards.append(c)
    fu_cards.sort(key=lambda c: (c.has_score, -c.rank))  # 弱牌优先
    
    # 所有手牌（扣底时picker已捡起底牌，cards_in_hand已包含）
    all_cards = []
    for color_cards in player.cards_in_hand.values():
        all_cards.extend(color_cards)
    
    # 策略1: 绝门优先 — 每个花色如果有≤4张非主牌，全扣
    for color in ['a', 'b', 'c', 'd']:
        color_fu = [c for c in player.cards_in_hand.get(color, [])
                    if not c.is_zhu(now_level, now_color)]
        if 0 < len(color_fu) <= 4:
            kou = list(color_fu[:4])
            if len(kou) < 4:
                # 从其他弱副牌补齐
                remaining = [c for c in fu_cards 
                           if c.color != color and c not in kou]
                kou.extend(remaining[:4 - len(kou)])
            if len(kou) == 4:
                _add_candidate(kou, f'void_{color}')
    
    # 策略2: 部分绝门 — 花色5~6张非主牌，扣最弱的4张
    for color in ['a', 'b', 'c', 'd']:
        color_fu = sorted(
            [c for c in player.cards_in_hand.get(color, [])
             if not c.is_zhu(now_level, now_color)],
            key=lambda c: (c.has_score, -c.rank)  # 弱牌优先扣
        )
        if 5 <= len(color_fu) <= 6:
            kou = list(color_fu[:4])
            _add_candidate(kou, f'partial_void_{color}')
    
    # 策略3: 藏分 — 扣无分弱牌（不扣5/10/K）
    no_score_fu = [c for c in fu_cards if not c.has_score]
    if len(no_score_fu) >= 4:
        _add_candidate(no_score_fu[:4], 'dump_noscore')
    
    # 策略4: 规则AI默认 — 调用 AI.decide_koupai
    try:
        from server.ai import AI
        ai = AI(player, now_level, now_color, 0)
        default_cards = ai.decide_koupai(hole_cards)
        if default_cards and len(default_cards) == 4:
            _add_candidate(default_cards, 'rule_ai')
    except Exception:
        pass
    
    # 策略5~K: 随机合法扣牌 — 保证多样性
    random.seed()  # 真随机
    attempts = 0
    while len(candidates) < K and attempts < 100:
        attempts += 1
        # 随机选4张非主牌（尽量）
        pool = list(fu_cards) if len(fu_cards) >= 4 else list(all_cards)
        if len(pool) < 4:
            break
        selected = random.sample(pool, min(4, len(pool)))
        if len(selected) == 4:
            _add_candidate(selected, f'random_{attempts}')
    
    # 截断到K个
    return candidates[:K]


# ============================================================
# Q-Network (小MLP)
# ============================================================

class SmallQNetwork(nn.Module):
    """小Q网络，用于亮主/吃牌/扣底决策"""
    
    def __init__(self, obs_dim, hidden_dim=128, max_actions=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_actions),  # Q(s, a) for all actions
        )
    
    def forward(self, obs):
        return self.net(obs)


# ============================================================
# Replay Buffer
# ============================================================

class PreplayBuffer:
    """独立replay buffer，用于单个pre-play网络"""
    
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
    
    def push(self, obs, action_idx, reward, next_obs, done, valid_action_count):
        entry = (obs, action_idx, reward, next_obs, done, valid_action_count)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones, valid_counts = zip(*batch)
        return (np.array(obs), np.array(actions), np.array(rewards),
                np.array(next_obs), np.array(dones), np.array(valid_counts))
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# Pre-play Trainer (单个网络)
# ============================================================

class PreplayTrainer:
    """独立训练单个pre-play网络"""
    
    def __init__(self, obs_dim, name, device='cpu',
                 hidden_dim=128, max_actions=20,
                 lr=1e-3, gamma=1.0, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=50000,
                 buffer_capacity=50000, batch_size=64):
        self.name = name
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.max_actions = max_actions
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.q_net = SmallQNetwork(obs_dim, hidden_dim, max_actions).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.buffer = PreplayBuffer(buffer_capacity)
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0
    
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               max(0, 1 - self.total_steps / self.epsilon_decay)
    
    def select_action(self, obs, valid_action_count):
        """epsilon-greedy 选择动作"""
        self.total_steps += 1  # 每次决策都计数，保证epsilon正确衰减
        eps = self.epsilon()
        if random.random() < eps:
            return random.randint(0, valid_action_count - 1)
        
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_vals = self.q_net(obs_t)[0, :valid_action_count]
            return q_vals.argmax().item()
    
    def store_transition(self, obs, action_idx, reward, next_obs, done, valid_action_count):
        self.buffer.push(obs, action_idx, reward, next_obs, done, valid_action_count)
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        obs, actions, rewards, next_obs, dones, valid_counts = \
            self.buffer.sample(self.batch_size)
        
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        q_all = self.q_net(obs_t)
        q_values = q_all.gather(1, actions_t).squeeze(1)
        
        # Target Q values (using target network)
        with torch.no_grad():
            # 对每个样本，只取valid_action_count个action的max
            next_q_all = self.target_net(next_obs_t)
            next_q_max = []
            for i in range(len(valid_counts)):
                n = int(valid_counts[i])
                if n > 0:
                    next_q_max.append(next_q_all[i, :n].max().item())
                else:
                    next_q_max.append(0.0)
            next_q_max = torch.FloatTensor(next_q_max).to(self.device)
            target_q = rewards_t + self.gamma * (1 - dones_t) * next_q_max
        
        loss = nn.MSELoss()(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        if self.total_steps % 100 == 0:
            for target_param, param in zip(self.target_net.parameters(),
                                            self.q_net.parameters()):
                target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt['total_steps']


# ============================================================
# 中间奖励计算
# ============================================================

# 奖励权重
W_EATEN = 1.0          # 被吃惩罚
W_EAT_SUCCESS = 1.0    # 吃成功奖励
W_EAT_FAIL = 1.0       # 吃失败惩罚
W_PICK_TRUMP = 1.0     # 底牌捡主奖励
W_HQ_NORM = 0.02       # 手牌质量变化归一化系数 (norm~50, 0.02*50≈1.0)
W_TEAMMATE_PICK = 1.0  # 队友捡主
W_OPPONENT_PICK = 1.0  # 对手捡主


def compute_lz_reward(player, now_level, now_color,
                      hq_before, hq_after,
                      was_eaten, picked_trump_from_bottom):
    """亮主中间奖励"""
    r = 0.0
    if was_eaten:
        r -= W_EATEN
    if picked_trump_from_bottom:
        r += W_PICK_TRUMP
    r += W_HQ_NORM * (hq_after - hq_before)
    return r


def compute_cp_reward(player, now_level, now_color,
                      hq_before, hq_after,
                      eat_result, picked_trump_from_bottom):
    """吃牌中间奖励
    
    eat_result: 'success' | 'fail' | 'pass'
    """
    r = 0.0
    if eat_result == 'success':
        r += W_EAT_SUCCESS
    elif eat_result == 'fail':
        r -= W_EAT_FAIL
    # 'pass' → 0
    if picked_trump_from_bottom:
        r += W_PICK_TRUMP
    r += W_HQ_NORM * (hq_after - hq_before)
    return r


def compute_kp_reward(player, now_level, now_color,
                      hq_before, hq_after,
                      koupai_cards, liangzhu_player_id, player_id,
                      huanpai_offer):
    """扣底中间奖励"""
    r = 0.0
    
    # 手牌质量变化
    r += W_HQ_NORM * (hq_after - hq_before)
    
    # 换牌影响
    if huanpai_offer:
        # 扣牌中有主花色牌，被亮主者捡走
        if liangzhu_player_id == player_id:
            # 自己是亮主者，自己捡 → 正面
            pass  # 已在hq变化中体现
        elif (liangzhu_player_id % 2) == (player_id % 2):
            # 队友捡 → 正面
            r += W_TEAMMATE_PICK
        else:
            # 对手捡 → 负面
            r -= W_OPPONENT_PICK
    
    return r
