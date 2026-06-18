"""
升级(Trump/Shengji)游戏启发式AI v6

基于完整游戏规则的策略AI，支持先手/跟牌/残局策略。
作为公共模块，供训练代码和后端服务共同引用。

策略概述：
- 先手：清副牌连对>对子>单张，闲家争赢墩(不出分牌)，庄家控场
- 跟牌-队友赢：闲家贴分跑分，庄家让路
- 跟牌-对手赢：最小代价压住，闲家优先用分牌压
- 跟牌-压不住：队友未出可贴分(队友可能压住)，否则出小止损
- 残局(≤3张)：全力压不省牌
"""

import numpy as np


def get_heuristic_action(game, pid, bankers, epsilon=0.0):
    """启发式AI v6：基于升级游戏规则的完整策略AI

    Args:
        game: ShengjiGame实例
        pid: 当前玩家ID (0-3)
        bankers: 庄家ID列表
        epsilon: 随机探索概率(0=纯策略, >0=有概率随机出牌)

    Returns:
        int: legal_actions列表中的动作索引
    """
    legal = game.get_legal_actions(pid)
    if not legal:
        return 0
    if len(legal) == 1:
        return 0

    if epsilon > 0 and np.random.random() < epsilon:
        return np.random.randint(len(legal))

    from server.card import Card
    from server.rules import compare_outcards, get_zhu_rank
    from rl_shengji.env import _ID_TO_CARD_TYPE

    room = game.room
    epoch_players = room.epoch_players
    epoch_cards = room.epoch_cards  # Card对象列表，与epoch_players一一对应
    n_played = len(epoch_players)
    is_first = (n_played == 0)
    is_banker = pid in bankers

    now_level = str(room.now_level)
    now_color = room.now_color
    partner_pid = (pid + 2) % 4

    # 手牌数量（用于残局判断）
    hand_size = room.players[pid].card_count

    action_cards = []
    for i, action in enumerate(legal):
        card_ids = action[1]
        cards = [Card(_ID_TO_CARD_TYPE[cid]) for cid in card_ids]
        action_cards.append((i, cards))

    def is_trump_card(card):
        return card.is_zhu(now_level, now_color)

    def card_strength(card):
        """牌力值：主牌0-50，副牌100+（花色×15+rank）"""
        if is_trump_card(card):
            return get_zhu_rank(card, now_level, now_color)
        else:
            suit_order = {'s': 0, 'h': 1, 'c': 2, 'd': 3}
            return 100 + suit_order.get(card.color, 0) * 15 + int(card.rank)

    def card_point_value(card):
        """分值牌：5=5分, 10=10分, K=10分，其他0分"""
        r = int(card.rank)
        if r == 5: return 5
        if r == 10 or r == 13: return 10
        return 0

    def action_point_value(cards):
        """一组牌的总分值"""
        return sum(card_point_value(c) for c in cards)

    def get_play_type(cards):
        if len(cards) == 1:
            return 'single'
        names = set(c.name for c in cards)
        if len(cards) == 2 and len(names) == 1:
            return 'pair'
        if len(cards) >= 4 and len(names) == 1:
            return 'liandui'
        if len(cards) >= 4:
            ranks = sorted(set(int(c.rank) for c in cards))
            if all(ranks[i+1] == ranks[i]+1 for i in range(len(ranks)-1)) and len(cards) == len(ranks) * 2:
                return 'liandui'
        return 'combo'

    def action_avg_strength(cards):
        """动作的平均牌力（用于排序）"""
        return sum(card_strength(c) for c in cards) / len(cards)

    # ─── 先手出牌策略 ───
    if is_first:
        non_trump_pairs = []
        non_trump_liandui = []
        non_trump_singles = []
        trump_actions = []

        for idx, cards in action_cards:
            pt = get_play_type(cards)
            has_trump = any(is_trump_card(c) for c in cards)
            if has_trump:
                trump_actions.append((idx, cards, pt))
            elif pt == 'pair':
                non_trump_pairs.append((idx, cards))
            elif pt in ('liandui',):
                non_trump_liandui.append((idx, cards))
            elif pt == 'single':
                non_trump_singles.append((idx, cards))

        # ① 优先清副牌连对（大的先出，不容易被压）
        if non_trump_liandui:
            non_trump_liandui.sort(key=lambda x: action_avg_strength(x[1]), reverse=True)
            return non_trump_liandui[0][0]

        # ② 清副牌对子（大的先出）
        if non_trump_pairs:
            non_trump_pairs.sort(key=lambda x: action_avg_strength(x[1]), reverse=True)
            return non_trump_pairs[0][0]

        # ③ 副牌单张
        if non_trump_singles:
            if not is_banker:
                # 闲家：优先出最大的非分值牌争赢墩（先手出分牌=送分）
                non_score_actions = [(idx, cards) for idx, cards in non_trump_singles
                                     if action_point_value(cards) == 0]
                if non_score_actions:
                    # 出最大的非分值牌争赢
                    return max(non_score_actions, key=lambda x: card_strength(x[1][0]))[0]
                # 只有分值牌了 → 出最大的（K>10>5，K大概率能赢）
                return max(non_trump_singles, key=lambda x: card_strength(x[1][0]))[0]
            else:
                # 庄家：出最小牌控场（保留大牌压对手）
                return min(non_trump_singles, key=lambda x: card_strength(x[1][0]))[0]

        # ④ 只有主牌了 → 出最小主牌
        if trump_actions:
            return min(trump_actions, key=lambda x: action_avg_strength(x[1]))[0]

        return 0

    # ─── 跟牌策略 ───
    else:
        # ── 判断队友是否在赢 ──
        partner_winning = False
        if n_played >= 2 and partner_pid in epoch_players:
            try:
                partner_idx_in_epoch = epoch_players.index(partner_pid)
                partner_card_objs = epoch_cards[partner_idx_in_epoch]

                # 比较当前所有已出牌，看partner是否最强
                partner_is_best = True
                for j, ep_pid in enumerate(epoch_players):
                    if ep_pid == partner_pid:
                        continue
                    try:
                        cmp_result = compare_outcards(epoch_cards[j], partner_card_objs, now_level, now_color)
                        if cmp_result > 0:  # other > partner
                            partner_is_best = False
                            break
                    except:
                        pass

                partner_winning = partner_is_best
            except Exception:
                # fallback: partner先出且唯一出牌者
                partner_winning = (n_played == 1 and epoch_players[0] == partner_pid)
        elif n_played == 1 and epoch_players[0] == partner_pid:
            partner_winning = True

        # ── 队友在赢 → 让路策略 ──
        if partner_winning:
            # 闲家：如果有分值牌且队友在赢，贴分值牌给队友跑分
            if not is_banker:
                score_actions = [(idx, cards) for idx, cards in action_cards
                                 if action_point_value(cards) > 0]
                if score_actions:
                    # 贴最小的分值牌跑分
                    return min(score_actions, key=lambda x: card_strength(x[1][0]))[0]
            # 庄家或无分值牌 → 出最小牌让路
            return min(action_cards, key=lambda x: card_strength(x[1][0]))[0]

        # ── 对手在赢 → 压牌策略 ──
        # 找当前轮次最强的已出牌
        best_played_cards = None
        try:
            for j, ep_pid in enumerate(epoch_players):
                other_card_objs = epoch_cards[j]
                if best_played_cards is None:
                    best_played_cards = other_card_objs
                else:
                    try:
                        if compare_outcards(other_card_objs, best_played_cards, now_level, now_color) > 0:
                            best_played_cards = other_card_objs
                    except:
                        pass
        except Exception:
            pass

        # 找legal中能压住的动作
        can_beat = []
        can_beat_with_score = []  # 带分值的压牌（闲家优先）
        for idx, cards in action_cards:
            if best_played_cards is not None:
                try:
                    cmp = compare_outcards(cards, best_played_cards, now_level, now_color)
                    if cmp > 0:  # 能压住
                        beat_cost = max(card_strength(c) for c in cards)
                        pts = action_point_value(cards)
                        can_beat.append((idx, beat_cost))
                        if pts > 0:
                            can_beat_with_score.append((idx, beat_cost, pts))
                except:
                    pass

        if can_beat:
            # 残局策略：手牌≤3时全力压（不省牌）
            if hand_size <= 3:
                return max(can_beat, key=lambda x: x[1])[0]  # 出最大确保赢

            # 闲家优先用带分值的牌压（跑分+赢墩）
            if not is_banker and can_beat_with_score:
                # 选代价最低的带分压牌
                return min(can_beat_with_score, key=lambda x: x[1])[0]

            # 用最小代价压住
            return min(can_beat, key=lambda x: x[1])[0]

        # ── 压不住 → 止损策略 ──
        # 闲家：如果队友还没出牌，贴最小分值牌（队友可能压住跑分）
        # 如果队友已出过且对手在赢，贴分=白送，出最小牌止损
        if not is_banker and partner_pid not in epoch_players:
            score_actions = [(idx, cards) for idx, cards in action_cards
                             if action_point_value(cards) > 0]
            if score_actions:
                return min(score_actions, key=lambda x: card_strength(x[1][0]))[0]

        # 庄家或对手已赢→出最小牌止损
        return min(action_cards, key=lambda x: card_strength(x[1][0]))[0]
