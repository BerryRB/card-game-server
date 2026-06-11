#!/usr/bin/env python3
"""
Pre-play 网络独立训练脚本

用 规则AI vs 规则AI 对局收集亮主/吃牌/扣底数据，
训练三个独立小网络。与主出牌网络完全解耦。

用法:
  python -u rl_dmc/preplay/train.py --device cuda --num_episodes 50000
"""

import sys
import os
import argparse
import time
import logging
import numpy as np
import torch

# 确保项目根目录在path中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rl_dmc.preplay.networks import (
    PreplayTrainer,
    compute_hand_quality,
    encode_lz_obs, encode_cp_obs, encode_kp_obs,
    get_lz_actions, get_cp_actions, get_kp_candidates,
    compute_lz_reward, compute_cp_reward, compute_kp_reward,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


def _auto_play_one_phase(room):
    """只推进一个阶段，返回当前阶段名
    
    处理所有robot在该阶段的动作，但不推进到下一阶段。
    """
    from server.constants import GamePhase
    current_phase = room.phase
    # 调用 auto_play_current_robot 会处理所有robot
    # 但可能一次推进多个阶段，所以需要限制
    max_steps = 20
    for _ in range(max_steps):
        if room.phase != current_phase:
            break
        if room.phase in (GamePhase.PLAYING, GamePhase.GAME_OVER):
            break
        actions = room.auto_play_current_robot()
        if not actions:
            break
    return room.phase


def _auto_play_to_phase(room, target_phase):
    """推进到指定阶段"""
    from server.constants import GamePhase
    max_steps = 50
    for _ in range(max_steps):
        if room.phase == target_phase or room.phase in (GamePhase.PLAYING, GamePhase.GAME_OVER):
            break
        room.auto_play_current_robot()


def collect_preplay_data(room, dmc_team, trainers):
    """从一局游戏中收集亮主/吃牌/扣底的训练数据
    
    逐步推进游戏阶段，在每个阶段给DMC玩家决策机会。
    
    Args:
        room: GameRoom 实例 (已start_game, 停在LIANGZHU阶段)
        dmc_team: DMC控制的玩家ID列表 [0,2] or [1,3]
        trainers: dict {'lz': PreplayTrainer, 'cp': PreplayTrainer, 'kp': PreplayTrainer}
    
    Returns:
        dict: {'lz_count': int, 'cp_count': int, 'kp_count': int}
    """
    from server.constants import GamePhase
    
    counts = {'lz': 0, 'cp': 0, 'kp': 0}
    hq_before = {}  # player_id → hand_quality before decision
    hq_at_playing = {}  # player_id → hand_quality at PLAYING start
    pending_transitions = []
    
    # ======== 亮主阶段 ========
    if room.phase == GamePhase.LIANGZHU:
        # DMC玩家先行动：直接调用handle_liangzhu
        for pid in dmc_team:
            if room.phase != GamePhase.LIANGZHU:
                break
            player = room.players[pid]
            obs = encode_lz_obs(player, room.now_level)
            actions = get_lz_actions(player, room.now_level)
            
            if len(actions) <= 1:  # 只有"不亮"
                pending_transitions.append(('lz', obs, 0, pid, {
                    'actions': actions,
                }))
                counts['lz'] += 1
                continue
            
            hq_before[pid] = compute_hand_quality(player, room.now_level, None)
            
            action_idx = trainers['lz'].select_action(obs, len(actions))
            action_info = actions[action_idx][1]
            
            if action_info['type'] != 'pass':
                card_strs = [c.card_type for c in action_info['cards']]
                selected_color = action_info.get('color')
                result = room.handle_liangzhu(pid, card_strs, selected_color)
                if result['status'] != 'ok':
                    # 亮主失败（比如队友已亮），退回"不亮"
                    action_idx = 0
            
            pending_transitions.append(('lz', obs, action_idx, pid, {
                'actions': actions,
            }))
            counts['lz'] += 1
        
        # 规则AI完成剩余亮主+吃牌
        while room.phase not in (GamePhase.PLAYING, GamePhase.GAME_OVER):
            room.auto_play_current_robot()
            # 检查是否到达吃牌阶段
            if room.phase == GamePhase.CHIPAI:
                break
    
    # ======== 吃牌阶段 ========
    if room.phase == GamePhase.CHIPAI:
        liang_team = room.liangzhu_player % 2 if room.liangzhu_player is not None else -1
        
        for pid in dmc_team:
            if room.phase != GamePhase.CHIPAI:
                break
            if pid % 2 == liang_team:
                continue  # 同队不吃
            
            player = room.players[pid]
            liangzhu_type = getattr(room, 'liangzhu_type', None)
            liangzhu_color = room.now_color
            liangzhu_player = room.liangzhu_player
            
            obs = encode_cp_obs(player, room.now_level, room.now_color,
                               liangzhu_player, liangzhu_type, liangzhu_color, pid)
            
            actions = get_cp_actions(room, player, room.now_level, room.now_color)
            claim_count = sum(1 for a in actions if a[1]['type'] == 'claim')
            obs[74] = claim_count / 4.0
            
            hq_before_cp = compute_hand_quality(player, room.now_level, room.now_color)
            hq_before[pid] = hq_before_cp
            
            action_idx = trainers['cp'].select_action(obs, len(actions))
            action_info = actions[action_idx][1]
            
            if action_info['type'] == 'pass':
                room.handle_chipai_pass(pid)
            else:
                # 吃牌：先提交亮牌，再claim
                # action_info['cards'] 已经是 card_str 列表
                card_strs = action_info['cards']
                selected_color = action_info.get('color')
                result = room.handle_liangzhu(pid, card_strs, selected_color)
                if result['status'] == 'ok':
                    room.handle_chipai_claim(pid)
                else:
                    # 亮牌失败，当做pass
                    room.handle_chipai_pass(pid)
            
            pending_transitions.append(('cp', obs, action_idx, pid, {
                'actions': actions,
            }))
            counts['cp'] += 1
        
        # 规则AI完成剩余吃牌+扣底
        while room.phase not in (GamePhase.PLAYING, GamePhase.GAME_OVER):
            room.auto_play_current_robot()
            if room.phase == GamePhase.KOUPAI:
                break
    
    # ======== 扣底阶段 ========
    if room.phase == GamePhase.KOUPAI:
        picker = room._get_koupai_picker()
        if picker in dmc_team:
            player = room.players[picker]
            liangzhu_player = room.liangzhu_player
            
            # 收集吃牌阶段结果，推断队友主牌强度
            chipai_result = getattr(room, 'chipai_result', None)
            teammate_chipai_success = False
            was_eaten_by_opponent = False
            teammate_zhu_strength = 0.5  # 默认中性
            
            if chipai_result:
                big_seat = chipai_result.get('big_seat')
                small_seat = chipai_result.get('small_seat')
                # 队友吃牌成功
                if big_seat is not None and big_seat % 2 == picker % 2:
                    teammate_chipai_success = True
                    teammate_zhu_strength = 0.8  # 队友吃牌成功=主牌强
                # 我方被对手吃了
                if small_seat is not None and small_seat % 2 == picker % 2:
                    was_eaten_by_opponent = True
                    teammate_zhu_strength = 0.3  # 被吃=主牌弱
                # 对手吃牌成功且不是我方被吃→队友可能中等
                if big_seat is not None and big_seat % 2 != picker % 2 and not was_eaten_by_opponent:
                    teammate_zhu_strength = 0.4  # 对手吃了但不是吃我的
            
            obs = encode_kp_obs(player, room.now_level, room.now_color,
                               room.hole_cards, liangzhu_player, picker)
            
            # 填充吃牌博弈特征 obs[104:107]
            obs[104] = 1.0 if teammate_chipai_success else 0.0
            obs[105] = 1.0 if was_eaten_by_opponent else 0.0
            obs[106] = teammate_zhu_strength
            # 重新计算守底能力 (obs[107])
            obs[107] = obs[100] * 0.6 + teammate_zhu_strength * 0.4
            # 重新计算扣分建议 (obs[124])
            obs[124] = min(obs[107] * 1.5, 1.0)
            # 重新计算建议扣分牌数 (obs[125])
            if obs[107] > 0.6:
                suggested = min(4, sum(1 for c in room.hole_cards if c.has_score))
            elif obs[107] > 0.35:
                suggested = min(2, sum(1 for c in room.hole_cards if c.has_score))
            else:
                suggested = 0
            obs[125] = suggested / 4.0
            
            candidates = get_kp_candidates(player, room.now_level, room.now_color,
                                           room.hole_cards, K=16)
            
            hq_before_kp = compute_hand_quality(player, room.now_level, room.now_color)
            hq_before[picker] = hq_before_kp
            
            action_idx = trainers['kp'].select_action(obs, len(candidates))
            action_info = candidates[action_idx][1]
            
            card_strs = [c.card_type for c in action_info['cards']]
            result = room.handle_koupai(picker, card_strs)
            if result['status'] != 'ok':
                from server.ai import AI
                ai = AI(player, room.now_level, room.now_color, 0)
                kou_cards = ai.decide_koupai(room.hole_cards)
                card_strs = [c.card_type for c in kou_cards]
                room.handle_koupai(picker, card_strs)
            
            pending_transitions.append(('kp', obs, action_idx, picker, {
                'candidates': candidates,
                'koupai_cards': action_info['cards'],
            }))
            counts['kp'] += 1
        
        # 让规则AI完成扣底和换牌
        while room.phase not in (GamePhase.PLAYING, GamePhase.GAME_OVER):
            room.auto_play_current_robot()
    
    # ---- 计算中间奖励 ----
    # 在PLAYING开始时评估手牌质量
    if room.phase == GamePhase.PLAYING or room.phase == GamePhase.HUANPAI:
        # 先让换牌完成
        while room.phase == GamePhase.HUANPAI:
            room.auto_play_current_robot()
        
        if room.now_color:
            for pid in dmc_team:
                player = room.players[pid]
                hq_at_playing[pid] = compute_hand_quality(
                    player, room.now_level, room.now_color)
    
    # 处理pending transitions，计算reward并存储
    for trainer_name, obs, action_idx, pid, extra in pending_transitions:
        hq_after = hq_at_playing.get(pid, hq_before.get(pid, 0))
        hq_b = hq_before.get(pid, 0)
        
        if trainer_name == 'lz':
            # 亮主奖励
            was_eaten = False
            picked_trump = False
            
            # 检查是否被吃
            chipai_result = getattr(room, 'chipai_result', None)
            if chipai_result and chipai_result.get('small_seat') == pid:
                was_eaten = True
            
            # 检查是否从底牌捡到主花色
            huanpai_picked = getattr(room, 'huanpai_picked_cards', [])
            if huanpai_picked and pid == room.liangzhu_player:
                picked_trump = True
            
            reward = compute_lz_reward(
                room.players[pid], room.now_level, room.now_color,
                hq_b, hq_after, was_eaten, picked_trump)
            
            # next_obs: PLAYING开始时的obs（简化：用同一obs，因为是单步决策）
            next_obs = obs  # 单步决策，done=True
            trainers['lz'].store_transition(
                obs, action_idx, reward, next_obs, True, len(extra['actions']))
        
        elif trainer_name == 'cp':
            # 吃牌奖励
            eat_result = 'pass'
            picked_trump = False
            
            chipai_result = getattr(room, 'chipai_result', None)
            if chipai_result:
                if chipai_result.get('big_seat') == pid:
                    eat_result = 'success'
                elif chipai_result.get('small_seat') == pid:
                    eat_result = 'fail'
            
            huanpai_picked = getattr(room, 'huanpai_picked_cards', [])
            if huanpai_picked and pid == room.liangzhu_player:
                picked_trump = True
            
            reward = compute_cp_reward(
                room.players[pid], room.now_level, room.now_color,
                hq_b, hq_after, eat_result, picked_trump)
            
            next_obs = obs
            trainers['cp'].store_transition(
                obs, action_idx, reward, next_obs, True, len(extra['actions']))
        
        elif trainer_name == 'kp':
            # 扣底奖励
            koupai_cards = extra.get('koupai_cards', [])
            huanpai_offer = getattr(room, 'huanpai_offer', [])
            liangzhu_player = room.liangzhu_player
            
            reward = compute_kp_reward(
                room.players[pid], room.now_level, room.now_color,
                hq_b, hq_after,
                koupai_cards, liangzhu_player, pid,
                huanpai_offer)
            
            next_obs = obs
            trainers['kp'].store_transition(
                obs, action_idx, reward, next_obs, True, len(extra.get('candidates', [])))
    
    return counts


def main():
    parser = argparse.ArgumentParser(description='Train pre-play networks')
    parser.add_argument('--num_episodes', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train_interval', type=int, default=100,
                       help='每收集N局数据训练一次')
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--save_dir', type=str, default='rl_dmc/preplay/models')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化三个独立trainer
    trainers = {
        'lz': PreplayTrainer(
            obs_dim=80, name='liangzhu', device=args.device,
            hidden_dim=128, max_actions=10,
            lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay=60000, buffer_capacity=50000),
        'cp': PreplayTrainer(
            obs_dim=110, name='chipai', device=args.device,
            hidden_dim=128, max_actions=8,
            lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay=20000, buffer_capacity=30000),
        'kp': PreplayTrainer(
            obs_dim=130, name='koupai', device=args.device,
            hidden_dim=128, max_actions=16,
            lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay=5000, buffer_capacity=50000),
    }
    
    # 统计
    stats = {k: {'total': 0, 'reward_sum': 0.0, 'reward_count': 0}
             for k in ['lz', 'cp', 'kp']}
    
    log.info(f"开始训练 pre-play 网络: {args.num_episodes} episodes, device={args.device}")
    start_time = time.time()
    
    for ep in range(1, args.num_episodes + 1):
        # 直接用 GameRoom，停在 LIANGZHU 阶段
        from server.game_engine import GameRoom
        room = GameRoom(f'preplay_{ep}')
        room.start_game()
        
        # 交替DMC控制的队伍，最大化数据量
        dmc_team = [0, 2] if ep % 2 == 1 else [1, 3]
        
        try:
            counts = collect_preplay_data(room, dmc_team, trainers)
        except Exception as e:
            log.warning(f"Ep {ep}: 收集数据异常 {e}")
            continue
        
        for k in ['lz', 'cp', 'kp']:
            stats[k]['total'] += counts[k]
        
        # 定期训练
        if ep % args.train_interval == 0:
            for name, trainer in trainers.items():
                if len(trainer.buffer) >= trainer.batch_size:
                    losses = []
                    for _ in range(10):  # 每次训练10步
                        loss = trainer.train_step()
                        if loss is not None:
                            losses.append(loss)
                    if losses:
                        avg_loss = np.mean(losses)
                        log.info(f"  [{name}] loss={avg_loss:.4f}, "
                                f"buf={len(trainer.buffer)}, eps={trainer.epsilon():.3f}")
        
        # 定期日志
        if ep % 1000 == 0:
            elapsed = time.time() - start_time
            speed = ep / elapsed if elapsed > 0 else 0
            buf_info = ' | '.join(
                f"{k}={len(trainers[k].buffer)}" for k in ['lz', 'cp', 'kp'])
            log.info(f"Ep {ep}/{args.num_episodes} | "
                    f"data: {buf_info} | "
                    f"speed={speed:.1f}ep/s | "
                    f"time={elapsed:.0f}s")
        
        # 定期评估和保存
        if ep % args.eval_interval == 0:
            # 评估: 用训练好的网络 vs 规则AI 打100局
            # 统计: 亮主成功率、吃牌成功率、扣底质量
            eval_stats = evaluate_preplay(trainers, dmc_team, num_games=200)
            log.info(f"Eval @Ep {ep}: " +
                    f"LZ: 亮主率={eval_stats['lz_rate']:.1%} " +
                    f"被吃率={eval_stats['lz_eaten_rate']:.1%} " +
                    f"avg_r={eval_stats['lz_avg_r']:.3f} | " +
                    f"CP: 吃率={eval_stats['cp_rate']:.1%} " +
                    f"成功率={eval_stats['cp_success_rate']:.1%} " +
                    f"avg_r={eval_stats['cp_avg_r']:.3f} | " +
                    f"KP: avg_r={eval_stats['kp_avg_r']:.3f} " +
                    f"换牌被对手捡率={eval_stats['kp_opp_pick_rate']:.1%}")
            
            # 保存checkpoint
            for name, trainer in trainers.items():
                path = os.path.join(args.save_dir, f'{name}_ep{ep}.pt')
                trainer.save(path)
            log.info(f"  Saved checkpoints to {args.save_dir}")
    
    # 训练结束，保存最终模型
    for name, trainer in trainers.items():
        path = os.path.join(args.save_dir, f'{name}_final.pt')
        trainer.save(path)
    
    elapsed = time.time() - start_time
    log.info(f"训练完成! 总耗时 {elapsed:.0f}s")
    log.info(f"数据量: " +
            ' | '.join(f"{k}={stats[k]['total']}" for k in ['lz', 'cp', 'kp']))


def _auto_play_to_phase(room, target_phase):
    """推进game room直到到达目标阶段或更晚的阶段
    
    GamePhase顺序: DEALING → HUANPAI → LIANGZHU → CHIPAI → KOUPAI → PLAYING → GAME_OVER
    如果target_phase=CHIPAI，则推进到CHIPAI/KOUPAI/PLAYING/GAME_OVER都算完成。
    """
    from server.constants import GamePhase
    phase_order = [
        GamePhase.WAITING, GamePhase.HUANPAI, GamePhase.LIANGZHU,
        GamePhase.CHIPAI, GamePhase.KOUPAI, GamePhase.PLAYING,
        GamePhase.SCORING, GamePhase.GAME_OVER
    ]
    try:
        target_idx = phase_order.index(target_phase)
    except ValueError:
        target_idx = len(phase_order) - 1
    max_steps = 50
    for _ in range(max_steps):
        try:
            current_idx = phase_order.index(room.phase)
        except ValueError:
            break
        if current_idx >= target_idx:
            break
        room.auto_play_current_robot()


def evaluate_preplay(trainers, dmc_team, num_games=200):
    """评估pre-play网络的决策质量
    
    评估时不更新epsilon（用当前训练的epsilon值），
    统计亮主率/被吃率/吃牌成功率/扣底质量等。
    """
    from server.game_engine import GameRoom
    from server.constants import GamePhase
    
    lz_count = 0
    lz_eaten = 0
    lz_rewards = []
    cp_count = 0
    cp_success = 0
    cp_rewards = []
    kp_rewards = []
    kp_opp_pick = 0
    kp_total = 0
    
    for _ in range(num_games):
        room = GameRoom(f'eval_{_}')
        room.start_game()
        
        # ---- 亮主阶段 ----
        lz_hq_before = {}
        for pid in dmc_team:
            if room.phase != GamePhase.LIANGZHU:
                break
            player = room.players[pid]
            obs = encode_lz_obs(player, room.now_level)
            actions = get_lz_actions(player, room.now_level)
            if len(actions) <= 1:
                continue
            # 评估时用greedy选择（epsilon=0）
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainers['lz'].device)
                q_vals = trainers['lz'].q_net(obs_t)[0, :len(actions)]
                action_idx = q_vals.argmax().item()
            action_info = actions[action_idx][1]
            if action_info['type'] != 'pass':
                lz_count += 1
                lz_hq_before[pid] = compute_hand_quality(
                    player, room.now_level, None)
                card_strs = [c.card_type for c in action_info['cards']]
                result = room.handle_liangzhu(pid, card_strs, action_info.get('color'))
                if result['status'] != 'ok':
                    lz_count -= 1
        
        # 推进到下一阶段（完整推进，不用单次auto_play）
        _auto_play_to_phase(room, GamePhase.CHIPAI)
        
        # ---- 吃牌阶段 ----
        cp_hq_before = {}
        if room.phase == GamePhase.CHIPAI and room.liangzhu_player is not None:
            liang_team = room.liangzhu_player % 2
            for pid in dmc_team:
                if pid % 2 == liang_team or room.phase != GamePhase.CHIPAI:
                    continue
                player = room.players[pid]
                obs = encode_cp_obs(player, room.now_level, room.now_color,
                                   room.liangzhu_player, getattr(room, 'liangzhu_type', None),
                                   room.now_color, pid)
                actions = get_cp_actions(room, player, room.now_level, room.now_color)
                if len(actions) <= 1:
                    room.handle_chipai_pass(pid)
                    continue
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainers['cp'].device)
                    q_vals = trainers['cp'].q_net(obs_t)[0, :len(actions)]
                    action_idx = q_vals.argmax().item()
                action_info = actions[action_idx][1]
                cp_hq_before[pid] = compute_hand_quality(
                    player, room.now_level, room.now_color)
                if action_info['type'] == 'claim':
                    cp_count += 1
                    card_strs = action_info['cards']
                    selected_color = action_info.get('color')
                    lz_result = room.handle_liangzhu(pid, card_strs, selected_color)
                    if lz_result['status'] == 'ok':
                        result = room.handle_chipai_claim(pid)
                        if result.get('chipai') and result.get('big_seat') == pid:
                            cp_success += 1
                    else:
                        room.handle_chipai_pass(pid)
                else:
                    room.handle_chipai_pass(pid)
        
        _auto_play_to_phase(room, GamePhase.KOUPAI)
        
        # ---- 扣底阶段 ----
        if room.phase == GamePhase.KOUPAI:
            picker = room._get_koupai_picker()
            if picker in dmc_team:
                player = room.players[picker]
                
                # 收集吃牌结果推断队友主牌强度
                chipai_result = getattr(room, 'chipai_result', None)
                teammate_zhu_str = 0.5
                teammate_cp_ok = False
                was_eaten = False
                if chipai_result:
                    big_seat = chipai_result.get('big_seat')
                    small_seat = chipai_result.get('small_seat')
                    if big_seat is not None and big_seat % 2 == picker % 2:
                        teammate_cp_ok = True
                        teammate_zhu_str = 0.8
                    if small_seat is not None and small_seat % 2 == picker % 2:
                        was_eaten = True
                        teammate_zhu_str = 0.3
                    if big_seat is not None and big_seat % 2 != picker % 2 and not was_eaten:
                        teammate_zhu_str = 0.4
                
                obs = encode_kp_obs(player, room.now_level, room.now_color,
                                   room.hole_cards, room.liangzhu_player, picker)
                obs[104] = 1.0 if teammate_cp_ok else 0.0
                obs[105] = 1.0 if was_eaten else 0.0
                obs[106] = teammate_zhu_str
                obs[107] = obs[100] * 0.6 + teammate_zhu_str * 0.4
                obs[124] = min(obs[107] * 1.5, 1.0)
                if obs[107] > 0.6:
                    suggested = min(4, sum(1 for c in room.hole_cards if c.has_score))
                elif obs[107] > 0.35:
                    suggested = min(2, sum(1 for c in room.hole_cards if c.has_score))
                else:
                    suggested = 0
                obs[125] = suggested / 4.0
                
                candidates = get_kp_candidates(player, room.now_level, room.now_color,
                                              room.hole_cards, K=16)
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainers['kp'].device)
                    q_vals = trainers['kp'].q_net(obs_t)[0, :len(candidates)]
                    action_idx = q_vals.argmax().item()
                action_info = candidates[action_idx][1]
                
                # 扣底前手牌质量
                kp_hq_b = compute_hand_quality(player, room.now_level, room.now_color)
                
                card_strs = [c.card_type for c in action_info['cards']]
                result = room.handle_koupai(picker, card_strs)
                
                kp_total += 1
                # 扣底后手牌质量
                kp_hq_after = compute_hand_quality(player, room.now_level, room.now_color)
                huanpai_offer = getattr(room, 'huanpai_offer', [])
                kp_r = compute_kp_reward(
                    player, room.now_level, room.now_color,
                    kp_hq_b, kp_hq_after,
                    action_info['cards'], room.liangzhu_player, picker,
                    huanpai_offer)
                kp_rewards.append(kp_r)
                
                if huanpai_offer and room.liangzhu_player is not None:
                    if (room.liangzhu_player % 2) != (picker % 2):
                        kp_opp_pick += 1
        
        # 完成游戏到PLAYING阶段
        _auto_play_to_phase(room, GamePhase.PLAYING)
        
        # 计算LZ和CP的reward（需要PLAYING阶段的hq_after）
        chipai_result = getattr(room, 'chipai_result', None)
        for pid in dmc_team:
            if room.now_color:
                hq_after = compute_hand_quality(
                    room.players[pid], room.now_level, room.now_color)
            else:
                hq_after = 0
            
            # LZ reward
            if pid in lz_hq_before:
                was_eaten_flag = False
                if chipai_result and chipai_result.get('small_seat') == pid:
                    was_eaten_flag = True
                hq_b = lz_hq_before[pid]
                r = compute_lz_reward(
                    room.players[pid], room.now_level, room.now_color,
                    hq_b, hq_after, was_eaten_flag, False)
                lz_rewards.append(r)
                if was_eaten_flag:
                    lz_eaten += 1
            
            # CP reward
            if pid in cp_hq_before:
                eat_result = 'pass'
                if chipai_result:
                    if chipai_result.get('big_seat') == pid:
                        eat_result = 'success'
                    elif chipai_result.get('small_seat') == pid:
                        eat_result = 'fail'
                hq_b = cp_hq_before[pid]
                r = compute_cp_reward(
                    room.players[pid], room.now_level, room.now_color,
                    hq_b, hq_after, eat_result, False)
                cp_rewards.append(r)
    
    return {
        'lz_rate': lz_count / max(num_games, 1),
        'lz_eaten_rate': lz_eaten / max(lz_count, 1),
        'lz_avg_r': np.mean(lz_rewards) if lz_rewards else 0,
        'cp_rate': cp_count / max(num_games, 1),
        'cp_success_rate': cp_success / max(cp_count, 1),
        'cp_avg_r': np.mean(cp_rewards) if cp_rewards else 0,
        'kp_avg_r': np.mean(kp_rewards) if kp_rewards else 0,
        'kp_opp_pick_rate': kp_opp_pick / max(kp_total, 1),
    }


if __name__ == '__main__':
    main()
