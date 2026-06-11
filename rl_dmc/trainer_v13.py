# -*- coding: utf-8 -*-
"""
DMC V13 训练器 - Actor-Critic + vs规则AI

核心改进vs V10/V11/V12：
1. Actor-Critic架构（Critic提供baseline V(s)，减少方差）
2. Advantage A(s,a) = G_t - V(s) 实现信用分配
3. 保留Bug1-4修复（epoch_history, per-step reward, next_obs, done）
4. vs规则AI训练（对手固定，信号稳定）

V10/V11/V12的DQN天花板~43%，根因：
- 纯MC return无信用分配 → Q(s,a)≈payoff均值，无法排序
- Q(s,零action)估计V(s')是噪声 → TD target无效
- 需要Critic提供baseline，advantage加权policy gradient
"""

import sys
import os
import json
import time
import random
import argparse
import logging
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_dmc.actor_critic import ActorCriticSet, CriticNetwork
from rl_dmc.q_network_v2 import QNetworkV2
from rl_dmc.obs_v11 import encode_obs_v11, OBS_DIM_V11
from rl_shengji.env import ShengjiGame, NUM_CARDS, cards_to_ids, cards_to_onehot

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class _ObsWrapper:
    """为encode_obs_v11提供兼容接口"""
    def __init__(self, game, initial_bankers):
        self._game = game
        self.room = game.room
        self.initial_bankers = initial_bankers
        self.played_cards = game.room.played_cards if hasattr(game.room, 'played_cards') else []
        self.epoch_history = getattr(game.room, 'epoch_history', [])
        self.huanpai_info = getattr(game.room, 'huanpai_info', {})
        self.liangzhu_info = getattr(game.room, 'liangzhu_info', {})
        self.koupai_info = getattr(game.room, 'koupai_info', {})
        self.players = game.room.players
        self.now_level = game.room.now_level
        self.now_color = game.room.now_color
        self.score_now = getattr(game.room, 'score_now', 0)
        self.score_koupai = getattr(game.room, 'score_koupai', 0)
        self.bankers = list(game.room.bankers) if game.room.bankers else [0, 2]

    def get_state(self, player_id):
        return self._game.get_state(player_id)


class DMCTrainerV13:
    """Actor-Critic DMC训练器"""
    
    def __init__(self, lr=1e-4, epsilon_start=0.5, epsilon_end=0.05,
                 epsilon_decay=0.99995, buffer_capacity=100000,
                 batch_size=256, hidden_dim=512, gamma=0.99,
                 train_steps_per_episode=4, device='cuda'):
        self.device = device
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.train_steps = train_steps_per_episode
        self.obs_dim = OBS_DIM_V11
        
        # Actor-Critic网络
        self.ac_set = ActorCriticSet(
            obs_dim=self.obs_dim,
            num_cards=NUM_CARDS,
            hidden_dim=hidden_dim,
            device=device,
        )
        
        # 优化器：Actor和Critic分别优化
        self.banker_actor_opt = optim.Adam(self.ac_set.banker_actor.parameters(), lr=lr)
        self.banker_critic_opt = optim.Adam(self.ac_set.banker_critic.parameters(), lr=lr)
        self.xianjia_actor_opt = optim.Adam(self.ac_set.xianjia_actor.parameters(), lr=lr)
        self.xianjia_critic_opt = optim.Adam(self.ac_set.xianjia_critic.parameters(), lr=lr)
        
        # Replay Buffer
        self.banker_buffer = deque(maxlen=buffer_capacity)
        self.xianjia_buffer = deque(maxlen=buffer_capacity)
        
        # 统计
        self.episode_count = 0
        self.train_step_count = 0
        self.best_combined_wr = 0.0
    
    def _get_action(self, obs, actions, is_banker, greedy=False):
        """用Actor网络选择动作（epsilon-greedy）"""
        num_legal = len(actions)
        if num_legal == 0:
            return 0, np.zeros(NUM_CARDS)
        if num_legal == 1:
            return 0, actions[0][2].copy()
        
        if not greedy and np.random.random() < self.epsilon:
            action_idx = np.random.randint(num_legal)
        else:
            with torch.no_grad():
                actor = self.ac_set.get_actor(is_banker)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                a_onehots = torch.tensor(
                    np.stack([a[2] for a in actions]),
                    dtype=torch.float32, device=self.device
                )
                q_values = actor.evaluate_actions(obs_t, a_onehots)
                action_idx = q_values.argmax().item()
        
        return action_idx, actions[action_idx][2].copy()
    
    def _get_rule_action(self, game, player_id):
        """规则AI(v28)选择动作"""
        from server.ai import AI
        from rl_shengji.env import id_to_card_type
        room = game.room
        p = room.players[player_id]
        ai = AI(p, room.now_level, room.now_color, room.score_koupai)
        is_first = len(room.epoch_cards) == 0
        play_cards = ai.decide_play(
            room.epoch_cards,
            [room.players[s] for s in room.epoch_players],
            is_first, room.score_now
        )
        
        actions = game.get_legal_actions(player_id)
        if not play_cards or not actions:
            return 0, actions
        
        card_strs = sorted([c.card_type for c in play_cards])
        action_idx = 0
        for i, (_, cids, _) in enumerate(actions):
            a_strs = sorted([id_to_card_type(c) for c in cids])
            if a_strs == card_strs:
                action_idx = i
                break
        
        return action_idx, actions
    
    def collect_episode(self, dmc_team=0):
        """收集一局完整轨迹（DMC vs 规则AI）
        
        记录DMC控制的2个玩家的transition：
        - obs, action_onehot, is_banker
        - episode结束后计算MC return G_t
        """
        game = ShengjiGame()
        room = game.room
        initial_bankers = list(room.bankers) if room.bankers else [0, 2]
        
        # DMC玩家的step数据
        episode_data = [[] for _ in range(4)]  # per-player
        
        while not game.is_over():
            player_id = game.get_current_player()
            if player_id < 0:
                break
            
            actions = game.get_legal_actions(player_id)
            if not actions:
                break
            
            is_banker = player_id in initial_bankers
            player_team = player_id % 2
            is_dmc = (player_team == dmc_team)
            
            # 编码obs
            obs_wrapper = _ObsWrapper(game, initial_bankers)
            obs = encode_obs_v11(obs_wrapper, player_id, initial_bankers)
            
            if is_dmc:
                action_idx, chosen_onehot = self._get_action(obs, actions, is_banker)
                episode_data[player_id].append({
                    'obs': obs.copy(),
                    'action_onehot': chosen_onehot.copy(),
                    'is_banker': is_banker,
                })
            else:
                action_idx, _ = self._get_rule_action(game, player_id)
            
            game.step(action_idx)
        
        # 计算payoff
        payoffs = game.get_payoffs()
        dmc_payoff = payoffs[dmc_team]
        
        # 构建transitions：每个DMC步骤的target = MC return (折扣)
        transitions = []
        for pid in range(4):
            if pid % 2 != dmc_team:
                continue
            steps = episode_data[pid]
            if not steps:
                continue
            
            # 折扣MC return: G_t = sum_{k=t}^{T} gamma^{k-t} * r_k
            # 对升级游戏：中间步骤reward=0，只有最终payoff
            # G_t = gamma^{T-t} * payoff
            n_steps = len(steps)
            for t, step in enumerate(steps):
                discount = self.gamma ** (n_steps - 1 - t)
                mc_return = discount * dmc_payoff
                transitions.append({
                    'obs': step['obs'],
                    'action_onehot': step['action_onehot'],
                    'mc_return': mc_return,
                    'is_banker': step['is_banker'],
                    'pid': pid,
                })
        
        return transitions, payoffs, dmc_team
    
    def train_step(self):
        """Actor-Critic训练更新
        
        Critic loss: MSE(V(s), G_t)
        Actor loss: 用advantage加权的Q值回归
        """
        for role, buffer, actor_opt, critic_opt, actor, critic in [
            ('banker', self.banker_buffer,
             self.banker_actor_opt, self.banker_critic_opt,
             self.ac_set.banker_actor, self.ac_set.banker_critic),
            ('xianjia', self.xianjia_buffer,
             self.xianjia_actor_opt, self.xianjia_critic_opt,
             self.ac_set.xianjia_actor, self.ac_set.xianjia_critic),
        ]:
            if len(buffer) < self.batch_size:
                continue
            
            batch = random.sample(buffer, self.batch_size)
            obs = np.stack([t['obs'] for t in batch])
            action_onehots = np.stack([t['action_onehot'] for t in batch])
            mc_returns = np.array([t['mc_return'] for t in batch], dtype=np.float32)
            
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_t = torch.tensor(action_onehots, dtype=torch.float32, device=self.device)
            mc_t = torch.tensor(mc_returns, dtype=torch.float32, device=self.device)
            
            # === Critic更新: V(s) → G_t ===
            values = critic(obs_t)  # (batch,)
            critic_loss = nn.functional.mse_loss(values, mc_t.detach())
            
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()
            
            # === Actor更新: advantage加权 ===
            # 重新计算values（更新后的critic）
            with torch.no_grad():
                values_new = critic(obs_t)
                advantages = mc_t - values_new  # A(s,a) = G_t - V(s)
                # 归一化advantage
                adv_std = advantages.std()
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / adv_std
            
            # Q(s,a) → 靠近advantage方向
            q_values = actor(obs_t, action_t).squeeze(-1)  # (batch,)
            actor_loss = nn.functional.mse_loss(q_values, values_new + advantages)
            
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()
        
        self.train_step_count += 1
    
    def evaluate(self, num_games=500):
        """评估当前策略 vs 规则AI(v28)"""
        from server.ai import AI
        from rl_shengji.env import id_to_card_type
        
        banker_wins = 0
        xianjia_wins = 0
        banker_games = 0
        xianjia_games = 0
        scores = []
        
        for game_idx in range(num_games):
            game = ShengjiGame()
            room = game.room
            initial_bankers = list(room.bankers) if room.bankers else [0, 2]
            dmc_team = game_idx % 2
            
            while not game.is_over():
                pid = game.get_current_player()
                if pid < 0:
                    break
                actions = game.get_legal_actions(pid)
                if not actions:
                    break
                
                is_banker = pid in initial_bankers
                player_team = pid % 2
                is_dmc = (player_team == dmc_team)
                
                if is_dmc:
                    obs_wrapper = _ObsWrapper(game, initial_bankers)
                    obs = encode_obs_v11(obs_wrapper, pid, initial_bankers)
                    _, chosen_onehot = self._get_action(obs, actions, is_banker, greedy=True)
                    # 用greedy action的index
                    with torch.no_grad():
                        actor = self.ac_set.get_actor(is_banker)
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        a_onehots = torch.tensor(
                            np.stack([a[2] for a in actions]),
                            dtype=torch.float32, device=self.device
                        )
                        q_values = actor.evaluate_actions(obs_t, a_onehots)
                        action_idx = q_values.argmax().item()
                else:
                    action_idx, _ = self._get_rule_action(game, pid)
                
                game.step(action_idx)
            
            payoffs = game.get_payoffs()
            dmc_payoff = payoffs[dmc_team]
            
            if dmc_team in initial_bankers:
                banker_games += 1
                if dmc_payoff > 0:
                    banker_wins += 1
            else:
                xianjia_games += 1
                if dmc_payoff > 0:
                    xianjia_wins += 1
        
        banker_wr = banker_wins / max(banker_games, 1)
        xianjia_wr = xianjia_wins / max(xianjia_games, 1)
        combined_wr = (banker_wins + xianjia_wins) / max(banker_games + xianjia_games, 1)
        
        return {
            'banker_wr': banker_wr,
            'xianjia_wr': xianjia_wr,
            'combined_wr': combined_wr,
            'banker_games': banker_games,
            'xianjia_games': xianjia_games,
            'num_games': num_games,
        }
    
    def load(self, path):
        """加载模型（从V9/V10的QNetworkV2 warm start Actor）"""
        ckpt = torch.load(path, map_location=self.device)
        if 'banker' in ckpt:
            self.ac_set.banker_actor.load_state_dict(ckpt['banker'])
            self.ac_set.xianjia_actor.load_state_dict(ckpt['xianjia'])
            logger.info(f"Actor从Q网络加载成功: {path}")
        else:
            # 直接是state_dict
            self.ac_set.banker_actor.load_state_dict(ckpt)
            logger.info(f"模型加载成功: {path}")
    
    def save(self, path, meta=None):
        """保存模型"""
        save_dict = {
            'banker_actor': self.ac_set.banker_actor.state_dict(),
            'banker_critic': self.ac_set.banker_critic.state_dict(),
            'xianjia_actor': self.ac_set.xianjia_actor.state_dict(),
            'xianjia_critic': self.ac_set.xianjia_critic.state_dict(),
        }
        if meta:
            save_dict['meta'] = meta
        torch.save(save_dict, path)


def main():
    parser = argparse.ArgumentParser(description='DMC V13 Actor-Critic训练')
    parser.add_argument('--num_episodes', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epsilon_start', type=float, default=0.5)
    parser.add_argument('--buffer_capacity', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--train_steps', type=int, default=4)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    trainer = DMCTrainerV13(
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        train_steps_per_episode=args.train_steps,
        device=args.device,
    )
    
    if args.load:
        trainer.load(args.load)
    
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_v13')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"=== DMC V13 Actor-Critic 训练开始 ===")
    logger.info(f"Episodes: {args.num_episodes}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Gamma: {args.gamma}")
    
    start_time = time.time()
    
    for episode in range(1, args.num_episodes + 1):
        dmc_team = episode % 2
        
        try:
            transitions, payoffs, _ = trainer.collect_episode(dmc_team=dmc_team)
            
            # 存入buffer
            for t in transitions:
                if t['is_banker']:
                    trainer.banker_buffer.append(t)
                else:
                    trainer.xianjia_buffer.append(t)
            
            # 训练
            for _ in range(args.train_steps):
                trainer.train_step()
            
        except Exception as e:
            logger.error(f"Episode {episode} error: {e}")
            continue
        
        # 衰减epsilon
        trainer.epsilon = max(trainer.epsilon_end, trainer.epsilon * trainer.epsilon_decay)
        trainer.episode_count += 1
        
        # 打印进度
        if episode % 100 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            buf_b = len(trainer.banker_buffer)
            buf_x = len(trainer.xianjia_buffer)
            dmc_pids = [i for i in range(4) if i % 2 == dmc_team]
            dmc_r = np.mean([payoffs[p] for p in dmc_pids])
            logger.info(
                f"Ep {episode}/{args.num_episodes} | "
                f"eps={trainer.epsilon:.3f} | "
                f"buf_b={buf_b} buf_x={buf_x} | "
                f"DMC_R={dmc_r:.2f} | "
                f"speed={eps_per_sec:.1f}ep/s | "
                f"time={int(elapsed)}s"
            )
        
        # 评估
        if episode % args.eval_interval == 0:
            logger.info(f"=== 评估 Episode {episode} ===")
            trainer.ac_set.eval()
            eval_result = trainer.evaluate(num_games=500)
            trainer.ac_set.train()
            
            combined = eval_result['combined_wr']
            is_best = combined > trainer.best_combined_wr
            if is_best:
                trainer.best_combined_wr = combined
            
            eval_data = {
                'episode': episode,
                'epsilon': trainer.epsilon,
                'eval': eval_result,
                'best_combined_wr': trainer.best_combined_wr,
            }
            
            # 保存评估结果
            eval_path = os.path.join(save_dir, f'eval_ep{episode}.json')
            with open(eval_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
            
            logger.info(
                f"Eval: 庄家{eval_result['banker_wr']:.1%} "
                f"闲家{eval_result['xianjia_wr']:.1%} "
                f"综合{combined:.1%} "
                f"({'BEST' if is_best else 'best=' + f'{trainer.best_combined_wr:.1%}'})"
            )
        
        # 保存模型
        if episode % args.save_interval == 0:
            meta = {
                'episode': episode,
                'epsilon': trainer.epsilon,
                'best_combined_wr': trainer.best_combined_wr,
            }
            
            save_path = os.path.join(save_dir, f'dmc_v13_ep{episode}.pt')
            trainer.save(save_path, meta)
            
            if trainer.best_combined_wr > 0:
                best_path = os.path.join(save_dir, 'dmc_v13_best.pt')
                trainer.save(best_path, meta)
                
                best_meta_path = os.path.join(save_dir, 'dmc_v13_best_meta.json')
                with open(best_meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            
            logger.info(f"模型已保存: {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, 'dmc_v13_final.pt')
    final_meta = {
        'episode': args.num_episodes,
        'epsilon': trainer.epsilon,
        'best_combined_wr': trainer.best_combined_wr,
    }
    trainer.save(final_path, final_meta)
    logger.info(f"训练完成! 最佳综合胜率: {trainer.best_combined_wr:.1%}")


if __name__ == '__main__':
    main()
