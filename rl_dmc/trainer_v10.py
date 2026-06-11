# -*- coding: utf-8 -*-
"""
DMC V10 训练器 - 基于最新规则的PyTorch DMC训练

基于V9模型架构(QNetworkV2) + V3/V4训练经验：
- DQN with experience replay + target network
- 庄家/闲家分别训练(banker_net / xianjia_net)
- MC Return作为目标(纯DMC风格)
- obs: 764维(v11编码)
- action: 108维onehot → ActionEncoder → 128维embedding
- 可从V9 best模型warm start

规则同步：env直接import server.game_engine，所有T1-T6修改、Counter验证
等规则修复自动生效。
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

from rl_dmc.q_network_v2 import QNetworkV2, DMCNetworkSetV2
from rl_dmc.obs_v11 import encode_obs_v11, OBS_DIM_V11
from rl_shengji.env import ShengjiGame, NUM_CARDS, cards_to_ids, cards_to_onehot

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """经验回放池"""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action_onehot, reward, next_obs, done, is_banker):
        """存入一条transition"""
        self.buffer.append({
            'obs': obs,
            'action_onehot': action_onehot,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'is_banker': is_banker,
        })
    
    def push_episode(self, transitions):
        """存入一局完整轨迹"""
        for t in transitions:
            self.push(**t)
    
    def sample(self, batch_size):
        """采样一个batch"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs = np.stack([t['obs'] for t in batch])
        action_onehots = np.stack([t['action_onehot'] for t in batch])
        rewards = np.array([t['reward'] for t in batch], dtype=np.float32)
        next_obs = np.stack([t['next_obs'] for t in batch])
        dones = np.array([t['done'] for t in batch], dtype=np.float32)
        is_bankers = np.array([t['is_banker'] for t in batch], dtype=np.bool_)
        return obs, action_onehots, rewards, next_obs, dones, is_bankers
    
    def __len__(self):
        return len(self.buffer)


# ============================================================
# DMC V10 Trainer
# ============================================================

class DMCTrainerV10:
    """DMC V10训练器
    
    核心设计：
    1. Self-play: 4个玩家使用当前策略(epsilon-greedy)收集轨迹
    2. MC Return: 用完整局面的最终payoff计算return
    3. DQN update: obs + action_onehot → Q值, target = MC return
    4. 庄家/闲家分开buffer和更新
    """
    
    def __init__(self,
                 obs_dim=OBS_DIM_V11,
                 num_cards=NUM_CARDS,
                 hidden_dim=512,
                 action_embed_dim=128,
                 dueling=False,
                 lr=1e-4,
                 gamma=0.99,
                 epsilon_start=0.5,
                 epsilon_end=0.05,
                 epsilon_decay=0.99995,
                 buffer_capacity=100000,
                 batch_size=256,
                 target_update_freq=500,
                 tau=0.005,
                 train_steps_per_episode=4,
                 device='cuda',
                 save_dir='rl_dmc/models_v10'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.num_cards = num_cards
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.train_steps_per_episode = train_steps_per_episode
        
        self.save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            save_dir
        )
        
        # 网络
        self.network_set = DMCNetworkSetV2(
            obs_dim=obs_dim,
            num_cards=num_cards,
            hidden_dim=hidden_dim,
            action_embed_dim=action_embed_dim,
            dueling=dueling,
            device=str(self.device),
        )
        
        # 优化器 - 庄家/闲家分开
        self.banker_optimizer = optim.Adam(
            self.network_set.banker_net.parameters(), lr=lr
        )
        self.xianjia_optimizer = optim.Adam(
            self.network_set.xianjia_net.parameters(), lr=lr
        )
        
        # 经验回放 - 庄家/闲家分开
        self.banker_buffer = ReplayBuffer(buffer_capacity)
        self.xianjia_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练统计
        self.episode_count = 0
        self.train_step_count = 0
        self.total_rewards = defaultdict(list)
        self.best_combined_wr = 0.0
        self.best_banker_wr = 0.0
        self.best_xianjia_wr = 0.0
    
    def collect_episode(self):
        """收集一局完整轨迹（DMC vs 规则AI）
        
        DMC控制一队(2个位置)，规则AI控制另一队(2个位置)。
        每局随机分配DMC做庄家或闲家，确保两个角色都能学习。
        只记录DMC控制位置的transition。
        """
        from server.ai import AI
        
        game = ShengjiGame()
        room = game.room
        
        # 记录初始庄家（room.bankers会在夺庄后改变，必须用初始值）
        initial_bankers = list(room.bankers) if room.bankers else [0, 2]
        
        # 随机分配DMC控制哪一队
        dmc_team = np.random.randint(2)  # 0 or 1
        
        # DMC控制位置的transition列表
        episode_data = []
        
        while not game.is_over():
            player_id = game.get_current_player()
            if player_id < 0:
                break
            
            actions = game.get_legal_actions(player_id)
            if not actions:
                break
            
            is_banker = player_id in initial_bankers
            player_team = player_id % 2
            
            if player_team == dmc_team:
                # DMC控制 - 记录transition
                obs_wrapper = _ObsWrapper(game, initial_bankers)
                obs = encode_obs_v11(obs_wrapper, player_id, obs_wrapper.initial_bankers)
                legal_onehots = [a[2] for a in actions]
                num_legal = len(actions)
                net = self.network_set.get_net(is_banker)
                
                if num_legal == 1:
                    action_idx = 0
                elif np.random.random() < self.epsilon:
                    action_idx = np.random.randint(num_legal)
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        action_onehots_t = torch.tensor(
                            np.stack(legal_onehots),
                            dtype=torch.float32, device=self.device
                        )
                        q_values = net.evaluate_actions(obs_tensor, action_onehots_t)
                        action_idx = q_values.argmax().item()
                
                chosen_onehot = legal_onehots[action_idx]
                episode_data.append({
                    'obs': obs.copy(),
                    'action_onehot': chosen_onehot.copy(),
                    'is_banker': is_banker,
                })
                
                game.step(action_idx)
            else:
                # 规则AI控制 - 不记录transition
                p = room.players[player_id]
                ai = AI(p, room.now_level, room.now_color, room.score_koupai)
                is_first = len(room.epoch_cards) == 0
                play_cards = ai.decide_play(
                    room.epoch_cards,
                    [room.players[s] for s in room.epoch_players],
                    is_first, room.score_now
                )
                
                action_idx = 0
                if play_cards:
                    card_strs = sorted([c.card_type for c in play_cards])
                    from rl_shengji.env import id_to_card_type
                    for i, (_, cids, _) in enumerate(actions):
                        a_strs = sorted([id_to_card_type(c) for c in cids])
                        if a_strs == card_strs:
                            action_idx = i
                            break
                
                game.step(action_idx)
        
        # 计算payoff
        payoffs = game.get_payoffs()
        
        # 只构建DMC控制位置的transition
        dmc_payoff = payoffs[dmc_team]  # DMC队友payoff相同
        transitions = []
        for t in episode_data:
            transitions.append({
                'obs': t['obs'],
                'action_onehot': t['action_onehot'],
                'reward': dmc_payoff,  # MC return
                'next_obs': t['obs'],
                'done': True,
                'is_banker': t['is_banker'],
            })
        
        return transitions, payoffs, dmc_team
    
    def train_step(self):
        """执行一步训练更新"""
        # 分别从庄家/闲家buffer采样
        for role, buffer, optimizer, net, target_net in [
            ('banker', self.banker_buffer, self.banker_optimizer,
             self.network_set.banker_net, self.network_set.banker_target),
            ('xianjia', self.xianjia_buffer, self.xianjia_optimizer,
             self.network_set.xianjia_net, self.network_set.xianjia_target),
        ]:
            if len(buffer) < self.batch_size:
                continue
            
            obs, action_onehots, rewards, _, dones, _ = buffer.sample(self.batch_size)
            
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_t = torch.tensor(action_onehots, dtype=torch.float32, device=self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
            # 当前Q值
            q_values = net(obs_t, action_t).squeeze(-1)  # (batch,)
            
            # MC target = reward (no bootstrap for DMC)
            with torch.no_grad():
                target_q = rewards_t
            
            # MSE loss
            loss = nn.functional.mse_loss(q_values, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        
        self.train_step_count += 1
        # 注：纯MC方法不需要target network，不做soft update
    
    def evaluate(self, num_games=50):
        """评估当前策略 vs 规则AI(v28)
        
        DMC控制2个位置（同队），规则AI控制另外2个位置（对手队）。
        分别在庄家方和闲家方各测num_games//2局。
        """
        from server.ai import AI
        
        banker_wins = 0
        xianjia_wins = 0
        banker_games = 0
        xianjia_games = 0
        scores = []
        
        for game_idx in range(num_games):
            game = ShengjiGame()
            room = game.room
            
            # 记录初始庄家（room.bankers会在夺庄后改变，必须用初始值）
            initial_bankers = list(room.bankers) if room.bankers else [0, 2]
            
            # 确定DMC控制哪一队
            # 偶数局: DMC=team0, 奇数局: DMC=team1
            dmc_team = game_idx % 2
            
            while not game.is_over():
                player_id = game.get_current_player()
                if player_id < 0:
                    break
                
                actions = game.get_legal_actions(player_id)
                if not actions:
                    break
                
                is_banker = player_id in initial_bankers
                player_team = player_id % 2
                
                if player_team == dmc_team:
                    # DMC策略
                    net = self.network_set.get_net(is_banker)
                    with torch.no_grad():
                        obs_wrapper = _ObsWrapper(game, initial_bankers)
                        obs = encode_obs_v11(obs_wrapper, player_id, obs_wrapper.initial_bankers)
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        action_onehots_t = torch.tensor(
                            np.stack([a[2] for a in actions]),
                            dtype=torch.float32, device=self.device
                        )
                        q_values = net.evaluate_actions(obs_tensor, action_onehots_t)
                        action_idx = q_values.argmax().item()
                    action_idx = min(action_idx, len(actions) - 1)
                else:
                    # 规则AI(v28) - 与game_engine.auto_play_robots相同逻辑
                    from server.ai import AI
                    p = room.players[player_id]
                    ai = AI(p, room.now_level, room.now_color, room.score_koupai)
                    is_first_play = len(room.epoch_cards) == 0
                    play_cards = ai.decide_play(
                        room.epoch_cards,
                        [room.players[s] for s in room.epoch_players],
                        is_first_play,
                        room.score_now
                    )
                    
                    if play_cards:
                        card_strs = [c.card_type for c in play_cards]
                        # 匹配到legal actions
                        action_idx = 0
                        best_match = -1
                        for i, (_, cids, _) in enumerate(actions):
                            from rl_shengji.env import id_to_card_type
                            a_strs = sorted([id_to_card_type(c) for c in cids])
                            if a_strs == sorted(card_strs):
                                best_match = i
                                break
                        if best_match >= 0:
                            action_idx = best_match
                    else:
                        action_idx = 0
                
                game.step(action_idx)
            
            payoffs = game.get_payoffs()
            score_now = getattr(room, 'score_now', 0)
            scores.append(score_now)
            
            # 统计DMC队伍胜率（核心指标）
            # DMC控制dmc_team (pid%2==dmc_team的2个玩家)
            dmc_team_won = payoffs[dmc_team] > 0  # 队友payoff同号
            
            # 用初始庄家判断（room.bankers在夺庄后会改变）
            banker_team = initial_bankers[0] % 2
            dmc_is_banker = (dmc_team == banker_team)
            
            if dmc_is_banker:
                banker_games += 1
                if dmc_team_won:
                    banker_wins += 1
            else:
                xianjia_games += 1
                if dmc_team_won:
                    xianjia_wins += 1
        
        dmc_wr = (banker_wins + xianjia_wins) / max(banker_games + xianjia_games, 1)
        banker_wr = banker_wins / max(banker_games, 1)
        xianjia_wr = xianjia_wins / max(xianjia_games, 1)
        avg_score = np.mean(scores) if scores else 0
        
        return {
            'banker_wr': banker_wr,
            'xianjia_wr': xianjia_wr,
            'combined_wr': dmc_wr,  # DMC队伍总胜率
            'avg_score': avg_score,
            'num_games': num_games,
            'banker_games': banker_games,
            'xianjia_games': xianjia_games,
            'total': banker_games + xianjia_games,
        }
    
    def train(self, num_episodes=30000, eval_interval=1000, save_interval=5000):
        """训练主循环"""
        logger.info(f"=== DMC V10 训练开始 ===")
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Obs dim: {self.obs_dim}")
        logger.info(f"Epsilon: {self.epsilon:.3f} -> {self.epsilon_end:.3f}")
        logger.info(f"Buffer capacity: {self.banker_buffer.capacity}")
        logger.info(f"Save dir: {self.save_dir}")
        logger.info("=" * 60)
        
        start_time = time.time()
        os.makedirs(self.save_dir, exist_ok=True)
        
        for episode in range(1, num_episodes + 1):
            # 收集轨迹
            try:
                transitions, payoffs, dmc_team = self.collect_episode()
            except Exception as e:
                logger.error(f"Episode {episode} 收集失败: {e}")
                continue
            
            # 存入buffer
            for t in transitions:
                if t['is_banker']:
                    self.banker_buffer.push(**{k: t[k] for k in 
                        ['obs', 'action_onehot', 'reward', 'next_obs', 'done', 'is_banker']})
                else:
                    self.xianjia_buffer.push(**{k: t[k] for k in 
                        ['obs', 'action_onehot', 'reward', 'next_obs', 'done', 'is_banker']})
            
            # 训练
            for _ in range(self.train_steps_per_episode):
                try:
                    self.train_step()
                except Exception as e:
                    logger.error(f"Train step error: {e}")
                    continue
            
            # 衰减epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episode_count += 1
            
            # 记录DMC队的平均reward
            dmc_pids = [i for i in range(4) if i % 2 == dmc_team]
            self.total_rewards['dmc'].append(np.mean([payoffs[p] for p in dmc_pids]))
            rule_pids = [i for i in range(4) if i % 2 != dmc_team]
            self.total_rewards['rule'].append(np.mean([payoffs[p] for p in rule_pids]))
            
            # 打印进度
            if episode % 100 == 0:
                elapsed = time.time() - start_time
                dmc_avg = np.mean(self.total_rewards['dmc'][-100:]) if self.total_rewards['dmc'] else 0
                rule_avg = np.mean(self.total_rewards['rule'][-100:]) if self.total_rewards['rule'] else 0
                eps_per_sec = episode / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Ep {episode}/{num_episodes} | "
                    f"eps={self.epsilon:.3f} | "
                    f"buf_b={len(self.banker_buffer)} buf_x={len(self.xianjia_buffer)} | "
                    f"DMC_R={dmc_avg:.2f} Rule_R={rule_avg:.2f} | "
                    f"speed={eps_per_sec:.1f}ep/s | "
                    f"time={elapsed:.0f}s"
                )
            
            # 评估
            if episode % eval_interval == 0:
                eval_result = self.evaluate(num_games=1000)
                logger.info(
                    f"  >>> Eval@{episode}: "
                    f"banker_wr={eval_result['banker_wr']:.2%} "
                    f"xianjia_wr={eval_result['xianjia_wr']:.2%} "
                    f"combined={eval_result['combined_wr']:.2%} "
                    f"avg_score={eval_result['avg_score']:.1f}"
                )
                
                # 保存best模型
                if eval_result['combined_wr'] > self.best_combined_wr:
                    self.best_combined_wr = eval_result['combined_wr']
                    self.best_banker_wr = eval_result['banker_wr']
                    self.best_xianjia_wr = eval_result['xianjia_wr']
                    self.save('best')
                    logger.info(f"  ★ New best! combined={self.best_combined_wr:.2%}")
                
                # 保存评估结果
                eval_path = os.path.join(self.save_dir, f'eval_ep{episode}.json')
                try:
                    with open(eval_path, 'w') as f:
                        json.dump({
                            'episode': episode,
                            'epsilon': self.epsilon,
                            'eval': eval_result,
                            'best_combined_wr': self.best_combined_wr,
                        }, f, indent=2)
                except:
                    pass
            
            # 定期保存checkpoint
            if episode % save_interval == 0:
                self.save(f'ep{episode}')
        
        # 训练结束
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"训练完成! 总局数: {num_episodes}, 总时间: {total_time:.1f}s")
        logger.info(f"最佳胜率: combined={self.best_combined_wr:.2%} "
                    f"banker={self.best_banker_wr:.2%} xianjia={self.best_xianjia_wr:.2%}")
        self.save('final')
    
    def save(self, tag='latest'):
        """保存模型"""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except OSError:
            pass
        
        path = os.path.join(self.save_dir, f'dmc_v10_{tag}.pt')
        self.network_set.save(path)
        
        # 额外保存训练状态
        meta_path = os.path.join(self.save_dir, f'dmc_v10_{tag}_meta.json')
        try:
            with open(meta_path, 'w') as f:
                json.dump({
                    'episode_count': self.episode_count,
                    'train_step_count': self.train_step_count,
                    'epsilon': self.epsilon,
                    'best_combined_wr': self.best_combined_wr,
                    'best_banker_wr': self.best_banker_wr,
                    'best_xianjia_wr': self.best_xianjia_wr,
                }, f, indent=2)
        except:
            pass
        
        logger.info(f"模型已保存: {path}")
    
    def load(self, path):
        """加载模型"""
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            return False
        
        self.network_set.load(path)
        logger.info(f"模型已加载: {path}")
        return True


class _ObsWrapper:
    """包装ShengjiGame以兼容obs_v11.encode_obs_v11接口
    
    不修改room.bankers，而是存储initial_bankers供encode_obs_v11使用。
    """
    
    def __init__(self, game, initial_bankers=None):
        self.room = game.room
        self._game = game
        self._initial_bankers = initial_bankers
    
    def get_state(self, player_id):
        """返回base 693维obs"""
        return {'obs': self._game._encode_observation(player_id)}
    
    @property
    def initial_bankers(self):
        return self._initial_bankers


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMC V10 训练升级AI')
    parser.add_argument('--num_episodes', type=int, default=30000, help='训练局数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--epsilon_start', type=float, default=0.5, help='初始探索率')
    parser.add_argument('--buffer_capacity', type=int, default=100000, help='经验池容量')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--dueling', action='store_true', default=False, help='Dueling DQN')
    parser.add_argument('--train_steps', type=int, default=4, help='每episode训练步数')
    parser.add_argument('--eval_interval', type=int, default=5000, help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存间隔')
    parser.add_argument('--load', type=str, default=None, help='加载模型路径(warm start)')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    
    args = parser.parse_args()
    
    trainer = DMCTrainerV10(
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dueling=args.dueling,
        train_steps_per_episode=args.train_steps,
        device=args.device,
    )
    
    if args.load:
        trainer.load(args.load)
    
    trainer.train(
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )
