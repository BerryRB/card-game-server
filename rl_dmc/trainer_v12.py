# -*- coding: utf-8 -*-
"""
DMC V12 训练器 - Self-play + TD target + vs规则AI评估

修复V11的4个Bug：
1. epoch_history → obs编码正确（game_engine已修）
2. per-step reward（赢墩+0.1/输墩-0.05 + 最终payoff）
3. next_obs记录真正的下一步obs
4. done只有最后一步True

架构：
- 训练：Self-play（4个玩家全DMC，庄家位用banker_net，闲家位用xianjia_net）
- 评估：DMC vs 规则AI(v28)
- Dueling DQN + target_net软更新
- TD target: V(s')=max_a Q(s',a) 在collect时用target_net预计算
- MC return作为辅助target（权重0.3）
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


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action_onehot, reward, next_obs, done, is_banker, mc_return):
        self.buffer.append({
            'obs': obs,
            'action_onehot': action_onehot,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'is_banker': is_banker,
            'mc_return': mc_return,
        })

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return (
            np.array([b['obs'] for b in batch]),
            np.array([b['action_onehot'] for b in batch]),
            np.array([b['reward'] for b in batch]),
            np.array([b['next_obs'] for b in batch]),
            np.array([b['done'] for b in batch], dtype=np.float32),
            np.array([b['is_banker'] for b in batch], dtype=np.float32),
            np.array([b['mc_return'] for b in batch]),
        )

    def __len__(self):
        return len(self.buffer)


class DMCTrainerV12:
    def __init__(self, device='cuda', lr=1e-4, gamma=0.99,
                 buffer_capacity=100000, batch_size=256,
                 epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.99995,
                 target_update_tau=0.005, dueling=False):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_tau = target_update_tau
        self.train_steps_per_episode = 2
        self.obs_dim = OBS_DIM_V11

        # 网络
        self.network_set = DMCNetworkSetV2(
            obs_dim=self.obs_dim,
            num_cards=NUM_CARDS,
            dueling=dueling,
            device=device,
        )

        self.save_dir = os.path.join(os.path.dirname(__file__), 'models_v12')

        # 优化器
        self.banker_optimizer = optim.Adam(
            self.network_set.banker_net.parameters(), lr=lr
        )
        self.xianjia_optimizer = optim.Adam(
            self.network_set.xianjia_net.parameters(), lr=lr
        )

        # 经验回放（庄家/闲家分开buffer）
        self.banker_buffer = ReplayBuffer(buffer_capacity)
        self.xianjia_buffer = ReplayBuffer(buffer_capacity)

        # 训练统计
        self.episode_count = 0
        self.train_step_count = 0
        self.total_rewards = defaultdict(list)
        self.best_combined_wr = 0.0

    def _get_action(self, obs, actions, is_banker, greedy=False):
        """用DMC网络选择动作（epsilon-greedy）"""
        num_legal = len(actions)
        if num_legal == 0:
            return 0, None
        if num_legal == 1:
            return 0, actions[0][2].copy()

        net = self.network_set.get_net(is_banker)

        if not greedy and np.random.random() < self.epsilon:
            action_idx = np.random.randint(num_legal)
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                a_onehots = torch.tensor(
                    np.stack([a[2] for a in actions]),
                    dtype=torch.float32, device=self.device
                )
                q_values = net.evaluate_actions(obs_t, a_onehots)
                action_idx = q_values.argmax().item()

        return action_idx, actions[action_idx][2].copy()

    def _estimate_v(self, obs, actions, is_banker):
        """用target_net估计V(s) = max_a Q(s,a)，枚举legal actions"""
        target_net = self.network_set.banker_target if is_banker else self.network_set.xianjia_target
        if not actions:
            return 0.0
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            a_onehots = torch.tensor(
                np.stack([a[2] for a in actions]),
                dtype=torch.float32, device=self.device
            )
            q_values = target_net.evaluate_actions(obs_t, a_onehots)
            return q_values.max().item()

    def _get_rule_action(self, game, player_id):
        """规则AI(v28)选择动作"""
        from server.ai import AI
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
        if not actions or not play_cards:
            return 0, actions
        from rl_shengji.env import id_to_card_type
        card_strs = sorted([c.card_type for c in play_cards])
        for i, (_, cids, _) in enumerate(actions):
            a_strs = sorted([id_to_card_type(c) for c in cids])
            if a_strs == card_strs:
                return i, actions
        return 0, actions

    # ============================================================
    # Self-play 训练数据收集
    # ============================================================

    def collect_episode(self):
        """收集一局完整轨迹（Self-play：4个玩家全DMC网络控制）

        庄家位(0,2)用banker_net，闲家位(1,3)用xianjia_net。
        每步记录obs, action_onehot, step_reward, next_obs, done,
        以及预计算的TD target和MC return。
        """
        game = ShengjiGame()
        room = game.room
        initial_bankers = list(room.bankers) if room.bankers else [0, 2]

        # 所有4个玩家的transition列表
        all_transitions = [[] for _ in range(4)]
        pending = {}  # pid -> (obs, action_onehot, step_reward, is_banker)
        last_player_id = None  # 记录最后出牌者（用于payoff处理）

        while not game.is_over():
            player_id = game.get_current_player()
            if player_id < 0:
                break

            actions = game.get_legal_actions(player_id)
            if not actions:
                break

            is_banker = player_id in initial_bankers

            # 编码当前obs
            obs_wrapper = _ObsWrapper(game, initial_bankers)
            obs = encode_obs_v11(obs_wrapper, player_id, initial_bankers)

            # Self-play: 所有玩家都用DMC网络
            action_idx, chosen_onehot = self._get_action(obs, actions, is_banker)

            # 完成上一步的transition
            if player_id in pending:
                p_obs, p_action, p_reward, p_banker = pending.pop(player_id)
                # V(s') = max_a Q(s', a)，用当前步骤的legal actions枚举
                v_next = self._estimate_v(obs, actions, p_banker)
                all_transitions[player_id].append({
                    'obs': p_obs,
                    'action_onehot': p_action,
                    'step_reward': p_reward,
                    'next_obs': obs.copy(),
                    'done': False,
                    'is_banker': p_banker,
                    'v_next': v_next,
                })

            # 执行动作
            _, _, step_reward, done = game.step(action_idx)
            if done:
                last_player_id = player_id

            # 所有玩家都存入pending（self-play）
            pending[player_id] = (obs.copy(), chosen_onehot, step_reward, is_banker)

        # 游戏结束，完成所有pending transition
        # Bug6修复：env.step()在done=True时已将payoff加入step_reward
        # 所以最后出牌者(last_player)的step_reward包含了payoff
        # 其他3个pending玩家的step_reward不包含payoff，需要手动加上
        payoffs = game.get_payoffs()

        for pid, (p_obs, p_action, p_reward, p_banker) in pending.items():
            if pid == last_player_id:
                # 最后出牌者：step_reward已包含payoff，不加
                final_reward = p_reward
            else:
                # 非最后出牌者：step_reward不含payoff，加上
                final_reward = p_reward + payoffs[pid]
            all_transitions[pid].append({
                'obs': p_obs,
                'action_onehot': p_action,
                'step_reward': final_reward,
                'next_obs': p_obs.copy(),
                'done': True,
                'is_banker': p_banker,
                'v_next': 0.0,
            })

        # 按玩家计算MC return，并组合TD target
        transitions = []
        for pid in range(4):
            traj = all_transitions[pid]
            if not traj:
                continue
            # 从后往前计算折扣MC return
            G = 0.0
            for t in reversed(range(len(traj))):
                G = traj[t]['step_reward'] + self.gamma * G
                r = traj[t]['step_reward']
                done = traj[t]['done']
                v_next = traj[t]['v_next']

                # TD target = r + gamma * V(s')
                td_target = r + self.gamma * v_next if not done else r

                transitions.append({
                    'obs': traj[t]['obs'],
                    'action_onehot': traj[t]['action_onehot'],
                    'reward': td_target,
                    'next_obs': traj[t]['next_obs'],
                    'done': traj[t]['done'],
                    'is_banker': traj[t]['is_banker'],
                    'mc_return': G,
                })

        payoffs = game.get_payoffs()
        return transitions, payoffs

    # ============================================================
    # 训练更新
    # ============================================================

    def train_step(self):
        """DQN训练更新（TD target + MC辅助loss）

        TD target在collect_episode中已预计算，直接用。
        同时加MC return辅助loss（权重0.3）减少方差。
        """
        for role, buffer, optimizer, net, target_net in [
            ('banker', self.banker_buffer, self.banker_optimizer,
             self.network_set.banker_net, self.network_set.banker_target),
            ('xianjia', self.xianjia_buffer, self.xianjia_optimizer,
             self.network_set.xianjia_net, self.network_set.xianjia_target),
        ]:
            if len(buffer) < self.batch_size:
                continue

            obs, action_onehots, td_targets, next_obs, dones, _, mc_returns = buffer.sample(self.batch_size)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action_t = torch.tensor(action_onehots, dtype=torch.float32, device=self.device)
            td_target_t = torch.tensor(td_targets, dtype=torch.float32, device=self.device)
            mc_return_t = torch.tensor(mc_returns, dtype=torch.float32, device=self.device)

            # 当前Q值
            q_values = net(obs_t, action_t).squeeze(-1)

            # 主loss: Q(s,a) vs TD target
            td_loss = nn.functional.smooth_l1_loss(q_values, td_target_t)

            # 辅助loss: Q(s,a) vs MC return（权重0.3）
            mc_loss = nn.functional.mse_loss(q_values, mc_return_t)

            loss = td_loss + 0.3 * mc_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            # 软更新target network
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(
                    self.target_update_tau * param.data + (1 - self.target_update_tau) * target_param.data
                )

            self.train_step_count += 1

    # ============================================================
    # 评估（vs 规则AI）
    # ============================================================

    def evaluate(self, num_games=500):
        """评估当前策略 vs 规则AI(v28)"""
        from server.ai import AI

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
                player_id = game.get_current_player()
                if player_id < 0:
                    break

                actions = game.get_legal_actions(player_id)
                if not actions:
                    break

                is_banker = player_id in initial_bankers
                player_team = player_id % 2

                if player_team == dmc_team:
                    # DMC选择动作
                    net = self.network_set.get_net(is_banker)
                    with torch.no_grad():
                        obs_wrapper = _ObsWrapper(game, initial_bankers)
                        obs = encode_obs_v11(obs_wrapper, player_id, initial_bankers)
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        a_onehots = torch.tensor(
                            np.stack([a[2] for a in actions]),
                            dtype=torch.float32, device=self.device
                        )
                        q_values = net.evaluate_actions(obs_t, a_onehots)
                        action_idx = q_values.argmax().item()
                    action_idx = min(action_idx, len(actions) - 1)
                else:
                    # 规则AI选择动作
                    action_idx, _ = self._get_rule_action(game, player_id)

                game.step(action_idx)

            payoffs = game.get_payoffs()
            score_now = getattr(room, 'score_now', 0)
            scores.append(score_now)

            dmc_team_won = payoffs[dmc_team] > 0
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
            'combined_wr': dmc_wr,
            'avg_score': avg_score,
            'num_games': num_games,
            'banker_games': banker_games,
            'xianjia_games': xianjia_games,
        }

    # ============================================================
    # 训练主循环
    # ============================================================

    def train(self, num_episodes=100000, eval_interval=5000, save_interval=10000):
        """训练主循环（self-play + vs规则AI评估）"""
        logger.info(f"=== DMC V12 Self-play 训练开始 ===")
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Obs dim: {self.obs_dim}")
        logger.info(f"Dueling: {self.network_set.banker_net.dueling}")
        logger.info(f"Epsilon: {self.epsilon:.3f} -> {self.epsilon_end:.3f}")
        logger.info(f"Gamma: {self.gamma}")
        logger.info(f"Buffer capacity: {self.banker_buffer.capacity}")
        logger.info(f"Train: Self-play (4 players DMC)")
        logger.info(f"Eval: vs Rule AI (500 games)")
        logger.info(f"Save dir: {self.save_dir}")
        logger.info("=" * 60)

        start_time = time.time()
        os.makedirs(self.save_dir, exist_ok=True)

        for episode in range(1, num_episodes + 1):
            try:
                transitions, payoffs = self.collect_episode()
            except Exception as e:
                logger.error(f"Episode {episode} collect failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 存入对应buffer
            for t in transitions:
                buf = self.banker_buffer if t['is_banker'] else self.xianjia_buffer
                buf.push(
                    obs=t['obs'],
                    action_onehot=t['action_onehot'],
                    reward=t['reward'],
                    next_obs=t['next_obs'],
                    done=t['done'],
                    is_banker=t['is_banker'],
                    mc_return=t['mc_return'],
                )

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

            # 打印进度
            if episode % 100 == 0:
                elapsed = time.time() - start_time
                eps_per_sec = episode / elapsed if elapsed > 0 else 0
                avg_payoff_b = np.mean([payoffs[i] for i in [0, 2]])  # 庄家队
                avg_payoff_x = np.mean([payoffs[i] for i in [1, 3]])  # 闲家队
                logger.info(
                    f"Ep {episode}/{num_episodes} | "
                    f"eps={self.epsilon:.3f} | "
                    f"buf_b={len(self.banker_buffer)} buf_x={len(self.xianjia_buffer)} | "
                    f"payoff_b={avg_payoff_b:.2f} payoff_x={avg_payoff_x:.2f} | "
                    f"speed={eps_per_sec:.1f}ep/s | "
                    f"time={elapsed:.0f}s"
                )

            # 评估（vs 规则AI）
            if episode % eval_interval == 0:
                eval_result = self.evaluate(num_games=500)
                logger.info(
                    f"  >>> Eval@{episode}: "
                    f"banker_wr={eval_result['banker_wr']:.2%} "
                    f"xianjia_wr={eval_result['xianjia_wr']:.2%} "
                    f"combined={eval_result['combined_wr']:.2%} "
                    f"avg_score={eval_result['avg_score']:.1f}"
                )

                if eval_result['combined_wr'] > self.best_combined_wr:
                    self.best_combined_wr = eval_result['combined_wr']
                    self.save('best')
                    logger.info(f"  ★ New best! combined={self.best_combined_wr:.2%}")

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

        # 最终保存
        self.save('final')
        logger.info(f"=== 训练完成 === best_combined_wr={self.best_combined_wr:.2%}")

    # ============================================================
    # 保存/加载
    # ============================================================

    def save(self, tag=''):
        path = os.path.join(self.save_dir, f'dmc_v12_{tag}.pt')
        meta = {
            'episode': self.episode_count,
            'epsilon': self.epsilon,
            'best_combined_wr': self.best_combined_wr,
            'train_step_count': self.train_step_count,
        }
        torch.save({
            'banker_net': self.network_set.banker_net.state_dict(),
            'banker_target': self.network_set.banker_target.state_dict(),
            'xianjia_net': self.network_set.xianjia_net.state_dict(),
            'xianjia_target': self.network_set.xianjia_target.state_dict(),
            'banker_optimizer': self.banker_optimizer.state_dict(),
            'xianjia_optimizer': self.xianjia_optimizer.state_dict(),
            'meta': meta,
        }, path)
        meta_path = os.path.join(self.save_dir, f'dmc_v12_{tag}_meta.json')
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
        except:
            pass
        logger.info(f"Saved: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network_set.banker_net.load_state_dict(ckpt['banker_net'])
        self.network_set.banker_target.load_state_dict(ckpt['banker_target'])
        self.network_set.xianjia_net.load_state_dict(ckpt['xianjia_net'])
        self.network_set.xianjia_target.load_state_dict(ckpt['xianjia_target'])
        if 'banker_optimizer' in ckpt:
            self.banker_optimizer.load_state_dict(ckpt['banker_optimizer'])
        if 'xianjia_optimizer' in ckpt:
            self.xianjia_optimizer.load_state_dict(ckpt['xianjia_optimizer'])
        meta = ckpt.get('meta', {})
        self.episode_count = meta.get('episode', 0)
        self.epsilon = meta.get('epsilon', self.epsilon)
        self.best_combined_wr = meta.get('best_combined_wr', 0.0)
        self.train_step_count = meta.get('train_step_count', 0)
        logger.info(f"Loaded: {path} (ep={self.episode_count}, best={self.best_combined_wr:.2%})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dueling', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    trainer = DMCTrainerV12(device=args.device, dueling=args.dueling)

    if args.load:
        trainer.load(args.load)

    trainer.train(
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )
