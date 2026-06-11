#!/usr/bin/env python3
"""DMC V11 训练脚本 - 修复所有已知Bug后从零训练

Bug修复清单:
1. game_engine._end_game() 加入banker_team字段 → payoff不再45%反转
2. evaluate 50局→1000局 → best模型更可靠
3. obs编码用initial_bankers传参 → 不修改room.bankers
4. 移除无用target network soft update
5. get_payoffs fallback用_initial_bankers
6. 日志只记录DMC队reward

用法:
    python rl_dmc/train_v11.py --num_episodes 100000 --device cuda
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_dmc.trainer_v10 import DMCTrainerV10

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DMC V11 Training')
    parser.add_argument('--num_episodes', type=int, default=100000, help='训练局数')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--dueling', action='store_true', default=False, help='Dueling DQN')
    parser.add_argument('--train_steps', type=int, default=4, help='每episode训练步数')
    parser.add_argument('--eval_interval', type=int, default=5000, help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=5000, help='保存间隔')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--save_dir', type=str, default='rl_dmc/models_v11', help='模型保存目录')
    args = parser.parse_args()
    
    print(f"=== DMC V11 训练 ===")
    print(f"  Bug修复: payoff反转/obs篡room.bankers/target net/日志")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Eval: 每{args.eval_interval}局 x 1000局")
    print(f"  Device: {args.device}")
    print(f"  Save: {args.save_dir}")
    
    trainer = DMCTrainerV10(
        device=args.device,
        hidden_dim=args.hidden_dim,
        dueling=args.dueling,
        save_dir=args.save_dir,
        epsilon_start=1.0,
    )
    
    trainer.train(
        num_episodes=args.num_episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )

if __name__ == '__main__':
    main()
