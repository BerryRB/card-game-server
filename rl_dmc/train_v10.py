#!/usr/bin/env python3
"""DMC V10 正式训练脚本 - DMC vs 规则AI"""
import sys, os, logging

# 确保项目根目录在path中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保stdout无缓冲
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)

from rl_dmc.trainer_v10 import DMCTrainerV10

trainer = DMCTrainerV10(
    device='cuda',
    hidden_dim=512,
    action_embed_dim=128,
    lr=1e-4,
    buffer_capacity=100000,
    batch_size=256,
    epsilon_start=1.0,    # 从纯探索开始
    epsilon_end=0.05,
    epsilon_decay=0.99997,  # 30K ep后约0.4
    target_update_freq=500,
    tau=0.005,
    gamma=0.99,
    dueling=False,
)

# 不加载V9，新规则从头训练
print('Training from scratch with new rules!', flush=True)

trainer.train(num_episodes=50000, eval_interval=2000, save_interval=5000)
print('TRAINING COMPLETE!', flush=True)
