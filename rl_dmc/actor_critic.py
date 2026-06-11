# -*- coding: utf-8 -*-
"""
Actor-Critic网络 for 升级游戏DMC训练

参考DouZero DMC架构：
- Actor: QNetworkV2评估legal actions → softmax概率 → 采样
- Critic: obs → V(s) baseline
- Advantage: A(s,a) = G_t - V(s)
- Actor loss: -log π(a|s) * A(s,a)  (policy gradient)
- Critic loss: MSE(V(s), G_t)
"""

import numpy as np
import torch
import torch.nn as nn
from rl_dmc.q_network_v2 import QNetworkV2


class CriticNetwork(nn.Module):
    """Critic网络：obs → V(s)"""
    
    def __init__(self, obs_dim=764, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
    
    def forward(self, obs):
        """V(s) = Critic(obs)"""
        return self.net(obs).squeeze(-1)


class ActorCriticSet(nn.Module):
    """Actor-Critic集合：庄家/闲家各有独立的Actor(Q网络)+Critic"""
    
    def __init__(self, obs_dim=764, num_cards=108, hidden_dim=512,
                 action_embed_dim=128, device='cuda'):
        super().__init__()
        self.device = device
        
        # 庄家
        self.banker_actor = QNetworkV2(obs_dim, num_cards, hidden_dim,
                                        action_embed_dim, dueling=False)
        self.banker_critic = CriticNetwork(obs_dim, hidden_dim)
        
        # 闲家
        self.xianjia_actor = QNetworkV2(obs_dim, num_cards, hidden_dim,
                                          action_embed_dim, dueling=False)
        self.xianjia_critic = CriticNetwork(obs_dim, hidden_dim)
        
        self.to(device)
    
    def get_actor(self, is_banker):
        return self.banker_actor if is_banker else self.xianjia_actor
    
    def get_critic(self, is_banker):
        return self.banker_critic if is_banker else self.xianjia_critic
