# -*- coding: utf-8 -*-
"""
DouZero DMC Q-Network for 升级(Trump)游戏 - v2

核心修正:
- 升级游戏中，出牌动作可能是单牌或对子
- env.py返回的每个legal action有一个index (0, 1, 2, ...)
- 每个action对应一个onehot向量(108维), 对子时多个位置为1

设计选择:
- Q网络输入: obs(764) + action_embedding(128) → Q值
- 每步遍历所有legal actions, 逐一评估Q值
- 选择Q值最高的legal action

这样设计的好处:
1. 正确处理对子/连对等多牌动作
2. 动作空间不固定, 适应每步不同的合法动作数
3. 类似DouZero原始设计: obs+action → Q
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActionEncoder(nn.Module):
    """动作编码器: 将108维onehot编码为低维embedding
    
    对子等组合动作的onehot中有多个1, 编码器需要能处理这种情况
    """
    
    def __init__(self, num_cards=108, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_cards, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
    
    def forward(self, action_onehot):
        """action_onehot: (batch, 108) 或 (num_actions, 108)"""
        return self.encoder(action_onehot)


class QNetworkV2(nn.Module):
    """DouZero风格Q网络 v2
    
    架构: Q(obs, action) → scalar
    - obs编码: 764维 → hidden
    - action编码: 108维onehot → embed_dim
    - 合并: [obs_hidden; action_embed] → Q值
    
    每步对每个legal action计算Q值, 选最大的
    """
    
    def __init__(self, obs_dim=764, num_cards=108, hidden_dim=512, 
                 action_embed_dim=128, dueling=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_cards = num_cards
        self.hidden_dim = hidden_dim
        self.dueling = dueling
        
        # obs编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # action编码器
        self.action_encoder = ActionEncoder(num_cards, action_embed_dim)
        
        # Q值头: obs_features + action_embed → Q
        merge_dim = hidden_dim + action_embed_dim
        
        if dueling:
            # Dueling: 分离状态价值和动作优势
            self.value_head = nn.Sequential(
                nn.Linear(merge_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(merge_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(merge_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # 最后一层小初始化
        if self.dueling:
            nn.init.orthogonal_(self.advantage_head[-1].weight, gain=0.01)
            nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        else:
            nn.init.orthogonal_(self.q_head[-1].weight, gain=0.01)
    
    def forward(self, obs, action_onehot):
        """计算Q(obs, action)
        
        Args:
            obs: (batch, obs_dim) 观测
            action_onehot: (batch, 108) 动作onehot编码
            
        Returns:
            q_value: (batch, 1) Q值
        """
        obs_features = self.obs_encoder(obs)        # (batch, hidden)
        action_embed = self.action_encoder(action_onehot)  # (batch, embed_dim)
        merged = torch.cat([obs_features, action_embed], dim=-1)  # (batch, hidden+embed)
        
        if self.dueling:
            value = self.value_head(merged)       # (batch, 1)
            advantage = self.advantage_head(merged)  # (batch, 1)
            # 注意: 这里不扣mean, 因为batch中每个(obs,action)来自不同状态
            # 正确的Dueling mean是在同一状态的不同动作间取
            # evaluate_actions中会正确处理
            q_value = value + advantage
        else:
            q_value = self.q_head(merged)         # (batch, 1)
        
        return q_value
    
    def evaluate_actions(self, obs, action_onehots):
        """评估多个动作的Q值
        
        Args:
            obs: (1, obs_dim) 或 (obs_dim,) 单个观测
            action_onehots: (num_actions, 108) 多个候选动作
            
        Returns:
            q_values: (num_actions,) 每个动作的Q值
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 扩展obs到num_actions个
        num_actions = action_onehots.shape[0]
        obs_expanded = obs.expand(num_actions, -1)  # (num_actions, obs_dim)
        
        q_values = self.forward(obs_expanded, action_onehots)  # (num_actions, 1)
        q_values = q_values.squeeze(-1)  # (num_actions,)
        
        # Dueling: 在同一状态的不同动作间扣减advantage均值
        # forward中已返回V+A, 这里需要减去mean(A)
        # 由于Q = V + A, 而所有动作共享V, 减去mean(Q)等价于减去mean(A)
        if self.dueling and q_values.numel() > 1:
            q_values = q_values - q_values.mean()
        
        return q_values
    
    def get_action(self, obs, action_onehots, epsilon=0.1, deterministic=False):
        """选择动作
        
        Args:
            obs: (obs_dim,) 观测向量
            action_onehots: (num_legal, 108) 合法动作的onehot编码
            epsilon: 探索率
            deterministic: 是否确定性
            
        Returns:
            action_idx: int, 选择的动作在legal列表中的index
            q_values: (num_legal,) 所有合法动作的Q值
        """
        num_legal = action_onehots.shape[0]
        
        if num_legal == 0:
            return 0, torch.tensor([])
        
        if num_legal == 1:
            with torch.no_grad():
                q_values = self.evaluate_actions(obs, action_onehots)
            return 0, q_values
        
        with torch.no_grad():
            q_values = self.evaluate_actions(obs, action_onehots)
        
        if not deterministic and np.random.random() < epsilon:
            action_idx = np.random.randint(num_legal)
        else:
            action_idx = q_values.argmax().item()
        
        return action_idx, q_values


class DMCNetworkSetV2:
    """4个玩家的Q网络集合 (v2: 动作感知)
    
    设计:
    - 庄家(0,2)共享一个网络: banker_net
    - 闲家(1,3)共享一个网络: xianjia_net  
    - 目标网络用于稳定训练
    """
    
    def __init__(self, obs_dim=764, num_cards=108, hidden_dim=512,
                 action_embed_dim=128, dueling=True, device='cuda'):
        self.device = device
        
        self.banker_net = QNetworkV2(obs_dim, num_cards, hidden_dim, 
                                      action_embed_dim, dueling).to(device)
        self.banker_target = QNetworkV2(obs_dim, num_cards, hidden_dim,
                                         action_embed_dim, dueling).to(device)
        self.banker_target.load_state_dict(self.banker_net.state_dict())
        self.banker_target.eval()
        
        self.xianjia_net = QNetworkV2(obs_dim, num_cards, hidden_dim,
                                       action_embed_dim, dueling).to(device)
        self.xianjia_target = QNetworkV2(obs_dim, num_cards, hidden_dim,
                                          action_embed_dim, dueling).to(device)
        self.xianjia_target.load_state_dict(self.xianjia_net.state_dict())
        self.xianjia_target.eval()
    
    def get_net(self, is_banker):
        """根据角色获取网络
        
        Args:
            is_banker: bool, 是否是庄家（基于game.room.bankers判断）
        """
        return self.banker_net if is_banker else self.xianjia_net
    
    def get_target_net(self, is_banker):
        return self.banker_target if is_banker else self.xianjia_target
    
    def update_targets(self, tau=0.005):
        for param, target_param in zip(self.banker_net.parameters(),
                                        self.banker_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.xianjia_net.parameters(),
                                        self.xianjia_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def hard_update_targets(self):
        self.banker_target.load_state_dict(self.banker_net.state_dict())
        self.xianjia_target.load_state_dict(self.xianjia_net.state_dict())
    
    def save(self, path):
        torch.save({
            'banker': self.banker_net.state_dict(),
            'banker_target': self.banker_target.state_dict(),
            'xianjia': self.xianjia_net.state_dict(),
            'xianjia_target': self.xianjia_target.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.banker_net.load_state_dict(checkpoint['banker'])
        self.banker_target.load_state_dict(checkpoint['banker_target'])
        self.xianjia_net.load_state_dict(checkpoint['xianjia'])
        self.xianjia_target.load_state_dict(checkpoint['xianjia_target'])
        return True
