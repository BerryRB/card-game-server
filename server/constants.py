# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - 常量定义"""

from enum import Enum

# 花色
COLOR_NAMES = {
    'a': '黑桃', 'b': '红桃', 'c': '梅花', 'd': '方块', 'z': '王'
}

COLOR_SYMBOLS = {
    'a': '♠', 'b': '♥', 'c': '♣', 'd': '♦', 'z': '🃏'
}

# 牌面名称到rank的映射
RANK_MAP = {
    '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
    '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13,
    'w': 14, 'W': 15
}

RANK_NAMES = {v: k for k, v in RANK_MAP.items() if k not in ('w', 'W')}
RANK_NAMES[14] = '小王'
RANK_NAMES[15] = '大王'

# 升级顺序
LEVEL_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

# 固定主牌名称
FIXED_ZHU_NAMES = {'2', '3', '5'}

# 分牌及对应分值
SCORE_CARDS = {'5': 5, '10': 10, 'K': 10}

# 亮主牌型优先级排名（数值越小越强）
# 多连对(duolian) > 三王(threeking) > 双连对(shuanglian) > 单对子(danlian)
LIANG_TYPE_RANK = {
    'duolian': 1,
    'threeking': 2,
    'shuanglian': 3,
    'danlian': 4,
}

# 出牌类型
class PlayType(str, Enum):
    FUDAN = 'fudan'
    FUDUI = 'fudui'
    FULIAN = 'fulian'
    ZHUDAN = 'zhudan'
    ZHUDUI = 'zhudui'
    ZHULIAN = 'zhulian'

# 游戏阶段
class GamePhase(str, Enum):
    WAITING = 'waiting'
    LIANGZHU = 'liangzhu'
    CHIPAI = 'chipai'
    KOUPAI = 'koupai'
    HUANPAI = 'huanpai'
    PLAYING = 'playing'
    SCORING = 'scoring'
    GAME_OVER = 'game_over'

# 默认端口
DEFAULT_PORT = 9999
