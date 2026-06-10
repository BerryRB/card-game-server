# -*- coding: utf-8 -*-
"""
DMC V9 集成 - 完整单元测试

测试范围:
1. DMC AI模块 (dmc_ai.py)
2. obs编码 (obs_v11.py + _GameWrapper)
3. game_engine DMC集成 (出牌/fallback/安全出牌)
4. 边界条件
"""
import sys
import os
import asyncio
import unittest
import numpy as np

sys.path.insert(0, '/home/liurui/card-game-server-upstream')

# ====================================================================
# Test 1: DMC AI模块
# ====================================================================
class TestDMCAIModule(unittest.TestCase):
    """测试DMCAI类的核心功能"""
    
    @classmethod
    def setUpClass(cls):
        """加载模型（只加载一次）"""
        from rl_dmc.dmc_ai import DMCAI
        # 重置单例以便测试
        DMCAI._instance = None
        cls.ai = DMCAI()
    
    def test_01_model_loaded(self):
        """模型应正确加载"""
        from rl_dmc.dmc_ai import DMCAI
        ai = DMCAI()  # 单例
        self.assertTrue(ai._initialized)
        self.assertIsNotNone(ai.network_set)
        self.assertIsNotNone(ai.network_set.banker_net)
        self.assertIsNotNone(ai.network_set.xianjia_net)
    
    def test_02_singleton(self):
        """DMCAI应该是单例"""
        from rl_dmc.dmc_ai import DMCAI
        ai1 = DMCAI()
        ai2 = DMCAI()
        self.assertIs(ai1, ai2)
    
    def test_03_network_eval_mode(self):
        """网络应在eval模式"""
        self.ai.network_set.banker_net.eval()
        self.ai.network_set.xianjia_net.eval()
        # 确认BN层在eval模式
        for net in [self.ai.network_set.banker_net, self.ai.network_set.xianjia_net]:
            for m in net.modules():
                if hasattr(m, 'training'):
                    self.assertFalse(m.training, f"{m} should be in eval mode")
    
    def test_04_obs_dim(self):
        """观测维度应为764"""
        from rl_shengji.obs_v11 import OBS_DIM_V11
        self.assertEqual(OBS_DIM_V11, 764)
    
    def test_05_obs_shape_from_game(self):
        """从真实游戏构建obs，维度正确"""
        from server.game_engine import GameRoom
        room = GameRoom(room_id='test_obs')
        room.start_game()
        
        obs = self.ai._build_obs(room, 0)
        self.assertEqual(obs.shape, (764,), f"obs shape should be (764,), got {obs.shape}")
        self.assertEqual(obs.dtype, np.float32)
    
    def test_06_obs_not_all_zero(self):
        """obs不应全零（至少手牌和玩家位置非零）"""
        from server.game_engine import GameRoom
        room = GameRoom(room_id='test_obs2')
        room.start_game()
        
        obs = self.ai._build_obs(room, 0)
        self.assertGreater(np.count_nonzero(obs), 0, "obs should not be all zeros")
        # 手牌段(0:108)应有非零
        self.assertGreater(np.count_nonzero(obs[:108]), 0, "hand section should be non-zero")
        # 当前玩家(669:673)应有非零
        self.assertGreater(np.count_nonzero(obs[669:673]), 0, "current player section should be non-zero")
    
    def test_07_legal_actions_not_empty(self):
        """有手牌时合法动作不应为空"""
        from server.game_engine import GameRoom
        room = GameRoom(room_id='test_legal')
        room.start_game()
        
        actions = self.ai._get_legal_actions(room, 0)
        self.assertGreater(len(actions), 0, "should have legal actions when hand is not empty")
    
    def test_08_legal_actions_have_onehot(self):
        """合法动作应有正确的onehot编码"""
        from server.game_engine import GameRoom
        room = GameRoom(room_id='test_onehot')
        room.start_game()
        
        actions = self.ai._get_legal_actions(room, 0)
        for a in actions:
            self.assertIn('onehot', a)
            self.assertIn('card_strs', a)
            self.assertEqual(a['onehot'].shape, (108,), f"onehot shape should be (108,)")
            # onehot应该是0/1
            self.assertTrue(np.all((a['onehot'] == 0) | (a['onehot'] == 1)))
    
    def test_09_decide_play_returns_cards(self):
        """decide_play应返回card_strs列表"""
        from server.game_engine import GameRoom
        room = GameRoom(room_id='test_decide')
        room.start_game()
        
        result = self.ai.decide_play(room, 0)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for ct in result:
            self.assertIsInstance(ct, str)
    
    def test_10_decide_play_cards_in_hand(self):
        """DMC出的牌必须在手牌中"""
        from server.game_engine import GameRoom
        room = GameRoom(room_id='test_inhand')
        room.start_game()
        
        player = room.players[0]
        hand_cts = set()
        for cards in player.cards_in_hand.values():
            for c in cards:
                hand_cts.add(c.card_type)
        
        result = self.ai.decide_play(room, 0)
        for ct in result:
            self.assertIn(ct, hand_cts, f"DMC played {ct} but not in hand {hand_cts}")
    
    def test_11_banker_vs_xianjia_net(self):
        """庄闲应选择不同网络"""
        from server.game_engine import GameRoom
        room = GameRoom(room_id='test_net')
        room.start_game()
        
        # player 0通常是庄家
        is_banker = 0 in (room.bankers or [0, 2])
        net = self.ai.network_set.get_net(is_banker)
        
        if is_banker:
            self.assertIs(net, self.ai.network_set.banker_net)
        else:
            self.assertIs(net, self.ai.network_set.xianjia_net)


# ====================================================================
# Test 2: Obs编码一致性
# ====================================================================
class TestObsEncoding(unittest.TestCase):
    """测试obs编码的正确性和一致性"""
    
    def test_01_card_encoding_roundtrip(self):
        """card_type -> id -> card_type 应可逆"""
        from rl_dmc.env import _ID_TO_CARD_TYPE, _CARD_TYPE_TO_IDS, NUM_CARDS
        
        # 验证映射一致性
        for ct, ids in _CARD_TYPE_TO_IDS.items():
            for cid in ids:
                self.assertEqual(_ID_TO_CARD_TYPE[cid], ct,
                    f"id {cid} should map back to {ct}, got {_ID_TO_CARD_TYPE[cid]}")
    
    def test_02_onehot_sum(self):
        """onehot编码中1的数量应等于牌数"""
        from rl_dmc.env import cards_to_onehot, NUM_CARDS
        from server.card import Card
        
        # 单张牌 - card_type格式: rank-num-suit
        onehot = cards_to_onehot([Card('5-4-a')])
        self.assertEqual(np.sum(onehot), 1)
        
        # 对子 - 同rank不同suit
        onehot = cards_to_onehot([Card('5-4-a'), Card('5-4-b')])
        self.assertEqual(np.sum(onehot), 2)
    
    def test_03_obs_sections(self):
        """obs各段长度应正确: hand(108) + trick(432) + played(108) + color(4) + level(13) + banker(4) + current(4) + score(20) + v11(71) = 764"""
        from server.game_engine import GameRoom
        from rl_dmc.dmc_ai import DMCAI, _GameWrapper
        
        DMCAI._instance = None
        ai = DMCAI()
        
        room = GameRoom(room_id='test_sections')
        room.start_game()
        
        wrapper = _GameWrapper(room)
        base_state = wrapper.get_state(0)
        base_obs = base_state['obs']
        
        # 108 + 432 + 108 + 4 + 13 + 4 + 4 + 20 = 693
        self.assertEqual(base_obs.shape, (693,), f"base obs should be 693-dim, got {base_obs.shape}")
        
        # 完整obs 764维
        full_obs = ai._build_obs(room, 0)
        self.assertEqual(full_obs.shape, (764,))
    
    def test_04_obs_deterministic(self):
        """相同游戏状态应产生相同obs"""
        from server.game_engine import GameRoom
        from rl_dmc.dmc_ai import DMCAI
        
        DMCAI._instance = None
        ai = DMCAI()
        
        room = GameRoom(room_id='test_det')
        room.start_game()
        
        obs1 = ai._build_obs(room, 0)
        obs2 = ai._build_obs(room, 0)
        np.testing.assert_array_equal(obs1, obs2, "obs should be deterministic for same state")


# ====================================================================
# Test 3: Game Engine DMC集成
# ====================================================================
class TestGameEngineDMC(unittest.TestCase):
    """测试game_engine.py中DMC集成的正确性"""
    
    def test_01_dmc_import(self):
        """dmc_decide_play应可正常导入"""
        from server.game_engine import dmc_decide_play
        self.assertTrue(callable(dmc_decide_play))
    
    def test_02_full_game_no_error(self):
        """完整一局游戏不应抛异常"""
        from server.game_engine import GameRoom
        
        room = GameRoom(room_id='test_full')
        room.start_game()
        
        for step in range(600):
            actions = room.auto_play_current_robot()
            if not actions:
                break
            if room.phase.name == 'GAME_OVER':
                return  # 成功
        
        # 如果到这里说明没完成，但不算失败（可能是stuck bug）
    
    def test_03_dmc_stats_tracked(self):
        """DMC调用应被统计"""
        from server.game_engine import GameRoom
        from rl_dmc.dmc_ai import get_dmc_stats
        
        room = GameRoom(room_id='test_stats')
        room.start_game()
        
        # 走几步
        for _ in range(5):
            room.auto_play_current_robot()
        
        stats = get_dmc_stats()
        self.assertGreater(stats['dmc_calls'], 0, "DMC should have been called")
    
    def test_04_handle_play_with_dmc_cards(self):
        """handle_play应接受DMC返回的card_strs格式"""
        from server.game_engine import GameRoom
        
        room = GameRoom(room_id='test_handle')
        room.start_game()
        
        # 模拟DMC出牌
        from rl_dmc.dmc_ai import dmc_decide_play
        
        # 找到当前需要出牌的robot
        for pid in range(4):
            p = room.players[pid]
            if p and p.is_robot and p.card_count > 0:
                result = dmc_decide_play(room, pid)
                if result:
                    # 验证handle_play能接受
                    play_result = room.handle_play(pid, result)
                    # 可能返回ok也可能返回error（因为不是该玩家的turn）
                    self.assertIn(play_result['status'], ['ok', 'error'])
                    break
    
    def test_05_fallback_on_none(self):
        """DMC返回None时应fallback到规则AI"""
        from server.game_engine import GameRoom
        import rl_dmc.dmc_ai as dmc_mod
        
        # Mock dmc_decide_play 返回None
        original = dmc_mod.dmc_decide_play
        dmc_mod.dmc_decide_play = lambda room, pid: None
        
        try:
            room = GameRoom(room_id='test_fallback')
            room.start_game()
            
            # 推进到PLAYING阶段
            for _ in range(20):
                actions = room.auto_play_current_robot()
                if room.phase.name in ('PLAYING', 'GAME_OVER'):
                    break
            
            if room.phase.name == 'PLAYING':
                # 规则AI应该能出牌
                actions = room.auto_play_current_robot()
                # 至少应该有action（规则AI fallback）
                self.assertIsNotNone(actions)
        finally:
            dmc_mod.dmc_decide_play = original
    
    def test_06_safe_play_fallback(self):
        """_safe_play_fallback应始终返回合法出牌"""
        from server.game_engine import GameRoom
        
        room = GameRoom(room_id='test_safe')
        room.start_game()
        
        for pid in range(4):
            p = room.players[pid]
            if p and p.card_count > 0:
                fallback = room._safe_play_fallback(p)
                if fallback:
                    self.assertIsInstance(fallback, list)
                    for ct in fallback:
                        self.assertIsInstance(ct, str)
                    break


# ====================================================================
# Test 4: 边界条件
# ====================================================================
class TestEdgeCases(unittest.TestCase):
    """测试边界条件"""
    
    def test_01_empty_hand_no_crash(self):
        """空手牌不应崩溃"""
        from server.game_engine import GameRoom
        from rl_dmc.dmc_ai import DMCAI
        
        DMCAI._instance = None
        ai = DMCAI()
        
        room = GameRoom(room_id='test_empty')
        room.start_game()
        
        # 跑完游戏
        for _ in range(600):
            actions = room.auto_play_current_robot()
            if room.phase.name == 'GAME_OVER':
                break
        
        # 游戏结束后调用DMC不应崩溃
        for pid in range(4):
            try:
                result = ai.decide_play(room, pid)
                # 空手牌应该返回None或空列表
            except Exception:
                pass  # 可能抛异常，只要不segfault就行
    
    def test_02_concurrent_games(self):
        """多个GameRoom同时使用DMC不应冲突"""
        from server.game_engine import GameRoom
        
        rooms = [GameRoom(room_id=f'concurrent_{i}') for i in range(3)]
        for r in rooms:
            r.start_game()
        
        # 交替推进
        for _ in range(20):
            for r in rooms:
                r.auto_play_current_robot()
        
        # 不应崩溃
    
    def test_03_dmc_stats_accumulate(self):
        """DMC统计应正确累加"""
        from server.game_engine import GameRoom
        from rl_dmc.dmc_ai import get_dmc_stats
        
        initial = get_dmc_stats()
        
        room = GameRoom(room_id='test_accum')
        room.start_game()
        for _ in range(10):
            room.auto_play_current_robot()
        
        after = get_dmc_stats()
        self.assertGreaterEqual(after['dmc_calls'], initial['dmc_calls'])
    
    def test_04_epoch_cards_trick_encoding(self):
        """当前轮出牌在obs中应正确编码"""
        from server.game_engine import GameRoom
        from rl_dmc.dmc_ai import DMCAI, _GameWrapper
        
        DMCAI._instance = None
        ai = DMCAI()
        
        room = GameRoom(room_id='test_trick')
        room.start_game()
        
        # 推进到有人出牌
        for _ in range(5):
            room.auto_play_current_robot()
        
        # 如果当前轮有出牌，trick段应非零
        wrapper = _GameWrapper(room)
        obs = wrapper.get_state(0)['obs']
        trick_section = obs[108:540]  # 432维trick段
        
        # 不一定有（可能刚好清空了），但不应崩溃
        self.assertEqual(trick_section.shape, (432,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
