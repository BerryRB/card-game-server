# -*- coding: utf-8 -*-
"""4个DMC V9机器人对局e2e测试

直接调用GameRoom引擎（不走网络），4个机器人全部用DMC V9出牌。
测试：
1. 能否完整打完一局（不崩溃）
2. 出牌合法性（无assert失败）
3. 对局结果合理性
4. 连续10局稳定性
"""

import sys
import os
import traceback
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.game_engine import GameRoom
from server.ai import AI
from server.constants import GamePhase


def run_single_game(game_num: int) -> dict:
    """运行一局4 DMC机器人对局，返回结果"""
    room = GameRoom(f'test_{game_num}')
    
    # 添加4个机器人
    for i in range(4):
        room.add_player(i, f'robot_{i}', f'机器人{i+1}')
        room.players[i].is_robot = True
    
    # 开始游戏
    room.start_game()
    
    max_steps = 2000  # 安全上限
    step = 0
    errors = []
    
    while room.phase != GamePhase.GAME_OVER and step < max_steps:
        step += 1
        actions = room.auto_play_current_robot()
        
        # 检查亮主阶段无人亮主
        if room.phase == GamePhase.LIANGZHU:
            all_ready = all(p.ready for p in room.players if p)
            if all_ready and room.liangzhu_player is None:
                room.handle_no_liang()
        
        # 检查动作中的错误
        for action in actions:
            if action.get('type') == 'play':
                result = action.get('result', {})
                if result.get('status') == 'error':
                    errors.append(f"Step {step}: {result.get('msg', 'unknown error')}")
    
    # 验证游戏完成
    game_over = room.phase == GamePhase.GAME_OVER
    game_result = getattr(room, 'game_result', None)
    
    # 统计
    score_now = room.score_now
    bankers = room.bankers or [0, 2]
    banker_score = max(0, 100 - score_now)
    
    return {
        'game_num': game_num,
        'game_over': game_over,
        'steps': step,
        'score_now': score_now,
        'banker_score': banker_score,
        'game_result': game_result,
        'errors': errors,
        'bankers': bankers,
        'error_count': len(errors),
    }


def run_single_game_rules_ai(game_num: int) -> dict:
    """运行一局4规则AI对局（对照组），使用auto_play_current_robot但禁用DMC"""
    room = GameRoom(f'rules_test_{game_num}')
    
    for i in range(4):
        room.add_player(i, f'robot_{i}', f'机器人{i+1}')
        room.players[i].is_robot = True
    
    room.start_game()
    
    max_steps = 2000
    step = 0
    errors = []
    
    # 临时禁用DMC，让auto_play走规则AI路径
    import rl_dmc.dmc_ai as _dm
    _orig_get = _dm.get_dmc_ai
    _dm.get_dmc_ai = lambda *a, **k: None
    
    try:
        while room.phase != GamePhase.GAME_OVER and step < max_steps:
            step += 1
            actions = room.auto_play_current_robot()
            
            if room.phase == GamePhase.LIANGZHU:
                all_ready = all(p.ready for p in room.players if p)
                if all_ready and room.liangzhu_player is None:
                    room.handle_no_liang()
            
            for action in actions:
                if action.get('type') == 'play':
                    result = action.get('result', {})
                    if result.get('status') == 'error':
                        errors.append(f"Step {step}: {result.get('msg', 'unknown error')}")
    finally:
        _dm.get_dmc_ai = _orig_get
    
    game_over = room.phase == GamePhase.GAME_OVER
    return {
        'game_num': game_num,
        'game_over': game_over,
        'steps': step,
        'score_now': room.score_now,
        'errors': errors,
        'error_count': len(errors),
    }


def main():
    print("=" * 60)
    print("  4×DMC V9 机器人对局 e2e 测试")
    print("=" * 60)
    
    # 先检查模型能否加载
    try:
        from rl_dmc.dmc_ai import get_dmc_ai
        ai = get_dmc_ai()
        print("✅ DMC V9 模型加载成功")
    except Exception as e:
        print(f"❌ DMC V9 模型加载失败: {e}")
        traceback.print_exc()
        return
    
    # ===== 测试1: DMC V9 对局 =====
    print("\n--- 测试1: 4×DMC V9 连续10局 ---")
    results = []
    total_errors = 0
    start_time = time.time()
    
    for i in range(10):
        try:
            r = run_single_game(i + 1)
            results.append(r)
            total_errors += r['error_count']
            status = "✅" if r['game_over'] and r['error_count'] == 0 else "❌"
            print(f"  第{i+1}局 {status} steps={r['steps']} "
                  f"闲家={r['score_now']}分 庄家={r['banker_score']}分 "
                  f"errors={r['error_count']}")
            if r['errors']:
                for e in r['errors'][:3]:
                    print(f"    → {e}")
        except Exception as e:
            print(f"  第{i+1}局 ❌ 崩溃: {e}")
            traceback.print_exc()
            results.append({'game_num': i+1, 'game_over': False, 'errors': [str(e)], 'error_count': 1})
            total_errors += 1
    
    elapsed = time.time() - start_time
    
    # 统计
    completed = sum(1 for r in results if r.get('game_over'))
    print(f"\n📊 DMC V9 统计:")
    print(f"  完成率: {completed}/10 ({completed*10}%)")
    print(f"  总错误: {total_errors}")
    print(f"  总耗时: {elapsed:.1f}s")
    
    if completed > 0:
        scores = [r['score_now'] for r in results if r.get('game_over')]
        avg_score = sum(scores) / len(scores)
        xianjia_wins = sum(1 for s in scores if s >= 80)
        banker_wins = sum(1 for s in scores if s < 80)
        print(f"  闲家均分: {avg_score:.1f}")
        print(f"  闲家胜: {xianjia_wins} 庄家胜: {banker_wins}")
    
    # ===== 测试2: 规则AI对照 =====
    print("\n--- 测试2: 4×规则AI 对照组3局 ---")
    rules_results = []
    for i in range(3):
        try:
            r = run_single_game_rules_ai(i + 1)
            rules_results.append(r)
            status = "✅" if r['game_over'] and r['error_count'] == 0 else "❌"
            print(f"  第{i+1}局 {status} steps={r['steps']} "
                  f"闲家={r['score_now']}分 errors={r['error_count']}")
            if r['errors']:
                for e in r['errors'][:3]:
                    print(f"    → {e}")
        except Exception as e:
            print(f"  第{i+1}局 ❌ 崩溃: {e}")
            traceback.print_exc()
    
    # ===== 汇总 =====
    print("\n" + "=" * 60)
    all_ok = completed == 10 and total_errors == 0
    print(f"  最终结果: {'✅ 全部通过' if all_ok else '❌ 存在问题'}")
    print("=" * 60)
    
    return all_ok


if __name__ == '__main__':
    main()
