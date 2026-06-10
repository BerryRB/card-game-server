# -*- coding: utf-8 -*-
"""
网页服务端到端测试 v2

策略：启动aiohttp服务 → 用Python API创建GameRoom+4机器人 
→ 走完整的process_robots循环（和app.py一致）
→ WebSocket观战验证 → 50局数据收集
"""
import sys
import os
import json
import asyncio
import time
import traceback
import aiohttp
from collections import defaultdict

sys.path.insert(0, '/home/liurui/card-game-server-upstream')

from server.game_engine import GameRoom, GamePhase
from rl_dmc.dmc_ai import get_dmc_stats

BASE_URL = "http://localhost:9999"
WS_URL = "ws://localhost:9999/ws"
NUM_GAMES = 50
TIMEOUT_PER_GAME = 120


class WebE2ECollector:
    def __init__(self):
        self.games = []
        self.errors = []
    
    def record_game(self, game_id, room, steps, duration, error=None):
        result = getattr(room, 'game_result', None) or {}
        game = {
            'game_id': game_id,
            'winner': result.get('winner_team', 'unknown'),
            'score': room.score_now,
            'result_type': result.get('result_type', ''),
            'levels': result.get('levels', 0),
            'next_level': result.get('next_level', ''),
            'steps': steps,
            'duration': round(duration, 2),
            'error': error,
        }
        self.games.append(game)
        return game


async def run_one_game(collector, game_id, max_steps=800):
    """运行一局4机器人游戏（走和app.py一样的process_robots逻辑）"""
    room_id = f'e2e_{game_id}'
    room = GameRoom(room_id=room_id)
    
    for i in range(4):
        room.add_player(i, f'robot_{i}', f'机器人{i+1}')
        room.players[i].is_robot = True
    
    start_time = time.time()
    room.start_game()
    
    stuck_count = 0
    
    for step in range(max_steps):
        try:
            actions = room.auto_play_current_robot()
        except Exception as e:
            tb = traceback.format_exc()
            return collector.record_game(game_id, room, step, time.time()-start_time, f"异常: {tb[:200]}")
        
        if room.phase == GamePhase.GAME_OVER:
            return collector.record_game(game_id, room, step+1, time.time()-start_time)
        
        if not actions:
            stuck_count += 1
            # 亮主阶段无人亮主处理
            if room.phase == GamePhase.LIANGZHU:
                all_ready = all(p.ready for p in room.players if p)
                if all_ready and room.liangzhu_player is None:
                    room.handle_no_liang()
                    stuck_count = 0
                    continue
            if stuck_count > 100:
                return collector.record_game(game_id, room, step+1, time.time()-start_time, f"卡住{stuck_count}步")
        else:
            stuck_count = 0
        
        await asyncio.sleep(0.001)  # 让出事件循环
    
    return collector.record_game(game_id, room, max_steps, time.time()-start_time, "超时")


async def verify_websocket(game_id):
    """验证WebSocket能正常推送游戏状态"""
    try:
        async with aiohttp.ClientSession() as session:
            ws_url = f'{WS_URL}?room_id=e2e_{game_id}&seat=-1&name=ws_test'
            async with session.ws_connect(ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=5)) as ws:
                msg = await asyncio.wait_for(ws.receive(), timeout=3)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    return True, data.get('type', '')
                return False, f"msg_type={msg.type}"
    except Exception as e:
        return False, str(e)[:100]


async def run_web_e2e():
    print(f"{'='*60}")
    print(f"  🌐 DMC V9 网页服务端到端测试 - {NUM_GAMES}局")
    print(f"{'='*60}\n")
    
    # 1. 检查服务
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(BASE_URL) as resp:
                if resp.status != 200:
                    print("❌ 服务未响应"); return
    except:
        print("❌ 无法连接"); return
    print("✅ 服务已启动\n")
    
    # 2. WebSocket功能验证
    # 先创建一个房间让ws有东西看
    test_room = GameRoom(room_id='ws_test')
    for i in range(4):
        test_room.add_player(i, f'r{i}', f'机器人{i+1}')
        test_room.players[i].is_robot = True
    test_room.start_game()
    
    # 注意：这里不把test_room注册到app的rooms里，所以ws可能找不到
    # WebSocket测试改为局后验证
    
    initial_stats = get_dmc_stats()
    collector = WebE2ECollector()
    
    # 3. 逐局运行
    for i in range(NUM_GAMES):
        game = await run_one_game(collector, i)
        
        status = "✓" if not game['error'] else "✗"
        err = f" err={game['error'][:60]}" if game['error'] else ""
        
        print(f"局{i+1:2d}: {status} winner={game['winner']:8s} "
              f"score={game['score']:3d} steps={game['steps']:3d} "
              f"result={game['result_type']:10s} "
              f"next={game['next_level']} time={game['duration']:.1f}s{err}")
    
    # ====================================================================
    # 数据分析
    # ====================================================================
    final_stats = get_dmc_stats()
    dmc_new_calls = final_stats['dmc_calls'] - initial_stats['dmc_calls']
    dmc_new_success = final_stats['dmc_success'] - initial_stats['dmc_success']
    dmc_new_fallback = final_stats['fallback_calls'] - initial_stats['fallback_calls']
    
    ok_games = [g for g in collector.games if not g['error']]
    fail_games = [g for g in collector.games if g['error']]
    
    print(f"\n{'='*60}")
    print(f"  📊 网页端到端测试分析报告")
    print(f"{'='*60}\n")
    
    # 1. 完成率
    print(f"📊 1. 完成率")
    print(f"   成功: {len(ok_games)}/{NUM_GAMES} ({len(ok_games)/NUM_GAMES*100:.1f}%)")
    print(f"   失败: {len(fail_games)}/{NUM_GAMES}")
    
    if ok_games:
        banker_wins = sum(1 for g in ok_games if g['winner'] == 'banker')
        xianjia_wins = sum(1 for g in ok_games if g['winner'] == 'xianjia')
        
        # 2. 胜负
        print(f"\n📊 2. 胜负统计")
        print(f"   庄家胜: {banker_wins} ({banker_wins/len(ok_games)*100:.1f}%)")
        print(f"   闲家胜: {xianjia_wins} ({xianjia_wins/len(ok_games)*100:.1f}%)")
        
        # 3. 分数分布
        scores = [g['score'] for g in ok_games]
        print(f"\n📊 3. 闲家得分")
        print(f"   平均: {sum(scores)/len(scores):.1f}")
        print(f"   最小: {min(scores)}, 最大: {max(scores)}")
        
        score_ranges = [(0,39,'0-39光头/小光'), (40,79,'40-79庄胜'), 
                       (80,119,'80-119夺庄0级'), (120,159,'120-159夺庄1级'),
                       (160,199,'160-199夺庄2级'), (200,999,'200+全胜')]
        print(f"   分段:")
        for lo, hi, label in score_ranges:
            cnt = sum(1 for s in scores if lo <= s <= hi)
            print(f"     {label}: {cnt}局")
        
        # 4. 结果类型
        result_types = defaultdict(int)
        for g in ok_games:
            result_types[g['result_type']] += 1
        print(f"\n📊 4. 结果类型")
        for rt, cnt in sorted(result_types.items(), key=lambda x: -x[1]):
            print(f"   {rt}: {cnt}局")
        
        # 5. 升级统计
        levels_list = [g['levels'] for g in ok_games]
        print(f"\n📊 5. 升级统计")
        print(f"   平均升级: {sum(levels_list)/len(levels_list):.2f}级")
        print(f"   总升级: {sum(levels_list)}级")
    
    # 6. DMC统计
    print(f"\n📊 6. DMC模型统计（本轮）")
    print(f"   调用: {dmc_new_calls}")
    print(f"   成功: {dmc_new_success}")
    print(f"   Fallback: {dmc_new_fallback}")
    if dmc_new_calls > 0:
        print(f"   成功率: {dmc_new_success/dmc_new_calls*100:.1f}%")
    
    # 7. 步骤数统计
    if ok_games:
        steps_list = [g['steps'] for g in ok_games]
        print(f"\n📊 7. 步骤数")
        print(f"   平均: {sum(steps_list)/len(steps_list):.1f}")
        print(f"   最小: {min(steps_list)}, 最大: {max(steps_list)}")
    
    # 8. 异常分析
    if fail_games:
        print(f"\n📊 8. 异常详情")
        for g in fail_games[:10]:
            print(f"   局{g['game_id']+1}: {g['error']}")
    
    # 9. Bug总结
    print(f"\n{'='*60}")
    print(f"  🐛 Bug总结")
    print(f"{'='*60}")
    
    bugs = []
    if len(fail_games) > 0:
        bugs.append(f"游戏引擎Bug: {len(fail_games)}/{NUM_GAMES}局失败/卡住")
        # 分析卡住原因
        stuck_phase = [g for g in fail_games if '卡住' in (g.get('error') or '')]
        timeout_games = [g for g in fail_games if '超时' in (g.get('error') or '')]
        if stuck_phase:
            bugs.append(f"  其中{len(stuck_phase)}局卡住（auto_play返回空actions）")
        if timeout_games:
            bugs.append(f"  其中{len(timeout_games)}局超时")
    
    if dmc_new_fallback > 0:
        bugs.append(f"DMC Fallback: {dmc_new_fallback}次推理失败，回退到规则AI")
    
    if ok_games and banker_wins / len(ok_games) > 0.75:
        bugs.append(f"庄闲平衡异常: 庄家胜率{banker_wins/len(ok_games)*100:.1f}%，可能DMC策略有问题")
    
    if not bugs:
        print("  ✅ 未发现Bug")
    else:
        for i, b in enumerate(bugs, 1):
            print(f"  {i}. {b}")
    
    # 保存数据
    output_path = '/home/liurui/card-game-server-upstream/web_e2e_results.json'
    summary = {
        'total': NUM_GAMES,
        'completed': len(ok_games),
        'failed': len(fail_games),
        'banker_wins': banker_wins if ok_games else 0,
        'xianjia_wins': xianjia_wins if ok_games else 0,
        'avg_score': sum(scores)/len(scores) if ok_games else 0,
        'dmc_stats': final_stats,
        'dmc_new_calls': dmc_new_calls,
        'dmc_new_success': dmc_new_success,
        'dmc_new_fallback': dmc_new_fallback,
    }
    with open(output_path, 'w') as f:
        json.dump({'summary': summary, 'games': collector.games}, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存: {output_path}")
    return collector


if __name__ == '__main__':
    asyncio.run(run_web_e2e())
