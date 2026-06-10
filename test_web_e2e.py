# -*- coding: utf-8 -*-
"""
网页服务端到端测试：启动aiohttp服务器 → HTTP API创建机器人对局 → WebSocket观战记录

测试流程：
1. 确认服务已启动
2. 逐局创建机器人房间（/api/start-robot-game）
3. 通过WebSocket连接观战，记录所有消息
4. 等待游戏结束，收集结果
5. 50局后输出分析报告
"""
import sys
import os
import json
import asyncio
import time
import aiohttp
import traceback
from collections import defaultdict

BASE_URL = "http://localhost:9999"
WS_URL = "ws://localhost:9999/ws"
NUM_GAMES = 50
TIMEOUT_PER_GAME = 120  # 每局超时120秒


class WebE2ECollector:
    """网页端到端数据收集器"""
    
    def __init__(self):
        self.games = []
        self.errors = []
        self.all_plays = []
        self.dmc_stats_initial = None
    
    def record_game(self, game_id, result_data, duration, error=None):
        winner = result_data.get('winner_team', 'unknown') if result_data else 'unknown'
        score = result_data.get('score_now', 0) if result_data else 0
        
        game = {
            'game_id': game_id,
            'winner': winner,
            'score': score,
            'result_type': result_data.get('result_type', '') if result_data else '',
            'levels': result_data.get('levels', 0) if result_data else 0,
            'duration': round(duration, 2),
            'error': error,
        }
        self.games.append(game)
        return game


async def run_one_web_game(session, collector, game_id):
    """通过网页API运行一局机器人游戏"""
    room_id = f'WEBTEST_{game_id}'
    start_time = time.time()
    result_data = None
    error = None
    play_count = 0
    
    try:
        # 1. 创建机器人房间 (POST, 参数在query string)
        async with session.post(f'{BASE_URL}/api/start-robot-game?room_id={room_id}') as resp:
            if resp.status != 200:
                error = f"start_robot_game返回 {resp.status}"
                text = await resp.text()
                # 可能房间已存在，等一下重试
                if '已在游戏中' in text or '400' in str(resp.status):
                    await asyncio.sleep(5)
                    async with session.post(f'{BASE_URL}/api/start-robot-game?room_id={room_id}') as resp2:
                        if resp2.status != 200:
                            error = f"重试仍失败: {resp2.status} {await resp2.text()}"
                            duration = time.time() - start_time
                            return collector.record_game(game_id, None, duration, error)
                else:
                    error = f"{error} {text[:100]}"
                    duration = time.time() - start_time
                    return collector.record_game(game_id, None, duration, error)
        
        # 2. WebSocket观战
        ws_url = f'{WS_URL}?room_id={room_id}&seat=-1&name=watcher_{game_id}'
        
        try:
            async with session.ws_connect(ws_url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get('type', '')
                        
                        if msg_type == 'play':
                            play_count += 1
                        
                        if msg_type == 'game_over':
                            result_data = data.get('result', data)
                            break
                    
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        break
                    
                    # 超时检查
                    if time.time() - start_time > TIMEOUT_PER_GAME:
                        error = "WebSocket超时"
                        break
        
        except Exception as e:
            error = f"WebSocket错误: {str(e)[:100]}"
    
    except Exception as e:
        error = f"API错误: {str(e)[:100]}"
    
    duration = time.time() - start_time
    game = collector.record_game(game_id, result_data, duration, error)
    game['play_count'] = play_count
    return game


async def run_web_e2e():
    """运行网页端到端测试"""
    print(f"{'='*60}")
    print(f"  🌐 DMC V9 网页服务端到端测试 - {NUM_GAMES}局")
    print(f"{'='*60}\n")
    
    # 1. 检查服务是否运行
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(BASE_URL) as resp:
                if resp.status != 200:
                    print("❌ 服务未响应")
                    return
    except:
        print("❌ 无法连接到服务器")
        return
    
    print("✅ 服务已启动\n")
    
    collector = WebE2ECollector()
    
    # 获取DMC初始统计
    from rl_dmc.dmc_ai import get_dmc_stats
    initial_stats = get_dmc_stats()
    
    async with aiohttp.ClientSession() as session:
        # 逐局运行（串行，因为房间ID需要唯一）
        for i in range(NUM_GAMES):
            game = await run_one_web_game(session, collector, i)
            
            status = "✓" if not game['error'] else "✗"
            err_msg = f" err={game['error'][:50]}" if game['error'] else ""
            plays = game.get('play_count', 0)
            
            print(f"局{i+1:2d}: {status} winner={game['winner']:8s} "
                  f"score={game['score']:3d} plays={plays:3d} "
                  f"result={game['result_type']:10s} "
                  f"time={game['duration']:.1f}s{err_msg}")
            
            # 局间间隔，确保房间清理
            await asyncio.sleep(0.5)
    
    # ====================================================================
    # 数据分析
    # ====================================================================
    final_stats = get_dmc_stats()
    dmc_new_calls = final_stats['dmc_calls'] - initial_stats['dmc_calls']
    dmc_new_success = final_stats['dmc_success'] - initial_stats['dmc_success']
    dmc_new_fallback = final_stats['fallback_calls'] - initial_stats['fallback_calls']
    
    ok_games = [g for g in collector.games if not g['error']]
    stuck_games = [g for g in collector.games if g['error']]
    
    print(f"\n{'='*60}")
    print(f"  📊 网页端到端测试分析报告")
    print(f"{'='*60}\n")
    
    # 1. 完成率
    print(f"📊 1. 完成率")
    print(f"   成功: {len(ok_games)}/{NUM_GAMES} ({len(ok_games)/NUM_GAMES*100:.1f}%)")
    print(f"   失败: {len(stuck_games)}/{NUM_GAMES}")
    
    if ok_games:
        # 2. 胜负
        banker_wins = sum(1 for g in ok_games if g['winner'] == 'banker')
        xianjia_wins = sum(1 for g in ok_games if g['winner'] == 'xianjia')
        print(f"\n📊 2. 胜负统计")
        print(f"   庄家胜: {banker_wins} ({banker_wins/len(ok_games)*100:.1f}%)")
        print(f"   闲家胜: {xianjia_wins} ({xianjia_wins/len(ok_games)*100:.1f}%)")
        
        # 3. 分数分布
        scores = [g['score'] for g in ok_games]
        print(f"\n📊 3. 闲家得分")
        print(f"   平均: {sum(scores)/len(scores):.1f}, 最小: {min(scores)}, 最大: {max(scores)}")
        
        # 4. 结果类型
        result_types = defaultdict(int)
        for g in ok_games:
            result_types[g['result_type']] += 1
        print(f"\n📊 4. 结果类型")
        for rt, cnt in sorted(result_types.items(), key=lambda x: -x[1]):
            print(f"   {rt}: {cnt}")
    
    # 5. DMC统计
    print(f"\n📊 5. DMC统计（本轮新增）")
    print(f"   调用: {dmc_new_calls}")
    print(f"   成功: {dmc_new_success}")
    print(f"   Fallback: {dmc_new_fallback}")
    if dmc_new_calls > 0:
        print(f"   成功率: {dmc_new_success/dmc_new_calls*100:.1f}%")
    
    # 6. 异常
    if stuck_games:
        print(f"\n📊 6. 异常详情")
        for g in stuck_games[:10]:
            print(f"   局{g['game_id']+1}: {g['error']}")
    
    # 7. Bug总结
    print(f"\n{'='*60}")
    print(f"  🐛 Bug总结")
    print(f"{'='*60}")
    
    bugs = []
    if len(stuck_games) > 0:
        bugs.append(f"网页模式卡住/失败: {len(stuck_games)}/{NUM_GAMES}局")
    if dmc_new_fallback > 0:
        bugs.append(f"DMC Fallback: {dmc_new_fallback}次推理失败")
    
    if not bugs:
        print("  ✅ 未发现Bug")
    else:
        for i, b in enumerate(bugs, 1):
            print(f"  {i}. {b}")
    
    # 保存数据
    output_path = '/home/liurui/card-game-server-upstream/web_e2e_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total': NUM_GAMES,
                'completed': len(ok_games),
                'failed': len(stuck_games),
                'banker_wins': banker_wins if ok_games else 0,
                'xianjia_wins': xianjia_wins if ok_games else 0,
                'avg_score': sum(scores)/len(scores) if ok_games else 0,
                'dmc_stats': final_stats,
            },
            'games': collector.games,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n数据已保存: {output_path}")


if __name__ == '__main__':
    asyncio.run(run_web_e2e())
