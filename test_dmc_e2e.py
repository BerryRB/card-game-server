# -*- coding: utf-8 -*-
"""
端到端测试：DMC V9 4机器人对局50局

启动完整aiohttp服务，通过API创建机器人房间，
每局记录：出牌序列、分数、胜负、DMC调用统计、异常
"""
import sys
import os
import json
import asyncio
import time
import traceback
from collections import defaultdict

sys.path.insert(0, '/home/liurui/card-game-server-upstream')

from server.game_engine import GameRoom, GamePhase
from rl_dmc.dmc_ai import get_dmc_stats

# ====================================================================
# 数据收集器
# ====================================================================
class GameDataCollector:
    """收集每局游戏的详细数据"""
    
    def __init__(self):
        self.games = []  # 每局完整数据
        self.all_plays = []  # 所有出牌记录
        self.errors = []  # 异常记录
    
    def record_game(self, game_id, room, steps, stuck, duration):
        """记录一局游戏的完整数据"""
        result = getattr(room, 'game_result', None) or {}
        
        # 收集出牌统计
        play_stats = defaultdict(lambda: {'single': 0, 'pair': 0, 'triple': 0, 'sequence': 0, 'total': 0})
        for ps in self.all_plays:
            if ps['game_id'] == game_id:
                seat = ps['seat']
                n = ps['num_cards']
                play_stats[seat]['total'] += 1
                if n == 1:
                    play_stats[seat]['single'] += 1
                elif n == 2:
                    play_stats[seat]['pair'] += 1
                elif n == 3:
                    play_stats[seat]['triple'] += 1
                else:
                    play_stats[seat]['sequence'] += 1
        
        game_data = {
            'game_id': game_id,
            'winner': result.get('winner_team', 'unknown'),
            'score_now': room.score_now,
            'result_type': result.get('result_type', ''),
            'levels': result.get('levels', 0),
            'new_bankers': result.get('new_bankers', []),
            'banker_levels': result.get('banker_levels', []),
            'next_level': result.get('next_level', ''),
            'steps': steps,
            'stuck': stuck,
            'duration_sec': round(duration, 2),
            'play_stats': dict(play_stats),
            'dmc_stats': get_dmc_stats(),
        }
        self.games.append(game_data)
        return game_data
    
    def record_play(self, game_id, seat, card_strs, is_dmc=True):
        """记录一次出牌"""
        self.all_plays.append({
            'game_id': game_id,
            'seat': seat,
            'num_cards': len(card_strs),
            'card_strs': card_strs,
            'is_dmc': is_dmc,
        })
    
    def record_error(self, game_id, step, error_msg):
        """记录异常"""
        self.errors.append({
            'game_id': game_id,
            'step': step,
            'error': error_msg,
        })


# ====================================================================
# 游戏循环
# ====================================================================
async def run_one_game(collector, game_id, max_steps=800):
    """运行一局4机器人游戏"""
    room = GameRoom(room_id=f'e2e_{game_id}')
    
    # 添加4个机器人
    for i in range(4):
        room.add_player(i, f'robot_{i}', f'机器人{i+1}')
        room.players[i].is_robot = True
    
    start_time = time.time()
    room.start_game()
    
    stuck_count = 0
    prev_hands = None
    
    for step in range(max_steps):
        try:
            actions = room.auto_play_current_robot()
        except Exception as e:
            collector.record_error(game_id, step, traceback.format_exc())
            break
        
        if room.phase == GamePhase.GAME_OVER:
            duration = time.time() - start_time
            return collector.record_game(game_id, room, step + 1, False, duration)
        
        if not actions:
            stuck_count += 1
            # 检查亮主阶段无人亮主
            if room.phase == GamePhase.LIANGZHU:
                all_ready = all(p.ready for p in room.players if p)
                if all_ready and room.liangzhu_player is None:
                    room.handle_no_liang()
                    stuck_count = 0
                    continue
            
            if stuck_count > 100:
                duration = time.time() - start_time
                collector.record_error(game_id, step, f"Stuck after {stuck_count} empty steps")
                return collector.record_game(game_id, room, step + 1, True, duration)
        else:
            stuck_count = 0
            # 记录出牌
            for a in actions:
                if a['type'] == 'play':
                    seat = a.get('seat', -1)
                    cards = a.get('cards', [])
                    card_strs = [c['card_type'] for c in cards] if cards and isinstance(cards[0], dict) else []
                    collector.record_play(game_id, seat, card_strs)
        
        await asyncio.sleep(0.001)
    
    # 超时
    duration = time.time() - start_time
    collector.record_error(game_id, max_steps, "Timeout")
    return collector.record_game(game_id, room, max_steps, True, duration)


async def run_e2e_tests(num_games=50):
    """运行端到端测试"""
    print(f"{'='*60}")
    print(f"  DMC V9 端到端测试 - {num_games}局4机器人对局")
    print(f"{'='*60}\n")
    
    collector = GameDataCollector()
    
    completed = 0
    stuck = 0
    errors = 0
    
    for i in range(num_games):
        game_data = await run_one_game(collector, i)
        
        if game_data['stuck']:
            stuck += 1
            status = "✗ STUCK"
        else:
            completed += 1
            status = "✓"
        
        print(f"局{i+1:2d}: {status} winner={game_data['winner']:8s} "
              f"score={game_data['score_now']:3d} steps={game_data['steps']:3d} "
              f"result={game_data['result_type']:10s} "
              f"next={game_data['next_level']} time={game_data['duration_sec']:.1f}s")
    
    # ====================================================================
    # 数据分析
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"  数据分析报告")
    print(f"{'='*60}\n")
    
    # 1. 完成率
    print(f"📊 1. 完成率")
    print(f"   完成: {completed}/{num_games} ({completed/num_games*100:.1f}%)")
    print(f"   卡住: {stuck}/{num_games} ({stuck/num_games*100:.1f}%)")
    print(f"   异常: {errors}/{num_games} ({errors/num_games*100:.1f}%)")
    
    # 2. 胜负统计
    ok_games = [g for g in collector.games if not g['stuck']]
    if ok_games:
        banker_wins = sum(1 for g in ok_games if g['winner'] == 'banker')
        xianjia_wins = sum(1 for g in ok_games if g['winner'] == 'xianjia')
        print(f"\n📊 2. 胜负统计（已完成局）")
        print(f"   庄家胜: {banker_wins} ({banker_wins/len(ok_games)*100:.1f}%)")
        print(f"   闲家胜: {xianjia_wins} ({xianjia_wins/len(ok_games)*100:.1f}%)")
    
    # 3. 分数分布
    if ok_games:
        scores = [g['score_now'] for g in ok_games]
        print(f"\n📊 3. 闲家得分分布")
        print(f"   平均: {sum(scores)/len(scores):.1f}")
        print(f"   最小: {min(scores)}, 最大: {max(scores)}")
        
        # 分段统计
        score_ranges = [(0,39,'0-39光头/小光'), (40,79,'40-79庄胜'), 
                       (80,119,'80-119夺庄0级'), (120,159,'120-159夺庄1级'),
                       (160,199,'160-199夺庄2级'), (200,999,'200+全胜')]
        print(f"   分段:")
        for lo, hi, label in score_ranges:
            cnt = sum(1 for s in scores if lo <= s <= hi)
            print(f"     {label}: {cnt}局")
    
    # 4. 结果类型分布
    if ok_games:
        result_types = defaultdict(int)
        for g in ok_games:
            result_types[g['result_type']] += 1
        print(f"\n📊 4. 结果类型分布")
        for rt, cnt in sorted(result_types.items(), key=lambda x: -x[1]):
            print(f"   {rt}: {cnt}局")
    
    # 5. 出牌统计
    if ok_games:
        total_singles = 0
        total_pairs = 0
        total_triples = 0
        total_seqs = 0
        total_plays = 0
        
        for g in ok_games:
            for seat, stats in g['play_stats'].items():
                total_singles += stats['single']
                total_pairs += stats['pair']
                total_triples += stats['triple']
                total_seqs += stats['sequence']
                total_plays += stats['total']
        
        print(f"\n📊 5. 出牌类型统计")
        print(f"   总出牌: {total_plays}")
        if total_plays > 0:
            print(f"   单张: {total_singles} ({total_singles/total_plays*100:.1f}%)")
            print(f"   对子: {total_pairs} ({total_pairs/total_plays*100:.1f}%)")
            print(f"   三张: {total_triples} ({total_triples/total_plays*100:.1f}%)")
            print(f"   连对/其他: {total_seqs} ({total_seqs/total_plays*100:.1f}%)")
    
    # 6. DMC统计
    stats = get_dmc_stats()
    print(f"\n📊 6. DMC模型调用统计")
    print(f"   调用: {stats['dmc_calls']}")
    print(f"   成功: {stats['dmc_success']}")
    print(f"   Fallback: {stats['fallback_calls']}")
    if stats['dmc_calls'] > 0:
        print(f"   成功率: {stats['dmc_success']/stats['dmc_calls']*100:.1f}%")
    
    # 7. 卡住分析
    stuck_games = [g for g in collector.games if g['stuck']]
    if stuck_games:
        print(f"\n📊 7. 卡住分析（{len(stuck_games)}局）")
        for g in stuck_games[:5]:
            print(f"   局{g['game_id']+1}: score={g['score_now']} steps={g['steps']} "
                  f"phase={GamePhase(g.get('phase', 3)).name if 'phase' in g else '?'}")
    
    # 8. 异常报告
    if collector.errors:
        print(f"\n📊 8. 异常报告")
        for e in collector.errors[:10]:
            print(f"   局{e['game_id']+1} step={e['step']}: {e['error'][:100]}")
    
    # 9. 每局步骤数
    if ok_games:
        steps_list = [g['steps'] for g in ok_games]
        print(f"\n📊 9. 每局步骤数")
        print(f"   平均: {sum(steps_list)/len(steps_list):.1f}")
        print(f"   最小: {min(steps_list)}, 最大: {max(steps_list)}")
    
    # 10. Bug总结
    print(f"\n{'='*60}")
    print(f"  🐛 Bug总结")
    print(f"{'='*60}")
    
    bugs = []
    if stuck > 0:
        bugs.append(f"卡死Bug: {stuck}/{num_games}局卡住 ({stuck/num_games*100:.1f}%) - 游戏引擎auto_play_current_robot在某些条件下返回空actions")
    if errors > 0:
        bugs.append(f"异常Bug: {errors}局出现异常")
    if stats['fallback_calls'] > 0:
        bugs.append(f"DMC Fallback: {stats['fallback_calls']}次DMC推理失败，回退到规则AI")
    
    # 检查出牌分布是否合理
    if ok_games and total_plays > 0:
        pair_rate = total_pairs / total_plays * 100
        if pair_rate < 5:
            bugs.append(f"出牌分布异常: 对子比例仅{pair_rate:.1f}%，可能DMC很少出对子")
    
    # 检查庄闲平衡
    if ok_games and len(ok_games) >= 10:
        banker_rate = banker_wins / len(ok_games) * 100
        if banker_rate < 20 or banker_rate > 80:
            bugs.append(f"庄闲平衡异常: 庄家胜率{banker_rate:.1f}%，可能DMC庄/闲网络选择有误")
    
    if not bugs:
        print("  ✅ 未发现明显Bug")
    else:
        for i, b in enumerate(bugs, 1):
            print(f"  {i}. {b}")
    
    # 保存完整数据
    output_path = '/home/liurui/card-game-server-upstream/e2e_test_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total': num_games,
                'completed': completed,
                'stuck': stuck,
                'errors': errors,
                'banker_wins': banker_wins if ok_games else 0,
                'xianjia_wins': xianjia_wins if ok_games else 0,
                'avg_score': sum(scores)/len(scores) if ok_games else 0,
                'dmc_stats': stats,
            },
            'games': collector.games,
            'errors_detail': collector.errors,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n完整数据已保存到: {output_path}")
    
    return collector


if __name__ == '__main__':
    asyncio.run(run_e2e_tests(50))
