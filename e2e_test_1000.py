#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端测试：1000局全AI对局，检测卡死、非法出牌、异常崩溃"""

import sys
import os
import time
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.game_engine import GameRoom
from server.constants import GamePhase

MAX_STEPS = 500  # 单局最多500步操作


def play_one_game(game_id: int) -> dict:
    """运行一局完整游戏，返回结果统计"""
    room = GameRoom(room_id=f"e2e_{game_id}")

    for i in range(4):
        room.add_player(i, f"robot_{game_id}_{i}", f"Robot{i}")
        room.players[i].is_robot = True

    result = {
        'game_id': game_id,
        'status': 'ok',
        'error': None,
        'tricks': 0,
        'illegal_plays': 0,
        'illegal_details': [],
        'winner_team': None,
        'score_now': 0,
        'steps': 0,
    }

    trick_count = 0

    try:
        start_result = room.start_game()
        if start_result.get('status') != 'ok':
            result['status'] = 'start_failed'
            result['error'] = str(start_result)
            return result

        for step in range(MAX_STEPS):
            if room.phase == GamePhase.GAME_OVER:
                break

            actions = room.auto_play_current_robot()

            if not actions:
                actions = room.auto_play_robots()

            if not actions:
                # 没有action不一定是卡死，可能是阶段转换的间隙
                # 只要在GAME_OVER就OK
                if room.phase == GamePhase.GAME_OVER:
                    break
                # 检查是否所有手牌出完但没结算
                all_empty = all(
                    room.players[i] is not None and room.players[i].card_count == 0
                    for i in range(4)
                )
                if all_empty and room.phase == GamePhase.PLAYING:
                    result['status'] = 'no_end_settlement'
                    result['error'] = f'All cards played but no GAME_OVER at step={step}'
                    result['tricks'] = trick_count
                    result['steps'] = step
                    return result
                # 否则继续循环，等状态推进
                continue

            result['steps'] = step + 1

            # 检查非法出牌
            for action in actions:
                action_type = action.get('type')
                action_result = action.get('result', {})

                if action_result.get('status') != 'ok':
                    result['illegal_plays'] += 1
                    if len(result['illegal_details']) < 10:
                        result['illegal_details'].append({
                            'type': action_type,
                            'seat': action.get('seat'),
                            'error': action_result.get('message', str(action_result)),
                            'phase': str(room.phase),
                            'step': step,
                        })

                if action_type == 'play':
                    trick_count += 1
        else:
            # 超过MAX_STEPS
            if room.phase != GamePhase.GAME_OVER:
                result['status'] = 'too_many_steps'
                result['error'] = f'Exceeded {MAX_STEPS} steps, phase={room.phase}, turn={getattr(room, "current_turn", None)}'
                result['tricks'] = trick_count
                return result

        # 正常结束
        result['tricks'] = trick_count
        game_result = getattr(room, 'game_result', {})
        result['winner_team'] = game_result.get('winner_team')
        result['score_now'] = game_result.get('score_now', room.score_now)

    except Exception as e:
        result['status'] = 'exception'
        result['error'] = f'{type(e).__name__}: {str(e)}'
        result['tricks'] = trick_count
        result['traceback'] = traceback.format_exc()

    return result


def main():
    total_games = 1000
    print(f"=== 端到端测试：{total_games}局全AI对局 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    stats = Counter()
    total_tricks = 0
    total_illegal = 0
    illegal_by_type = Counter()
    illegal_by_phase = Counter()
    exceptions = []
    stuck_cases = []
    illegal_cases = []
    winner_stats = Counter()
    trick_counts = []
    score_distribution = defaultdict(int)
    result_type_stats = Counter()
    steps_list = []

    t0 = time.time()

    for i in range(total_games):
        result = play_one_game(i)
        stats[result['status']] += 1
        total_tricks += result.get('tricks', 0)
        total_illegal += result.get('illegal_plays', 0)
        trick_counts.append(result.get('tricks', 0))
        steps_list.append(result.get('steps', 0))

        wt = result.get('winner_team')
        if wt:
            winner_stats[wt] += 1

        sn = result.get('score_now')
        if sn is not None:
            score_distribution[sn] += 1

        for detail in result.get('illegal_details', []):
            illegal_by_type[detail['type']] += 1
            illegal_by_phase[detail['phase']] += 1
            if len(illegal_cases) < 10:
                illegal_cases.append(detail)

        if result['status'] == 'exception':
            exceptions.append({
                'game_id': i,
                'error': result['error'],
                'traceback': result.get('traceback', ''),
            })

        if result['status'] in ('too_many_steps', 'no_end_settlement'):
            stuck_cases.append({
                'game_id': i,
                'error': result['error'],
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            err = total_games - stats.get('ok', 0)
            print(f"  [{i+1}/{total_games}] {elapsed:.1f}s, {rate:.1f}局/s, "
                  f"异常={stats.get('exception',0)} 卡死={stats.get('too_many_steps',0)+stats.get('no_end_settlement',0)} "
                  f"非法={total_illegal}")

    elapsed = time.time() - t0

    # ============ 报告 ============
    print("\n" + "=" * 60)
    print(f"端到端测试报告 — {total_games}局")
    print("=" * 60)

    ok_count = stats.get('ok', 0)
    print(f"\n【总体结果】")
    print(f"  总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  速率: {total_games/elapsed:.1f}局/s")
    print(f"  正常完成: {ok_count} ({ok_count/total_games*100:.1f}%)")
    for k, v in sorted(stats.items()):
        if k != 'ok' and v > 0:
            print(f"  {k}: {v}")

    print(f"\n【出牌统计】")
    print(f"  总轮数: {total_tricks}")
    if ok_count > 0:
        print(f"  平均每局轮数: {total_tricks/ok_count:.1f}")
    if trick_counts:
        print(f"  轮数范围: {min(trick_counts)}-{max(trick_counts)}")
    if steps_list and ok_count > 0:
        avg_steps = sum(steps_list) / ok_count
        print(f"  平均每局步骤数: {avg_steps:.1f}")

    print(f"\n【非法出牌】")
    print(f"  总次数: {total_illegal}")
    if illegal_by_type:
        print(f"  按类型: {dict(illegal_by_type)}")
    if illegal_by_phase:
        print(f"  按阶段: {dict(illegal_by_phase)}")
    if illegal_cases:
        print(f"  示例(前10):")
        for c in illegal_cases:
            print(f"    {c}")

    print(f"\n【胜率统计】")
    for team, count in sorted(winner_stats.items()):
        print(f"  {team}: {count}胜 ({count/total_games*100:.1f}%)")

    if score_distribution:
        print(f"\n【闲家得分分布】")
        for score in sorted(score_distribution.keys()):
            cnt = score_distribution[score]
            bar = '#' * min(cnt, 50)
            print(f"  {score:4d}分: {cnt:4d}局 {bar}")

    if stuck_cases:
        print(f"\n【卡死/超限案例】({len(stuck_cases)}个)")
        for c in stuck_cases[:5]:
            print(f"  Game {c['game_id']}: {c['error']}")

    if exceptions:
        print(f"\n【异常案例】({len(exceptions)}个)")
        for c in exceptions[:5]:
            print(f"  Game {c['game_id']}: {c['error']}")
            if c.get('traceback'):
                tb_lines = c['traceback'].strip().split('\n')
                for line in tb_lines[-3:]:
                    print(f"    {line}")

    # 最终判定
    print(f"\n{'='*60}")
    error_count = total_games - ok_count
    if error_count == 0 and total_illegal == 0:
        print("✅ 测试通过！1000局全部正常完成，无卡死/非法出牌/异常")
    elif error_count == 0 and total_illegal > 0:
        print(f"⚠️ 全部完成但有{total_illegal}次非法出牌（AI返回了引擎拒绝的牌）")
    else:
        print(f"❌ 测试未通过：{error_count}局异常，{total_illegal}次非法出牌")

    return error_count == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
