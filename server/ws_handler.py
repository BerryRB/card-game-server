# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - WebSocket服务端"""

from __future__ import annotations
import asyncio
import json
import time
from typing import Optional
from server.game_engine import GameRoom

# 房间管理
rooms: dict[str, GameRoom] = {}
# sid -> room_id 映射
sid_room: dict[str, str] = {}


def get_or_create_room(room_id: str) -> GameRoom:
    if room_id not in rooms:
        rooms[room_id] = GameRoom(room_id)
    return rooms[room_id]


async def handle_connection(websocket, path=None):
    """处理WebSocket连接"""
    sid = id(websocket)
    room_id = None

    try:
        async for message in websocket:
            data = json.loads(message)
            action = data.get('action', '')
            result = {}

            if action == 'join':
                room_id = data['room_id']
                name = data.get('name', f'玩家{sid % 10000}')
                seat = data.get('seat', -1)

                room = get_or_create_room(room_id)
                # 自动选座
                if seat < 0 or room.players[seat] is not None:
                    for i in range(4):
                        if room.players[i] is None:
                            seat = i
                            break

                if not room.add_player(seat, str(sid), name):
                    result = {'action': 'error', 'msg': '入座失败，座位已被占'}
                else:
                    sid_room[str(sid)] = room_id
                    result = {
                        'action': 'joined',
                        'room_id': room_id,
                        'seat': seat,
                        'state': room.get_state(str(sid)),
                    }
                    # 广播房间更新
                    await broadcast_room(room, exclude=str(sid))

            elif action == 'leave':
                if str(sid) in sid_room:
                    room_id = sid_room[str(sid)]
                    room = rooms.get(room_id)
                    if room:
                        seat = room.remove_player(str(sid))
                        del sid_room[str(sid)]
                        result = {'action': 'left', 'seat': seat}
                        await broadcast_room(room)

            elif action == 'ready':
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    player = room.get_player_by_sid(str(sid))
                    if player:
                        player.ready = True
                        result = {'action': 'ready', 'seat': player.player_id}
                        await broadcast_room(room)

                        # 检查是否所有人都准备好了
                        if room.is_full and all(
                            p.ready for p in room.players if p
                        ):
                            start_result = room.start_game()
                            result = {
                                'action': 'game_started',
                                'state': room.get_state(str(sid)),
                            }
                            await broadcast_room(room)

                            # 自动处理机器人
                            await auto_handle_robots(room)

            elif action == 'start_game':
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    if room.phase.value == 'waiting':
                        start_result = room.start_game()
                        result = {
                            'action': 'game_started',
                            'state': room.get_state(str(sid)),
                        }
                        await broadcast_room(room)
                        await auto_handle_robots(room)

            elif action == 'liangzhu':
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    seat = room.get_seat_by_sid(str(sid))
                    card_strs = data.get('cards', [])
                    result_data = room.handle_liangzhu(seat, card_strs)
                    result = {'action': 'liangzhu_result', **result_data}
                    await broadcast_room(room)
                    await auto_handle_robots(room)

            elif action == 'koupai':
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    seat = room.get_seat_by_sid(str(sid))
                    card_strs = data.get('cards', [])
                    result_data = room.handle_koupai(seat, card_strs)
                    result = {'action': 'koupai_result', **result_data}
                    await broadcast_room(room)
                    await auto_handle_robots(room)

            elif action == 'play':
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    seat = room.get_seat_by_sid(str(sid))
                    card_strs = data.get('cards', [])
                    result_data = room.handle_play(seat, card_strs)
                    result = {'action': 'play_result', **result_data}
                    await broadcast_room(room)
                    await auto_handle_robots(room)

            elif action == 'get_state':
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    result = {
                        'action': 'state',
                        'state': room.get_state(str(sid)),
                    }

            elif action == 'pass_liangzhu':
                """不亮主"""
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    # 标记玩家已选择不亮
                    player = room.get_player_by_sid(str(sid))
                    if player:
                        player.ready = True
                        result = {'action': 'pass_liangzhu', 'seat': player.player_id}

                        # 如果所有人都pass了，随机定主
                        all_passed = all(p.ready for p in room.players if p and not p.is_robot)
                        if all_passed:
                            # 随机选底牌一张定主
                            if room.hole_cards:
                                for hc in room.hole_cards:
                                    if not hc.is_joker:
                                        room.now_color = hc.color
                                        break
                                else:
                                    room.now_color = None  # 无主
                            room._set_bankers()
                            room.phase = 'koupai'
                            await broadcast_room(room)
                            await auto_handle_robots(room)

            elif action == 'chat':
                room_id = sid_room.get(str(sid))
                if room_id:
                    room = rooms[room_id]
                    player = room.get_player_by_sid(str(sid))
                    if player:
                        msg = data.get('msg', '')
                        result = {
                            'action': 'chat',
                            'seat': player.player_id,
                            'name': player.name,
                            'msg': msg,
                        }
                        await broadcast_room(room)

            if result:
                await safe_send(websocket, json.dumps(result, ensure_ascii=False))

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # 清理
        if str(sid) in sid_room:
            rid = sid_room[str(sid)]
            room = rooms.get(rid)
            if room:
                room.remove_player(str(sid))
                await broadcast_room(room)
            del sid_room[str(sid)]


async def auto_handle_robots(room: GameRoom):
    """自动处理机器人的操作"""
    max_loops = 20  # 防止无限循环
    for _ in range(max_loops):
        actions = room.auto_play_current_robot()
        if not actions:
            break
        await asyncio.sleep(0.3)  # 机器人延迟
        await broadcast_room(room)

        # 检查是否游戏结束
        if room.phase.value == 'game_over':
            break


async def broadcast_room(room: GameRoom, exclude: str = ''):
    """广播房间状态给所有玩家"""
    import websockets as ws_lib

    # 这里需要在实际运行时关联websocket对象
    # 由外部调用者处理
    pass


async def safe_send(websocket, message: str):
    """安全发送消息"""
    try:
        await websocket.send(message)
    except Exception:
        pass
