# -*- coding: utf-8 -*-
"""升级(Trump)纸牌游戏 - 网页版服务端入口"""

import asyncio
import json
import sys
import os
from collections import Counter
from aiohttp import web, WSMsgType

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.game_engine import GameRoom
from server.constants import GamePhase
from server.card import card_from_str

# ============ 全局状态 ============
rooms: dict[str, GameRoom] = {}
sid_room: dict[str, str] = {}
connections: dict[str, web.WebSocketResponse] = {}
room_counter = 0


def quiet_connection_reset(loop, context):
    exception = context.get('exception')
    if isinstance(exception, ConnectionResetError):
        return
    loop.default_exception_handler(context)


def get_or_create_room(room_id: str) -> GameRoom:
    if room_id not in rooms:
        rooms[room_id] = GameRoom(room_id)
    return rooms[room_id]


async def reattach_player(room: GameRoom, seat: int, sid: str, client_id: str = ''):
    """Bind a reconnecting browser to an existing player without losing hand cards."""
    player = room.players[seat]
    old_sid = player.sid
    if client_id:
        player.client_id = client_id
    player.sid = sid

    if old_sid and old_sid != sid and old_sid in connections:
        try:
            await connections[old_sid].close()
        except Exception:
            pass


async def broadcast_room(room: GameRoom, message: dict = None):
    """广播房间状态/消息给所有连接的玩家和观战者"""
    # 发给房间内所有玩家
    for i, player in enumerate(room.players):
        if player and player.sid in connections:
            ws = connections[player.sid]
            state = room.get_state(player.sid)
            msg = message or {'action': 'state_update', 'state': state}
            if message is None:
                msg = {'action': 'state_update', 'state': state}
            try:
                await ws.send_str(json.dumps(msg, ensure_ascii=False))
            except Exception:
                pass
    
    # 发给所有观战者（sid_room中有room_id但不在players中的连接）
    player_sids = {p.sid for p in room.players if p}
    for sid, rid in list(sid_room.items()):
        if rid == room.room_id and sid not in player_sids and sid in connections:
            state = room.get_state(None)  # 观战者无sid
            msg = message or {'action': 'state_update', 'state': state}
            try:
                await connections[sid].send_str(json.dumps(msg, ensure_ascii=False))
            except Exception:
                pass


async def process_robots(room: GameRoom):
    """处理机器人的自动操作，逐步进行并广播"""
    max_loops = 50
    for _ in range(max_loops):
        if room.phase == GamePhase.GAME_OVER:
            break

        actions = room.auto_play_current_robot()
        if not actions:
            # 检查是否所有人都不亮主
            if room.phase == GamePhase.LIANGZHU:
                all_ready = all(p.ready for p in room.players if p)
                if all_ready and room.liangzhu_player is None:
                    room.handle_no_liang()
                    await broadcast_room(room)
                    continue
            break

        # 广播状态更新
        await broadcast_room(room)
        
        # 如果有epoch_result（一轮结束），加长延迟让玩家看清
        has_epoch_result = any(
            'result' in a and 'epoch_result' in a.get('result', {})
            for a in actions
        )
        if has_epoch_result:
            await asyncio.sleep(2.5)  # 轮次间更长延迟，让玩家看清4张牌
            # 主动清空出牌区并广播，让前端看到"4张牌→清空"的过渡
            if getattr(room, '_pending_clear_epoch', False):
                room.epoch_cards = []
                room._pending_clear_epoch = False
                await broadcast_room(room)
                await asyncio.sleep(0.5)  # 清空后再短暂等待
        else:
            await asyncio.sleep(1.2)  # 单人出牌延迟

        # 检查游戏结果
        for action in actions:
            if action.get('type') == 'play' and 'result' in action:
                result = action['result']
                if 'game_over' in result:
                    await broadcast_room(room)
                    return


# ============ HTTP路由 ============

async def index(request):
    """首页"""
    base_dir = os.path.dirname(__file__)
    template_path = os.path.join(base_dir, 'templates', 'index.html')
    static_dir = os.path.join(base_dir, 'static')
    css_path = os.path.join(static_dir, 'css', 'style.css')
    js_path = os.path.join(static_dir, 'js', 'game.js')
    version = str(int(max(os.path.getmtime(css_path), os.path.getmtime(js_path))))
    with open(template_path, 'r', encoding='utf-8') as handle:
        html = handle.read().replace('__ASSET_VERSION__', version)
    return web.Response(
        text=html,
        content_type='text/html',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
        },
    )


async def no_cache_static(request):
    """Serve static assets without browser cache so mobile WebViews get fixes."""
    rel_path = request.match_info['path']
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    full_path = os.path.abspath(os.path.join(static_dir, rel_path))
    if not full_path.startswith(os.path.abspath(static_dir) + os.sep):
        raise web.HTTPForbidden()
    if not os.path.isfile(full_path):
        raise web.HTTPNotFound()
    return web.FileResponse(
        full_path,
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
        },
    )


async def create_room_api(request):
    """创建房间API"""
    global room_counter
    room_counter += 1
    room_id = f"{room_counter:04d}"
    room = get_or_create_room(room_id)
    return web.json_response({'room_id': room_id})


async def list_rooms_api(request):
    """列出房间API"""
    result = []
    for rid, room in rooms.items():
        result.append({
            'room_id': rid,
            'player_count': room.player_count,
            'phase': room.phase.value if hasattr(room.phase, 'value') else str(room.phase),
        })
    return web.json_response(result)


async def debug_seed_liangzhu_api(request):
    """Seed the human player's hand with known liangzhu options for local UI testing."""
    if os.environ.get('CHAOYANG_DEBUG_TOOLS') != '1':
        raise web.HTTPNotFound()
    if request.remote not in ('127.0.0.1', '::1', 'localhost'):
        raise web.HTTPForbidden()

    room_id = request.query.get('room_id')
    seat = int(request.query.get('seat', '0'))
    if not room_id or room_id not in rooms or not (0 <= seat <= 3):
        raise web.HTTPBadRequest(text='invalid room_id or seat')

    room = rooms[room_id]
    player = room.players[seat]
    if not player:
        raise web.HTTPBadRequest(text='seat is empty')

    seed_cards = [
        '7-6-b', '7-6-b', '8-7-b', '8-7-b',
        '9-8-a', '9-8-a',
        'w-14-z', 'W-15-z', 'W-15-z',
    ]
    desired = Counter(seed_cards)
    existing = Counter(card.card_type for cards in player.cards_in_hand.values() for card in cards)
    for card_type, count in desired.items():
        for _ in range(max(0, count - existing[card_type])):
            player.add_cards([card_from_str(card_type)])
    player.cards_in_hand = {
        color: sorted(cards, key=lambda c: c.rank)
        for color, cards in sorted(player.cards_in_hand.items())
    }
    await broadcast_room(room)
    return web.json_response({
        'status': 'ok',
        'room_id': room_id,
        'seat': seat,
        'candidates': room.get_state(player.sid).get('liangzhu_available_types', []),
    })


# ============ WebSocket路由 ============

async def websocket_handler(request):
    """WebSocket处理"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    sid = str(id(ws))
    connections[sid] = ws

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                action = data.get('action', '')

                if action == 'join':
                    room_id = data['room_id']
                    name = data.get('name', f'玩家{sid[-4:]}')
                    seat = data.get('seat', -1)
                    client_id = data.get('client_id', '')

                    room = get_or_create_room(room_id)
                    sid_room[sid] = room_id  # 先记录房间关联
                    active_game = room.phase not in (GamePhase.WAITING, GamePhase.GAME_OVER)

                    if client_id:
                        reconnect_seat = -1
                        for i, player in enumerate(room.players):
                            if player and not player.is_robot and getattr(player, 'client_id', '') == client_id:
                                reconnect_seat = i
                                break

                        if reconnect_seat >= 0:
                            await reattach_player(room, reconnect_seat, sid, client_id)
                            await ws.send_str(json.dumps({
                                'action': 'joined',
                                'room_id': room_id,
                                'seat': reconnect_seat,
                                'state': room.get_state(sid),
                            }, ensure_ascii=False))
                            await broadcast_room(room)
                            continue

                    if active_game:
                        reconnect_seat = -1
                        requested = room.players[seat] if 0 <= seat <= 3 else None
                        if requested and not requested.is_robot and requested.name == name and requested.sid not in connections:
                            reconnect_seat = seat
                        else:
                            for i, player in enumerate(room.players):
                                if player and not player.is_robot and player.name == name and player.sid not in connections:
                                    reconnect_seat = i
                                    break

                        if reconnect_seat >= 0:
                            await reattach_player(room, reconnect_seat, sid, client_id)
                            await ws.send_str(json.dumps({
                                'action': 'joined',
                                'room_id': room_id,
                                'seat': reconnect_seat,
                                'state': room.get_state(sid),
                            }, ensure_ascii=False))
                            await broadcast_room(room)
                            continue

                        await ws.send_str(json.dumps({
                            'action': 'joined',
                            'room_id': room_id,
                            'seat': -1,
                            'msg': '牌局已开始，无法中途入座，请新开一局',
                            'state': room.get_state(sid),
                        }, ensure_ascii=False))
                        await broadcast_room(room)
                        continue

                    # 自动选座
                    if seat < 0 or room.players[seat] is not None:
                        found_seat = False
                        for i in range(4):
                            if room.players[i] is None:
                                seat = i
                                found_seat = True
                                break
                        if not found_seat:
                            # 房间已满，作为观战者加入
                            await ws.send_str(json.dumps({
                                'action': 'joined',
                                'room_id': room_id,
                                'seat': -1,  # 观战者
                                'state': room.get_state(None),
                            }, ensure_ascii=False))
                            await broadcast_room(room)
                            continue

                    if not room.add_player(seat, sid, name):
                        await ws.send_str(json.dumps({'action': 'error', 'msg': '入座失败'}))
                        continue
                    if client_id and room.players[seat]:
                        room.players[seat].client_id = client_id

                    sid_room[sid] = room_id
                    await ws.send_str(json.dumps({
                        'action': 'joined',
                        'room_id': room_id,
                        'seat': seat,
                        'state': room.get_state(sid),
                    }, ensure_ascii=False))
                    await broadcast_room(room)

                elif action == 'select_seat':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        new_seat = data.get('seat', -1)
                        old_seat = room.get_seat_by_sid(sid)

                        if 0 <= new_seat <= 3 and room.players[new_seat] is None and old_seat >= 0:
                            # 换座
                            player = room.get_player_by_sid(sid)
                            if player:
                                room.players[old_seat] = None
                                room.players[new_seat] = player
                                player.player_id = new_seat
                                await broadcast_room(room)

                elif action == 'leave':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms.get(room_id)
                        if room:
                            room.remove_player(sid)
                            del sid_room[sid]
                            await broadcast_room(room)

                elif action == 'start_game':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        if room.phase in (GamePhase.WAITING, GamePhase.GAME_OVER):
                            room.start_game()
                            await broadcast_room(room)
                            # 机器人亮主：先让亮主阶段排在前面的机器人决定
                            # 人类玩家通过liangzhu/pass_liangzhu操作
                            await process_robots(room)

                elif action == 'liangzhu':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        seat = room.get_seat_by_sid(sid)
                        card_strs = data.get('cards', [])
                        selected_color = data.get('color')
                        was_chipai = room.phase == GamePhase.CHIPAI
                        result_data = room.handle_liangzhu(seat, card_strs, selected_color)
                        if result_data.get('status') == 'ok' and was_chipai:
                            claim_data = room.handle_chipai_claim(seat)
                            if claim_data.get('status') == 'ok':
                                result_data = {**result_data, **claim_data}
                        resp = {'action': 'liangzhu_result', **result_data}
                        await ws.send_str(json.dumps(resp, ensure_ascii=False))
                        if result_data.get('status') == 'ok':
                            await broadcast_room(room)
                            await process_robots(room)

                elif action == 'pass_liangzhu':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        player = room.get_player_by_sid(sid)
                        if player:
                            if room.phase == GamePhase.CHIPAI:
                                result_data = room.handle_chipai_pass(player.player_id)
                                await ws.send_str(json.dumps({
                                    'action': 'pass_liangzhu_ack',
                                    'seat': player.player_id,
                                    **result_data,
                                }, ensure_ascii=False))
                                if result_data.get('status') == 'ok':
                                    await broadcast_room(room)
                                    await process_robots(room)
                            else:
                                player.ready = True
                                await ws.send_str(json.dumps({
                                    'action': 'pass_liangzhu_ack',
                                    'seat': player.player_id
                                }, ensure_ascii=False))
                                await broadcast_room(room)

                                # 检查是否所有人都ready了（包括机器人）
                                all_ready = all(p.ready for p in room.players if p)
                                if all_ready and room.liangzhu_player is None:
                                    # 所有人都不亮主
                                    room.handle_no_liang()
                                    await broadcast_room(room)
                                    await process_robots(room)
                                else:
                                    # 继续让后续机器人操作
                                    await process_robots(room)

                elif action == 'koupai':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        seat = room.get_seat_by_sid(sid)
                        card_strs = data.get('cards', [])
                        result_data = room.handle_koupai(seat, card_strs)
                        resp = {'action': 'koupai_result', **result_data}
                        await ws.send_str(json.dumps(resp, ensure_ascii=False))
                        if result_data.get('status') == 'ok':
                            await broadcast_room(room)
                            await process_robots(room)

                elif action == 'huanpai':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        seat = room.get_seat_by_sid(sid)
                        accept = bool(data.get('accept', False))
                        card_strs = data.get('cards', [])
                        if accept:
                            pick_strs = data.get('pick_cards') or []
                            if not pick_strs:
                                offer = room.get_state(sid).get('huanpai_offer', [])
                                pick_strs = [card.get('card_type') for card in offer[:len(card_strs)] if card.get('card_type')]
                            result_data = room.handle_shipai(seat, pick_strs, card_strs)
                        else:
                            result_data = room.handle_shipai(seat, [], [])
                        resp = {'action': 'huanpai_result', **result_data}
                        await ws.send_str(json.dumps(resp, ensure_ascii=False))
                        if result_data.get('status') == 'ok':
                            await broadcast_room(room)
                            await process_robots(room)

                elif action == 'play':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        seat = room.get_seat_by_sid(sid)
                        card_strs = data.get('cards', [])
                        result_data = room.handle_play(seat, card_strs)
                        resp = {'action': 'play_result', **result_data}
                        await ws.send_str(json.dumps(resp, ensure_ascii=False))
                        if result_data.get('status') == 'ok':
                            await broadcast_room(room)
                            # 如果这轮4人出完，等2.5秒让玩家看清，再清空出牌区
                            if 'epoch_result' in result_data:
                                await asyncio.sleep(2.5)
                                if getattr(room, '_pending_clear_epoch', False):
                                    room.epoch_cards = []
                                    room._pending_clear_epoch = False
                                    await broadcast_room(room)
                                    await asyncio.sleep(0.5)
                            await process_robots(room)

                elif action == 'get_state':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        await ws.send_str(json.dumps({
                            'action': 'state',
                            'state': room.get_state(sid),
                        }, ensure_ascii=False))

                elif action == 'suggest_play':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        seat = room.get_seat_by_sid(sid)
                        suggestions = room.suggest_play(seat)
                        await ws.send_str(json.dumps({
                            'action': 'suggest_play_result',
                            'suggestions': suggestions,
                        }, ensure_ascii=False))

                elif action == 'chat':
                    room_id = sid_room.get(sid)
                    if room_id:
                        room = rooms[room_id]
                        player = room.get_player_by_sid(sid)
                        if player:
                            chat_msg = data.get('msg', '')
                            await broadcast_room(room, {
                                'action': 'chat',
                                'seat': player.player_id,
                                'name': player.name,
                                'msg': chat_msg,
                            })

            elif msg.type == WSMsgType.ERROR:
                break

    except Exception as e:
        print(f"Handler error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if sid in connections:
            del connections[sid]
        if sid in sid_room:
            room_id = sid_room[sid]
            room = rooms.get(room_id)
            if room:
                if room.phase in (GamePhase.WAITING, GamePhase.GAME_OVER):
                    room.remove_player(sid)
                else:
                    player = room.get_player_by_sid(sid)
                    if player:
                        player.sid = ''
                await broadcast_room(room)
            del sid_room[sid]

    return ws


# ============ 启动 ============

async def start_robot_game_api(request):
    """一键创建4机器人房间并开始游戏（用于录屏/测试）
    
    所有玩家都设为机器人，方便全自动运行。
    观战者可通过WebSocket连接加入房间观看。
    """
    room_id = request.query.get('room_id', 'REC1')
    room = get_or_create_room(room_id)
    if room.phase != GamePhase.WAITING:
        return web.json_response({'error': '房间已在游戏中'}, status=400)
    
    # 填补空位为机器人 + 把现有玩家也设为机器人
    for i in range(4):
        if room.players[i] is None:
            room.add_player(i, f'robot_{i}', f'机器人{i+1}')
        room.players[i].is_robot = True
    
    # 开始游戏
    room.start_game()
    
    # 启动机器人自动操作
    asyncio.create_task(_robot_game_loop(room))
    
    return web.json_response({'room_id': room_id, 'status': 'started', 'phase': room.phase.value})


async def _robot_game_loop(room: GameRoom):
    """机器人游戏循环（后台任务）"""
    while room.phase != GamePhase.GAME_OVER:
        actions = room.auto_play_current_robot()
        
        # 亮主阶段：检查无人亮主
        if room.phase == GamePhase.LIANGZHU:
            all_ready = all(p.ready for p in room.players if p)
            if all_ready and room.liangzhu_player is None:
                room.handle_no_liang()
        
        # 广播状态给所有连接的观众
        await broadcast_room(room)
        
        await asyncio.sleep(0.5)  # 每0.5秒一步
    
    # 游戏结束，广播最终状态
    await broadcast_room(room)
    print(f"Room {room.room_id} game over: {room.game_result}")


async def on_startup(app):
    asyncio.get_running_loop().set_exception_handler(quiet_connection_reset)
    print("=" * 50)
    print("  升级(Trump)纸牌游戏 - 网页版服务端")
    print("=" * 50)
    print(f"  访问地址: http://0.0.0.0:9999")
    print("=" * 50)


def create_app():
    app = web.Application()

    # HTTP路由
    app.router.add_get('/', index)
    app.router.add_get('/api/rooms', list_rooms_api)
    app.router.add_post('/api/rooms', create_room_api)
    app.router.add_post('/api/debug/seed-liangzhu', debug_seed_liangzhu_api)
    app.router.add_post('/api/start-robot-game', start_robot_game_api)
    app.router.add_get('/static/{path:.*}', no_cache_static)

    # WebSocket
    app.router.add_get('/ws', websocket_handler)

    app.on_startup.append(on_startup)

    return app


if __name__ == '__main__':
    app = create_app()
    web.run_app(app, host='0.0.0.0', port=9999)
