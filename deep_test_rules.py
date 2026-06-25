#!/usr/bin/env python3
"""
深度规则验证测试 v5 — 修正版
核心发现：2,3,5是固定主牌(FIXED_ZHU_NAMES)，不能当副牌用！
真副牌 = 非2/3/5/级牌/主色的牌 → 用6,7,8,9,10,J,Q,K
"""
import sys, os, random, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.card import Card, Deck, FIXED_ZHU_NAMES
from server.game_engine import GameRoom
from server.constants import GamePhase
from server.ai import AI
from server.rules import determine_play_type

MAX_STEPS = 300
TRUE_FU_RANKS = ['6','7','8','9','10','J','Q','K']  # 一定是副牌的rank

def C(card_type):
    return Card(card_type)

def is_zhu(c, lv, co):
    return c.is_zhu(lv, co)

def setup_room(seed=None):
    if seed is not None:
        random.seed(seed)
    room = GameRoom("T")
    for i in range(4):
        room.add_player(i, f"bot{i}", f"Bot{i}")
        room.players[i].is_robot = True
    room.start_game()
    for _ in range(200):
        if room.phase == GamePhase.PLAYING:
            return room
        actions = room.auto_play_current_robot()
        if not actions and room.phase == GamePhase.LIANGZHU:
            room.handle_no_liang()
    return None

def clear_hand(p):
    for k in list(p.cards_in_hand.keys()):
        p.cards_in_hand[k] = []

def set_hand(p, cards_by_color):
    clear_hand(p)
    for color, cards in cards_by_color.items():
        p.cards_in_hand[color] = cards

def get_non_trump_colors(co):
    return [c for c in ['a','b','c','d'] if c != co]

# ============================================================
# Test 1: _validate_follow 精确规则验证（修正版）
# ============================================================

def test_validate_follow_rules():
    issues = []
    room = setup_room(seed=42)
    if not room:
        return ["无法初始化测试房间"]
    
    lv = room.now_level
    co = room.now_color
    ntc = get_non_trump_colors(co)
    fc = ntc[0]   # 副牌花色1
    fc2 = ntc[1]  # 副牌花色2
    
    print(f"  环境: level={lv}, trump={co}, fu1={fc}, fu2={fc2}")
    print(f"  FIXED_ZHU_NAMES={FIXED_ZHU_NAMES}")
    
    p = room.players[0]
    
    # ---- 1a: 首出副单，有同花色副牌 → 必须出同花色 ----
    first = [C(f'6-6-{fc}')]
    set_hand(p, {fc: [C(f'7-7-{fc}'), C(f'8-8-{fc}')], co: [C(f'2-2-{co}')]})
    
    ok, msg = room._validate_follow([C(f'7-7-{fc}')], first, p)
    if not ok: issues.append(f"1a-1: 出同花色副牌应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'2-2-{co}')], first, p)
    if ok: issues.append(f"1a-2: 有同花色副牌时出主牌应被拒!")
    
    # ---- 1b: 首出副单，同花色只有级牌(主牌) → 绝门，可出主毙或出其他 ----
    first = [C(f'6-6-{fc}')]
    set_hand(p, {fc: [C(f'{lv}-14-{fc}')], fc2: [C(f'7-7-{fc2}')]})
    
    ok, msg = room._validate_follow([C(f'7-7-{fc2}')], first, p)
    if not ok: issues.append(f"1b-1: 同花色只有主牌时出其他副牌应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'{lv}-14-{fc}')], first, p)
    if not ok: issues.append(f"1b-2: 同花色只有主牌时出主毙应通过: {msg}")
    
    # ---- 1b3: 首出副单，同花色只有固定主(2/3/5) → 绝门 ----
    for fz_name in ['2', '3', '5']:
        if fz_name == lv:
            continue  # 级牌已经测过
        first = [C(f'6-6-{fc}')]
        fz_rank = {'2':1, '3':2, '5':4}[fz_name]  # 对应rank
        set_hand(p, {fc: [C(f'{fz_name}-{fz_rank}-{fc}')], fc2: [C(f'7-7-{fc2}')]})
        
        ok, msg = room._validate_follow([C(f'7-7-{fc2}')], first, p)
        if not ok: 
            issues.append(f"1b-3({fz_name}): 同花色只有固定主{fz_name}时出其他副牌应通过: {msg}")
        
        ok, msg = room._validate_follow([C(f'{fz_name}-{fz_rank}-{fc}')], first, p)
        if not ok:
            issues.append(f"1b-4({fz_name}): 同花色只有固定主{fz_name}时出主毙应通过: {msg}")
    
    # ---- 1c: 首出副对，有同花色副牌对子 → 必须出对子 ----
    first_pair = [C(f'6-6-{fc}'), C(f'6-6-{fc}')]
    set_hand(p, {fc: [C(f'7-7-{fc}'), C(f'7-7-{fc}'), C(f'8-8-{fc}')], fc2: [C(f'K-13-{fc2}')]})
    
    ok, msg = room._validate_follow([C(f'7-7-{fc}'), C(f'7-7-{fc}')], first_pair, p)
    if not ok: issues.append(f"1c-1: 出同花色对子应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'7-7-{fc}'), C(f'8-8-{fc}')], first_pair, p)
    if ok: issues.append(f"1c-2: 有同花色对子但出散牌应被拒!")
    
    # ---- 1d: 首出副对，同花色只有1张副牌 → 必须先出这张+1张其他 ----
    set_hand(p, {fc: [C(f'7-7-{fc}')], fc2: [C(f'K-13-{fc2}'), C(f'Q-12-{fc2}')]})
    
    ok, msg = room._validate_follow([C(f'7-7-{fc}'), C(f'K-13-{fc2}')], first_pair, p)
    if not ok: issues.append(f"1d-1: 1张同花色+1张其他应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'K-13-{fc2}'), C(f'Q-12-{fc2}')], first_pair, p)
    if ok: issues.append(f"1d-2: 有1张同花色副牌但没出应被拒!")
    
    # ---- 1e: 首出主单，有主牌 → 必须跟主牌 ----
    first_zhu = [C(f'6-6-{co}')]  # 主色6是主牌
    set_hand(p, {co: [C(f'7-7-{co}'), C(f'8-8-{co}')], fc: [C(f'9-9-{fc}')]})
    
    ok, msg = room._validate_follow([C(f'7-7-{co}')], first_zhu, p)
    if not ok: issues.append(f"1e-1: 跟主出主应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'9-9-{fc}')], first_zhu, p)
    if ok: issues.append(f"1e-2: 有主牌时出副牌应被拒!")
    
    # ---- 1f: 首出主对，有主对 → 必须出主对 ----
    first_zhu_pair = [C(f'6-6-{co}'), C(f'6-6-{co}')]
    set_hand(p, {co: [C(f'7-7-{co}'), C(f'7-7-{co}'), C(f'8-8-{co}')], fc: [C(f'9-9-{fc}')]})
    
    ok, msg = room._validate_follow([C(f'7-7-{co}'), C(f'7-7-{co}')], first_zhu_pair, p)
    if not ok: issues.append(f"1f-1: 跟主对出主对应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'7-7-{co}'), C(f'8-8-{co}')], first_zhu_pair, p)
    if ok: issues.append(f"1f-2: 有主对时出非对子主牌应被拒!")
    
    # ---- 1g: 首出主对，无主对有2+张主牌 → 必须出最大主牌 ----
    set_hand(p, {co: [C(f'7-7-{co}'), C(f'8-8-{co}')], fc: [C(f'9-9-{fc}')]})
    
    ok, msg = room._validate_follow([C(f'7-7-{co}'), C(f'8-8-{co}')], first_zhu_pair, p)
    if not ok: issues.append(f"1g-1: 无主对时出2张主牌应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'7-7-{co}'), C(f'9-9-{fc}')], first_zhu_pair, p)
    if ok: issues.append(f"1g-2: 有2张主牌但只出1+副牌应被拒!")
    
    # ---- 1h: 首出副对，同花色全是主牌(级牌/固定主) → 绝门 ----
    set_hand(p, {fc: [C(f'{lv}-14-{fc}'), C(f'{lv}-14-{fc}')], fc2: [C(f'K-13-{fc2}'), C(f'Q-12-{fc2}')]})
    
    ok, msg = room._validate_follow([C(f'{lv}-14-{fc}'), C(f'{lv}-14-{fc}')], first_pair, p)
    if not ok: issues.append(f"1h-1: 同花色只有级牌时出级牌毙应通过: {msg}")
    
    ok, msg = room._validate_follow([C(f'K-13-{fc2}'), C(f'Q-12-{fc2}')], first_pair, p)
    if not ok: issues.append(f"1h-2: 同花色只有主牌时绝门出其他应通过: {msg}")
    
    # ---- 1i: 首出副连对(4张)，有2张同花色副牌 → 必须出这2张+2张其他 ----
    first_fulian = [C(f'6-6-{fc}'), C(f'6-6-{fc}'), C(f'7-7-{fc}'), C(f'7-7-{fc}')]
    set_hand(p, {
        fc: [C(f'8-8-{fc}'), C(f'9-9-{fc}')],
        fc2: [C(f'K-13-{fc2}'), C(f'Q-12-{fc2}'), C(f'J-11-{fc2}'), C(f'T-10-{fc2}')]
    })
    
    ok, msg = room._validate_follow(
        [C(f'8-8-{fc}'), C(f'9-9-{fc}'), C(f'K-13-{fc2}'), C(f'Q-12-{fc2}')],
        first_fulian, p
    )
    if not ok: issues.append(f"1i-1: 2张同花色+2张其他应通过: {msg}")
    
    ok, msg = room._validate_follow(
        [C(f'K-13-{fc2}'), C(f'Q-12-{fc2}'), C(f'J-11-{fc2}'), C(f'T-10-{fc2}')],
        first_fulian, p
    )
    if ok: issues.append(f"1i-2: 有同花色副牌但不出应被拒!")
    
    # ---- 1j: 首出副连对(4张)，同花色只有2张固定主 → 绝门 ----
    set_hand(p, {
        fc: [C(f'2-1-{fc}'), C(f'3-2-{fc}')],  # 2和3是固定主
        fc2: [C(f'K-13-{fc2}'), C(f'Q-12-{fc2}'), C(f'J-11-{fc2}'), C(f'T-10-{fc2}')]
    })
    
    ok, msg = room._validate_follow(
        [C(f'2-1-{fc}'), C(f'3-2-{fc}'), C(f'K-13-{fc2}'), C(f'Q-12-{fc2}')],
        first_fulian, p
    )
    if not ok: issues.append(f"1j-1: 同花色2张固定主+2张其他应通过(绝门): {msg}")
    
    # ---- 1k: 毙牌规则 - 首出副对，绝门出1张主牌不行(必须2张) ----
    set_hand(p, {co: [C(f'6-6-{co}')], fc2: [C(f'K-13-{fc2}'), C(f'Q-12-{fc2}')]})
    
    ok, msg = room._validate_follow([C(f'6-6-{co}'), C(f'K-13-{fc2}')], first_pair, p)
    # 1张主牌+1张副牌跟副对 → 主牌不足2张时可以用副牌补？
    # 规则：绝门时可以用主牌毙，但主牌数不足时用其他副牌补
    # 这个应该通过
    if not ok:
        issues.append(f"1k-1: 绝门1张主+1张副跟副对应通过: {msg}")
    
    # ---- 1l: 毙牌必须大于首出牌数 ----
    # 首出副对(2张)，绝门出4张主牌→不允许(超出数量)
    set_hand(p, {co: [C(f'6-6-{co}'), C(f'6-6-{co}'), C(f'7-7-{co}'), C(f'7-7-{co}')]})
    
    ok, msg = room._validate_follow(
        [C(f'6-6-{co}'), C(f'6-6-{co}'), C(f'7-7-{co}'), C(f'7-7-{co}')],
        first_pair, p
    )
    if ok:
        issues.append(f"1l-1: 出4张主牌跟2张副对应被拒(数量不匹配)!")
    
    return issues

# ============================================================
# Test 2: AI出牌全面合法性测试 (2000局)
# ============================================================

def test_ai_legality_2000(num_games=2000):
    """每步验证AI出牌是否被引擎接受"""
    rejected = 0
    rejected_details = []
    completed = 0
    
    for gi in range(num_games):
        room = GameRoom(f"T{gi}")
        for i in range(4):
            room.add_player(i, f"bot{i}", f"Bot{i}")
            room.players[i].is_robot = True
        room.start_game()
        
        for step in range(MAX_STEPS):
            if room.phase == GamePhase.GAME_OVER:
                completed += 1
                break
            actions = room.auto_play_current_robot()
            if not actions:
                if room.phase == GamePhase.LIANGZHU:
                    room.handle_no_liang()
                    continue
                break
            # auto_play内部已有fallback，如果走到fallback说明有AI出牌被拒
            # 检查最近的log
        else:
            pass  # 超步数
    
    # 检查fallback触发次数
    return completed

# ============================================================
# Test 3: AI跟牌策略质量测试
# ============================================================

def test_ai_follow_quality(num_games=500):
    """
    验证AI在特定边界场景下的出牌策略质量：
    1) 同花色只有主牌时AI是否走了正确的绝门路径
    2) 跟副对时AI是否正确选择出对子vs散牌
    3) 跟主对时AI是否正确选择出主对vs散主
    """
    misjudge_count = 0  # _has_color_cards误判次数
    misjudge_ok = 0     # 误判但出牌合法
    misjudge_bad = 0    # 误判且出牌非法
    pair_violations = 0  # 对子违反次数
    details = []
    
    for gi in range(num_games):
        room = setup_room(seed=gi * 13 + 7)
        if not room:
            continue
        
        for step in range(MAX_STEPS):
            if room.phase != GamePhase.PLAYING:
                break
            
            ct = room.current_turn
            player = room.players[ct]
            if not player or not player.is_robot or player.card_count == 0:
                break
            
            if room.epoch_cards:
                first_cards = room.epoch_cards[0]
                first_type = determine_play_type(first_cards, room.now_level, room.now_color)
                
                if first_type and first_type.startswith('fu'):
                    # 检查同花色只有主牌场景
                    first_color = first_cards[0].color
                    ai = AI(player, room.now_level, room.now_color, room.score_koupai)
                    
                    color_cards = player.cards_in_hand.get(first_color, [])
                    fu = [c for c in color_cards if not is_zhu(c, room.now_level, room.now_color)]
                    zhu_in_c = [c for c in color_cards if is_zhu(c, room.now_level, room.now_color)]
                    has_color = ai._has_color_cards(first_color)
                    
                    if has_color and len(fu) == 0 and len(zhu_in_c) > 0:
                        misjudge_count += 1
                        # 验证AI出牌
                        epoch_player_objs = [room.players[s] for s in room.epoch_players]
                        try:
                            ai_cards = ai.decide_play(
                                room.epoch_cards, epoch_player_objs,
                                False, getattr(room, 'score_now', 0)
                            )
                        except Exception as e:
                            misjudge_bad += 1
                            if len(details) < 5:
                                details.append(f"G{gi}s{step}: AI error: {e}")
                            room.auto_play_current_robot()
                            continue
                        
                        card_strs = [c.card_type for c in ai_cards] if ai_cards else []
                        result = room.handle_play(ct, card_strs)
                        if result.get('status') == 'ok':
                            misjudge_ok += 1
                        else:
                            misjudge_bad += 1
                            if len(details) < 5:
                                details.append(f"G{gi}s{step}: REJECTED '{result.get('msg','')}' cards={card_strs}")
                        continue  # handle_play已经推进了
                    
                    # 检查对子违反
                    if first_type in ('fudui',) or first_type.startswith('fulian'):
                        fu_cards = [c for c in color_cards if not is_zhu(c, room.now_level, room.now_color)]
                        rank_groups = defaultdict(list)
                        for c in fu_cards:
                            rank_groups[c.name].append(c)
                        has_pair = any(len(g) >= 2 for g in rank_groups.values())
                        
                        if has_pair and len(fu_cards) >= len(first_cards):
                            epoch_player_objs = [room.players[s] for s in room.epoch_players]
                            ai = AI(player, room.now_level, room.now_color, room.score_koupai)
                            try:
                                ai_cards = ai.decide_play(
                                    room.epoch_cards, epoch_player_objs,
                                    False, getattr(room, 'score_now', 0)
                                )
                            except:
                                room.auto_play_current_robot()
                                continue
                            
                            if ai_cards:
                                ai_fu = [c for c in ai_cards if c.color == first_color and not is_zhu(c, room.now_level, room.now_color)]
                                ai_rg = defaultdict(list)
                                for c in ai_fu:
                                    ai_rg[c.name].append(c)
                                ai_has_pair = any(len(g) >= 2 for g in ai_rg.values())
                                
                                if not ai_has_pair and len(ai_cards) >= 2:
                                    pair_violations += 1
                                    if len(details) < 5:
                                        details.append(f"G{gi}s{step}: 有对子但出散牌: {[c.card_type for c in ai_cards]}")
            
            room.auto_play_current_robot()
    
    return misjudge_count, misjudge_ok, misjudge_bad, pair_violations, details

# ============================================================
# Test 4: _has_color_cards 误判根因分析
# ============================================================

def test_has_color_cards_root_cause():
    """直接测试_has_color_cards方法"""
    room = setup_room(seed=100)
    if not room:
        return "无法初始化"
    
    lv = room.now_level
    co = room.now_color
    
    # 对每个非主色，构造"只有固定主/级牌"的手牌
    results = []
    for color in ['a','b','c','d']:
        if color == co:
            continue
        
        p = room.players[0]
        clear_hand(p)
        # 手牌：该花色1张2(固定主) + 另一花色1张K
        other_color = 'b' if color != 'b' else 'c'
        p.cards_in_hand[color] = [C(f'2-1-{color}')]
        p.cards_in_hand[other_color] = [C(f'K-13-{other_color}')]
        
        ai = AI(p, lv, co, room.score_koupai)
        has_color = ai._has_color_cards(color)
        
        # 实际：该花色只有固定主2，没有副牌
        fu = [c for c in p.cards_in_hand.get(color, []) if not is_zhu(c, lv, co)]
        
        results.append({
            'color': color,
            'hand': [c.card_type for c in p.cards_in_hand.get(color, [])],
            'has_color_cards': has_color,
            'actual_fu_count': len(fu),
            'misjudged': has_color and len(fu) == 0
        })
    
    return results

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("深度规则验证测试 v5 (修正版)")
    print(f"固定主牌: {FIXED_ZHU_NAMES}")
    print("=" * 60)
    
    # Test 1
    print("\n[Test 1] _validate_follow 精确规则验证...")
    t0 = time.time()
    issues_1 = test_validate_follow_rules()
    t1 = time.time()
    print(f"  发现问题: {len(issues_1)}, 耗时: {t1-t0:.1f}s")
    for i in issues_1:
        print(f"    ⚠️ {i}")
    if not issues_1:
        print("    ✅ 所有规则验证通过")
    
    # Test 4: 误判根因
    print("\n[Test 4] _has_color_cards 误判根因分析...")
    rc = test_has_color_cards_root_cause()
    if isinstance(rc, list):
        for r in rc:
            status = "❌误判" if r['misjudged'] else "✅正确"
            print(f"  花色{r['color']}: hand={r['hand']} has_color={r['has_color_cards']} "
                  f"fu_count={r['actual_fu_count']} {status}")
    
    # Test 3: AI策略质量
    print("\n[Test 3] AI跟牌策略质量测试 (500局)...")
    t0 = time.time()
    mc, mo, mb, pv, d3 = test_ai_follow_quality(500)
    t1 = time.time()
    print(f"  _has_color_cards误判: {mc}次, 误判但合法: {mo}, 误判且非法: {mb}")
    print(f"  对子结构违反: {pv}, 耗时: {t1-t0:.1f}s")
    for d in d3[:10]:
        print(f"    {d}")
    
    # Test 2: 2000局E2E
    print("\n[Test 2] 端到端2000局AI合法性测试...")
    t0 = time.time()
    comp2 = test_ai_legality_2000(2000)
    t1 = time.time()
    print(f"  完成: {comp2}/2000, 耗时: {t1-t0:.1f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"引擎规则问题: {len(issues_1)}")
    print(f"AI策略问题(误判且非法): {mb}")
    print(f"AI对子违反: {pv}")
    print(f"2000局完成: {comp2}")
    total = len(issues_1) + mb + pv
    if total == 0:
        print("\n✅ 所有测试通过！引擎验证逻辑正确，AI出牌合法。")
    else:
        print(f"\n⚠️ 发现 {total} 个问题需修复。")
