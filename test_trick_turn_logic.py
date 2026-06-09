from server.card import card_from_str
from server.game_engine import GameRoom, Player
from server.rules import compare_outcards, determine_play_type
from server.constants import GamePhase


def C(card_type: str):
    return card_from_str(card_type)


def assert_false(value, msg):
    if value:
        raise AssertionError(msg)


def assert_equal(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def make_room():
    room = GameRoom("test")
    room.players = [Player(i, name=f"P{i}") for i in range(4)]
    room.phase = GamePhase.PLAYING
    room.now_level = "A"
    room.now_color = "a"
    room.bankers = [0, 2]
    return room


def test_equal_duplicate_does_not_overtake():
    assert_false(
        compare_outcards([C("K-12-b")], [C("K-12-b")], "A", "a"),
        "same duplicate card must not overtake earlier play",
    )


def test_off_suit_sub_card_does_not_win():
    assert_false(
        compare_outcards([C("K-12-c")], [C("4-3-b")], "A", "a"),
        "different side suit must not win the lead side suit",
    )


def test_fixed_trump_beats_plain_trump_suit():
    assert compare_outcards(
        [C("5-4-b")], [C("K-12-a")], "A", "a"
    ), "fixed trump 5 must beat ordinary trump-suit K"


def test_trick_winner_keeps_next_turn_across_two_rounds():
    room = make_room()

    room.epoch_cards = [
        [C("K-12-b")],
        [C("K-12-b")],
        [C("Q-11-b")],
        [C("J-10-b")],
    ]
    room.epoch_players = [0, 1, 2, 3]
    winner, _ = room._resolve_epoch()
    assert_equal(winner, 0, "first duplicate K should keep trick")
    room.current_turn = winner

    room.epoch_cards = [
        [C("5-4-b")],
        [C("K-12-a")],
        [C("3-2-b")],
        [C("2-1-b")],
    ]
    room.epoch_players = [room.current_turn, 1, 2, 3]
    winner, _ = room._resolve_epoch()
    assert_equal(winner, 0, "fixed trump 5 should keep next trick")


def test_handle_play_keeps_human_turn_after_two_winning_tricks():
    room = make_room()
    hands = [
        ["K-12-b", "5-4-b", "6-5-c"],
        ["K-12-b", "K-12-a", "7-6-c"],
        ["Q-11-b", "3-2-b", "8-7-c"],
        ["J-10-b", "2-1-b", "9-8-c"],
    ]
    for seat, cards in enumerate(hands):
        room.players[seat].cards_in_hand = {}
        room.players[seat].add_cards([C(card) for card in cards])

    for seat, card in enumerate(["K-12-b", "K-12-b", "Q-11-b", "J-10-b"]):
        result = room.handle_play(seat, [card])
        assert_equal(result["status"], "ok", f"round 1 play by seat {seat}")

    assert_equal(room.last_epoch_winner, 0, "round 1 winner")
    assert_equal(room.current_turn, 0, "round 1 next turn")

    room.epoch_cards = []
    room.epoch_players = []
    room._pending_clear_epoch = False

    for seat, card in enumerate(["5-4-b", "K-12-a", "3-2-b", "2-1-b"]):
        result = room.handle_play(seat, [card])
        assert_equal(result["status"], "ok", f"round 2 play by seat {seat}")

    assert_equal(room.last_epoch_winner, 0, "round 2 winner")
    assert_equal(room.current_turn, 0, "round 2 next turn")


def test_koupai_can_put_picked_hole_cards_back():
    room = GameRoom("koupai-test")
    room.players = [Player(i, name=f"P{i}") for i in range(4)]
    room.phase = GamePhase.KOUPAI
    room.now_level = "A"
    room.now_color = "a"
    room.liangzhu_player = 0
    room.bankers = [0, 2]
    room.hole_cards = [C(card) for card in ["2-1-a", "5-4-b", "10-9-c", "K-12-d"]]

    picker = room._get_koupai_picker()
    assert_equal(picker, 2, "seat 2 should pick bottom cards")
    room.players[picker].add_cards([C(card) for card in ["6-5-a", "7-6-b", "8-7-c", "9-8-d"]])

    result = room.handle_koupai(picker, ["2-1-a", "5-4-b", "10-9-c", "K-12-d"])
    assert_equal(result["status"], "ok", "picked bottom cards can be put back")
    assert_equal(room.players[picker].card_count, 4, "picker hand count returns to original size")
    assert_equal([c.card_type for c in room.koupai_cards], ["2-1-a", "5-4-b", "10-9-c", "K-12-d"], "bottom is updated")


def test_koupai_hole_cards_visible_only_to_picker():
    room = GameRoom("koupai-state-test")
    room.players = [Player(i, sid=f"sid-{i}", name=f"P{i}") for i in range(4)]
    room.phase = GamePhase.KOUPAI
    room.now_level = "A"
    room.now_color = "a"
    room.liangzhu_player = 0
    room.bankers = [0, 2]
    room.hole_cards = [C(card) for card in ["2-1-a", "5-4-b", "10-9-c", "K-12-d"]]

    picker_state = room.get_state("sid-2")
    other_state = room.get_state("sid-1")
    assert_equal(picker_state["koupai_picker"], 2, "state exposes picker")
    assert_equal(len(picker_state["hole_cards"]), 4, "picker sees picked bottom cards")
    assert_equal(other_state["hole_cards"], [], "other players do not see bottom cards before扣牌完成")


def test_public_koupai_cards_wait_until_return_flow_finished():
    room = GameRoom("public-koupai-test")
    room.players = [Player(i, sid=f"sid-{i}", name=f"P{i}") for i in range(4)]
    room.now_level = "A"
    room.now_color = "d"
    room.liangzhu_player = 0
    room.bankers = [0, 2]
    room.koupai_cards = [C(card) for card in ["5-4-d", "10-9-b", "K-12-c", "2-1-a"]]

    room.phase = GamePhase.HUANPAI
    hidden_state = room.get_state("sid-1")
    assert_equal(hidden_state["public_koupai_cards"], [], "bottom stays hidden while pick/return is pending")
    assert_equal(hidden_state["koupai_cards"], [], "legacy bottom field also stays hidden while pending")

    room.phase = GamePhase.PLAYING
    visible_state = room.get_state("sid-1")
    assert_equal(
        [card["card_type"] for card in visible_state["public_koupai_cards"]],
        ["5-4-d", "10-9-b", "K-12-c", "2-1-a"],
        "bottom is public only after return flow is complete",
    )


def test_chipai_stronger_liang_type_takes_over_and_returns_cards():
    room = GameRoom("chipai-test")
    room.players = [Player(i, sid=f"sid-{i}", name=f"P{i}") for i in range(4)]
    room.phase = GamePhase.LIANGZHU
    room.now_level = "A"
    room.hole_cards = [C(card) for card in ["2-1-a", "3-2-b", "4-3-c", "5-4-d"]]

    room.players[0].add_cards([C(card) for card in ["8-7-b", "8-7-b", "w-14-z", "6-5-c"]])
    room.players[1].add_cards([C(card) for card in ["9-8-c", "9-8-c", "10-9-c", "10-9-c", "w-14-z", "7-6-d"]])

    first = room.handle_liangzhu(0, ["8-7-b", "8-7-b", "w-14-z"], "b")
    assert_equal(first["status"], "ok", "first liangzhu should succeed")
    assert_equal(room.phase, GamePhase.CHIPAI, "first liangzhu enters chipai")

    same_team = room.handle_chipai_pass(2)
    assert_equal(same_team["status"], "error", "same team cannot respond to chipai")

    chipai_liang = room.handle_liangzhu(1, ["9-8-c", "9-8-c", "10-9-c", "10-9-c", "w-14-z"], "c")
    assert_equal(chipai_liang["status"], "ok", "stronger chipai liangzhu should succeed")
    claim = room.handle_chipai_claim(1)
    assert_equal(claim["status"], "ok", "chipai claim should succeed")
    assert_equal(room.liangzhu_player, 1, "stronger chipai player becomes liangzhu player")
    assert_equal(room.now_color, "c", "chipai player updates trump color")
    assert_equal(room.phase, GamePhase.KOUPAI, "chipai proceeds to koupai after return cards")


def test_robot_can_liangzhu_even_when_human_has_not_passed():
    room = GameRoom("robot-liangzhu-test")
    room.players = [
        Player(0, sid="human", name="P0"),
        Player(1, sid="robot-1", name="P1", is_robot=True),
        Player(2, sid="robot-2", name="P2", is_robot=True),
        Player(3, sid="robot-3", name="P3", is_robot=True),
    ]
    room.phase = GamePhase.LIANGZHU
    room.now_level = "A"
    room.hole_cards = [C(card) for card in ["2-1-a", "3-2-b", "4-3-c", "5-4-d"]]
    room.players[1].add_cards([C(card) for card in [
        "4-3-c", "4-3-c", "5-4-c", "5-4-c", "6-5-c", "6-5-c",
        "10-9-c", "10-9-c", "K-12-c", "K-12-c", "w-14-z"
    ]])

    actions = room.auto_play_current_robot()
    assert actions, "robot should act without waiting for human pass"
    assert_equal(room.liangzhu_player, 1, "robot can become liangzhu player")
    assert_equal(room.phase, GamePhase.CHIPAI, "robot liangzhu enters chipai")


def test_follow_main_pair_without_main_pair_must_play_top_trumps():
    room = make_room()
    room.current_turn = 0
    hands = [
        ["5-4-a", "5-4-a", "6-5-c", "7-6-c"],
        ["2-1-a", "3-2-a", "K-12-a", "8-7-b"],
        ["6-5-c", "7-6-c", "8-7-c", "9-8-c"],
        ["6-5-d", "7-6-d", "8-7-d", "9-8-d"],
    ]
    for seat, cards in enumerate(hands):
        room.players[seat].cards_in_hand = {}
        room.players[seat].add_cards([C(card) for card in cards])

    result = room.handle_play(0, ["5-4-a", "5-4-a"])
    assert_equal(result["status"], "ok", "seat 0 leads main pair")

    bad = room.handle_play(1, ["2-1-a", "K-12-a"])
    assert_equal(bad["status"], "error", "small trump follow must be rejected")

    good = room.handle_play(1, ["3-2-a", "2-1-a"])
    assert_equal(good["status"], "ok", "top two trumps are valid when no main pair")


def test_mixed_suit_main_cards_are_not_main_pair():
    assert_equal(
        determine_play_type([C("A-13-a"), C("A-13-c")], "A", "a"),
        "zhusan",
        "same-rank trump cards in different suits are main singles, not a pair",
    )

    room = make_room()
    room.players[1].cards_in_hand = {}
    room.players[1].add_cards([C(card) for card in ["A-13-a", "A-13-c", "w-14-z", "5-4-a"]])
    pairs = room._get_zhu_pairs(room._get_player_zhu_cards(room.players[1]))
    assert_equal(pairs, [], "mixed-suit main cards should not be treated as main pairs")


def test_score_result_sets_next_bankers_level_and_continues_next_game():
    room = make_room()
    room.phase = GamePhase.PLAYING
    room.game_num = 1
    room.now_level = "A"
    room.bankers = [0, 2]
    room.score_now = 120
    room.score_koupai = 0
    room.last_epoch_winner = 1

    result = room._end_game()
    assert_equal(result["winner_team"], "xianjia", "idle team should win at 120 points")
    assert_equal(result["levels"], 1, "120 points promotes idle team one level")
    assert_equal(result["new_bankers"], [1, 3], "idle team becomes next bankers")
    assert_equal(result["next_level"], "2", "next level follows new banker level")

    next_start = room.start_game()
    assert_equal(next_start["status"], "ok", "next game can start after game over")
    assert_equal(room.phase, GamePhase.LIANGZHU, "next game enters liangzhu")
    assert_equal(room.bankers, [1, 3], "next game keeps resolved bankers")
    assert_equal(room.now_level, "2", "next game uses upgraded banker level")


def test_final_trick_saves_current_cards_before_bottom_score():
    room = make_room()
    room.current_turn = 0
    room.score_now = 70
    room.score_koupai = 10
    room.koupai_cards = [C("10-9-c")]
    room.last_epoch_winner = -1
    room.last_epoch_cards = []
    room.last_epoch_players = []

    hands = [
        ["4-3-b"],
        ["5-4-a"],
        ["6-5-b"],
        ["7-6-b"],
    ]
    for seat, cards in enumerate(hands):
        room.players[seat].cards_in_hand = {}
        room.players[seat].add_cards([C(card) for card in cards])

    for seat, card in enumerate(["4-3-b", "5-4-a", "6-5-b", "7-6-b"]):
        result = room.handle_play(seat, [card])
        assert_equal(result["status"], "ok", f"final trick play by seat {seat}")

    assert "game_over" in result, "final trick should end the game"
    assert_equal(room.last_epoch_winner, 1, "current final trick winner must be saved")
    assert_equal(room.score_now, 85, "idle side should add trick score and bottom score from the actual final trick")
    assert_equal(result["game_over"]["score_now"], 85, "result should include bottom-adjusted score")
    assert_equal(result["game_over"]["bottom_base_score"], 10, "result should include bottom base score")
    assert_equal(result["game_over"]["koudi_multiplier"], 1, "single main card on final trick should use 1x bottom multiplier")
    assert_equal(result["game_over"]["xianjia_koudi_score"], 10, "idle side should show added bottom score separately")


def test_banker_guard_bottom_score_is_reported_separately():
    room = make_room()
    room.phase = GamePhase.PLAYING
    room.score_now = 35
    room.score_koupai = 10
    room.koupai_cards = [C("10-9-c")]
    room.last_epoch_winner = 0
    room.last_epoch_players = [0, 1, 2, 3]
    room.last_epoch_cards = [
        [C("A-13-a"), C("A-13-a")],
        [C("K-12-b"), C("K-12-b")],
        [C("Q-11-b"), C("Q-11-b")],
        [C("J-10-b"), C("J-10-b")],
    ]

    result = room._end_game()

    assert_equal(result["bottom_base_score"], 10, "result should include bottom base score")
    assert_equal(result["koudi_multiplier"], 1, "two main cards should use 1x bottom multiplier")
    assert_equal(result["xianjia_koudi_score"], 0, "banker final trick should not add idle bottom score")
    assert_equal(result["banker_guard_score"], 10, "banker guard score should be reported separately")
    assert_equal(result["score_now"], 25, "banker guard should subtract bottom score from idle total")


def test_huanpai_pick_requires_same_rank_return_and_picker_leads():
    room = GameRoom("huanpai-test")
    room.players = [Player(i, sid=f"sid-{i}", name=f"P{i}") for i in range(4)]
    room.phase = GamePhase.HUANPAI
    room.now_level = "A"
    room.now_color = "a"
    room.liangzhu_player = 0
    room.bankers = [0, 2]
    room.bottom_picker = 2
    room.koupai_cards = [C(card) for card in ["5-4-a", "9-8-a", "10-9-c", "K-12-d"]]
    room.huanpai_offer = [room.koupai_cards[0], room.koupai_cards[1]]
    room.players[0].add_cards([C(card) for card in ["5-4-b", "9-8-c", "8-7-d", "Q-11-c"]])

    state = room.get_state("sid-0")
    assert_equal(
        [card["card_type"] for card in state["huanpai_return_candidates"]],
        ["5-4-b", "9-8-c"],
        "state should show only same-rank cards that can be returned",
    )

    bad = room.handle_shipai(0, ["5-4-a"], ["8-7-d"])
    assert_equal(bad["status"], "error", "different-rank return must be rejected")

    result = room.handle_shipai(0, ["5-4-a"], ["5-4-b"])
    assert_equal(result["status"], "ok", "same-rank different-suit return should be accepted")
    assert_equal(room.phase, GamePhase.PLAYING, "game enters playing after shipai")
    assert_equal(room.current_turn, 0, "last bottom picker after shipai should lead")
    assert_equal(room.bottom_picker, 0, "shipai player becomes the last bottom picker")
    assert_equal([c.card_type for c in room.huanpai_picked_cards], ["5-4-a"], "picked cards should be remembered")
    assert "5-4-a" not in [c.card_type for c in room.koupai_cards], "picked card leaves bottom"
    assert "5-4-b" in [c.card_type for c in room.koupai_cards], "returned card enters bottom"


def test_no_huanpai_original_bottom_picker_leads():
    room = GameRoom("no-huanpai-test")
    room.players = [Player(i, name=f"P{i}") for i in range(4)]
    room.phase = GamePhase.KOUPAI
    room.now_level = "A"
    room.now_color = "a"
    room.liangzhu_player = 0
    room.bankers = [0, 2]
    room.hole_cards = [C(card) for card in ["2-1-c", "5-4-c", "10-9-c", "K-12-d"]]
    picker = room._get_koupai_picker()
    room.players[picker].add_cards([C(card) for card in ["6-5-a", "7-6-b", "8-7-c", "9-8-d"]])

    result = room.handle_koupai(picker, ["2-1-c", "5-4-c", "10-9-c", "K-12-d"])
    assert_equal(result["status"], "ok", "koupai succeeds")
    assert_equal(result["huanpai"], False, "no trump-color bottom cards means no shipai")
    assert_equal(room.phase, GamePhase.PLAYING, "game enters playing")
    assert_equal(room.current_turn, picker, "original bottom picker should lead when no shipai")


if __name__ == "__main__":
    test_equal_duplicate_does_not_overtake()
    test_off_suit_sub_card_does_not_win()
    test_fixed_trump_beats_plain_trump_suit()
    test_trick_winner_keeps_next_turn_across_two_rounds()
    test_handle_play_keeps_human_turn_after_two_winning_tricks()
    test_koupai_can_put_picked_hole_cards_back()
    test_koupai_hole_cards_visible_only_to_picker()
    test_follow_main_pair_without_main_pair_must_play_top_trumps()
    test_mixed_suit_main_cards_are_not_main_pair()
    test_score_result_sets_next_bankers_level_and_continues_next_game()
    test_final_trick_saves_current_cards_before_bottom_score()
    test_banker_guard_bottom_score_is_reported_separately()
    test_huanpai_pick_requires_same_rank_return_and_picker_leads()
    test_no_huanpai_original_bottom_picker_leads()
    print("trick turn logic tests passed")
