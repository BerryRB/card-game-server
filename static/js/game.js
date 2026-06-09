const PHASE_LABELS = {
    waiting: "等待",
    liangzhu: "亮主",
    chipai: "吃牌",
    koupai: "整理底牌",
    huanpai: "拾牌",
    playing: "出牌中",
    scoring: "计分",
    game_over: "结束"
};

const COLOR_SYMBOLS = {
    a: "♠",
    b: "♥",
    c: "♣",
    d: "♦",
    z: "★"
};

const COLOR_NAMES = {
    a: "黑桃",
    b: "红桃",
    c: "梅花",
    d: "方块",
    z: "王"
};

const COLOR_ORDER = { a: 0, b: 1, c: 2, d: 3, z: 4 };
const FIXED_TRUMP_NAMES = new Set(["5", "3", "2"]);
const FIXED_TRUMP_ORDER = { "5": 0, "3": 1, "2": 2 };

let ws = null;
let currentRoomId = "";
let mySeat = -1;
let currentState = null;
let selectedCards = new Set();
let selectedCardKeys = new Set();
let selectedHuanpaiOffers = new Set();
let toastTimer = null;
let handLayoutFrame = null;
let handResizeObserver = null;
let landscapeRequested = false;
let autoStartAfterJoin = false;
let clientId = "";

document.addEventListener("DOMContentLoaded", () => {
    clientId = getClientId();
    loadRooms();
    setActionVisibility({});
    window.addEventListener("resize", scheduleHandLayout);
    window.addEventListener("orientationchange", () => {
        setTimeout(() => {
            updateLandscapeButton();
            scheduleHandLayout();
        }, 240);
    });
    document.addEventListener("fullscreenchange", () => {
        if (!document.fullscreenElement && landscapeRequested) {
            setLandscapeMode(false);
        } else {
            updateLandscapeButton();
        }
    });
    if ("ResizeObserver" in window) {
        handResizeObserver = new ResizeObserver(scheduleHandLayout);
        handResizeObserver.observe($("myHand"));
    }
    updateLandscapeButton();
});

function $(id) {
    return document.getElementById(id);
}

function showScreen(id) {
    document.querySelectorAll(".screen").forEach((screen) => screen.classList.remove("active"));
    $(id).classList.add("active");
    if (id === "game") {
        scheduleHandLayout();
    }
}

function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.hidden = true;
    }, 2600);
}

async function toggleLandscapeMode() {
    if (document.body.classList.contains("landscape-mode")) {
        await exitLandscapeMode();
        return;
    }
    await enterLandscapeMode();
}

async function enterLandscapeMode() {
    landscapeRequested = true;
    setLandscapeMode(true);
    forceLandscapeButtonStyle(true);
    showToast("横屏布局已开启，请将手机横过来");

    try {
        if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
            document.documentElement.requestFullscreen().catch(() => {});
        }
    } catch (error) {
        // Some mobile browsers require fullscreen/orientation to fail silently.
    }

    try {
        if (screen.orientation?.lock) {
            await screen.orientation.lock("landscape");
            showToast("已切换横屏模式");
        }
    } catch (error) {
        // Keep CSS landscape mode active even when the browser refuses locking.
    }

    scheduleHandLayout();
}

async function exitLandscapeMode() {
    landscapeRequested = false;
    setLandscapeMode(false);

    try {
        if (screen.orientation?.unlock) {
            screen.orientation.unlock();
        }
    } catch (error) {
        // Ignore unsupported unlock calls.
    }

    try {
        if (document.fullscreenElement && document.exitFullscreen) {
            document.exitFullscreen().catch(() => {});
        }
    } catch (error) {
        // Ignore fullscreen exit failures.
    }

    scheduleHandLayout();
}

function setLandscapeMode(enabled) {
    document.body.classList.toggle("landscape-mode", enabled);
    forceLandscapeButtonStyle(enabled);
    updateLandscapeButton();
    scheduleHandLayout();
}

function updateLandscapeButton() {
    const button = $("landscapeBtn");
    const text = $("landscapeBtnText");
    if (!button || !text) {
        return;
    }
    const active = document.body.classList.contains("landscape-mode");
    button.classList.toggle("active", active);
    text.textContent = active ? "竖屏" : "横屏";
}

function forceLandscapeButtonStyle(active) {
    const button = $("landscapeBtn");
    if (!button) {
        return;
    }
    button.style.position = "absolute";
    button.style.zIndex = "14";
    button.style.top = document.body.classList.contains("landscape-mode") ? "8px" : "12px";
    button.style.right = document.body.classList.contains("landscape-mode") ? "8px" : "12px";
    button.style.minWidth = document.body.classList.contains("landscape-mode") ? "54px" : "62px";
    button.style.minHeight = document.body.classList.contains("landscape-mode") ? "32px" : "38px";
    button.style.padding = document.body.classList.contains("landscape-mode") ? "6px 9px" : "8px 12px";
    button.style.borderRadius = "8px";
    button.style.fontWeight = "800";
    button.style.color = "#fff5cb";
    button.style.border = active ? "1px solid rgba(255,212,109,.86)" : "1px solid rgba(255,255,255,.32)";
    button.style.background = active ? "rgba(151,103,15,.82)" : "rgba(3,19,14,.72)";
}

async function loadRooms() {
    try {
        const response = await fetch("/api/rooms");
        const rooms = await response.json();
        const list = $("roomList");
        list.innerHTML = "";
        if (!rooms.length) {
            list.innerHTML = '<div class="room-item"><span>暂无房间</span><span>先创建一个</span></div>';
            return;
        }
        rooms.forEach((room) => {
            const item = document.createElement("div");
            item.className = "room-item";
            item.innerHTML = `
                <div>
                    <strong>${escapeHtml(room.room_id)}</strong>
                    <span>${room.player_count || 0}/4 · ${PHASE_LABELS[room.phase] || room.phase}</span>
                </div>
                <button class="btn secondary compact" type="button">加入</button>
            `;
            item.querySelector("button").addEventListener("click", () => {
                $("roomId").value = room.room_id;
                joinRoom();
            });
            list.appendChild(item);
        });
    } catch (error) {
        showToast("房间列表加载失败");
    }
}

async function createRoom() {
    try {
        const response = await fetch("/api/rooms", { method: "POST" });
        const data = await response.json();
        $("roomId").value = data.room_id;
        await joinRoom();
    } catch (error) {
        showToast("创建房间失败");
    }
}

async function startRobotRoom() {
    autoStartAfterJoin = true;
    try {
        const response = await fetch("/api/rooms", { method: "POST" });
        const data = await response.json();
        $("roomId").value = data.room_id;
        await joinRoom(false);
    } catch (error) {
        autoStartAfterJoin = false;
        showToast("人机试玩创建失败");
    }
}

async function joinRoom(asSpectator = false) {
    const roomId = $("roomId").value.trim();
    if (!roomId) {
        showToast("请输入房间号");
        return;
    }

    currentRoomId = roomId;
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${location.host}/ws`);

    ws.addEventListener("open", () => {
        ws.send(JSON.stringify({
            action: "join",
            room_id: roomId,
            name: $("playerName").value.trim() || "玩家",
            seat: asSpectator ? -1 : 0,
            client_id: clientId
        }));
    });

    ws.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        handleMessage(message);
    });

    ws.addEventListener("close", () => {
        showToast("连接已断开");
    });

    ws.addEventListener("error", () => {
        showToast("连接失败");
    });
}

function handleMessage(message) {
    if (message.action === "joined") {
        mySeat = message.seat;
        currentRoomId = message.room_id;
        $("displayRoomId").textContent = currentRoomId;
        if (message.msg) {
            showToast(message.msg);
        }
        showScreen("room");
        updateState(message.state);
        if (autoStartAfterJoin && message.seat >= 0 && message.state?.phase === "waiting") {
            autoStartAfterJoin = false;
            startGame();
        }
        if (message.state && message.state.phase !== "waiting") {
            showScreen("game");
        }
        return;
    }

    if (message.action === "state_update" || message.action === "state") {
        updateState(message.state);
        return;
    }

    if (message.action === "play_result" || message.action === "koupai_result" || message.action === "liangzhu_result" || message.action === "huanpai_result") {
        if (message.status === "error") {
            showToast(message.msg || "操作失败");
        }
        return;
    }

    if (message.action === "suggest_play_result") {
        applySuggestion(message.suggestions || []);
        return;
    }

    if (message.action === "chat") {
        showToast(`${message.name}: ${message.msg}`);
    }
}

function updateState(state) {
    if (!state) {
        return;
    }

    currentState = state;
    mySeat = typeof state.my_seat === "number" ? state.my_seat : mySeat;
    currentRoomId = state.room_id || currentRoomId;
    if (state.phase !== "huanpai" || state.liangzhu_player !== mySeat) {
        selectedHuanpaiOffers.clear();
    } else {
        const offerCount = state.huanpai_offer?.length || 0;
        selectedHuanpaiOffers = new Set([...selectedHuanpaiOffers].filter((index) => index >= 0 && index < offerCount));
    }

    $("hudRoom").textContent = currentRoomId || "----";
    $("displayRoomId").textContent = currentRoomId || "----";
    $("hudPhase").textContent = PHASE_LABELS[state.phase] || state.phase || "等待";
    $("infoLevel").textContent = state.now_level || "A";
    $("infoColor").innerHTML = renderTrumpHud(state);
    $("infoScore").textContent = String(state.score_now ?? 0);
    $("infoRound").textContent = String(state.round_num || 1);

    renderSeats(state);
    renderPlayers(state);
    renderHoleCards(state);
    renderKoupaiCards(state);
    renderPlayedCards(state);
    if (state.phase === "waiting") {
        showScreen("room");
    } else {
        showScreen("game");
    }

    renderHand(state);
    renderCenterMessage(state);
    setActionVisibility(state);

    if (state.phase === "game_over" && state.game_result) {
        showResult(state.game_result);
    }
}

function renderSeats(state) {
    for (let seat = 0; seat < 4; seat += 1) {
        const el = $(`seat${seat}`);
        const player = getPlayer(state, seat);
        el.classList.toggle("occupied", Boolean(player));
        el.classList.toggle("me", seat === mySeat);
        el.querySelector(".seat-name").textContent = player ? player.name : "空位";
    }
}

function renderPlayers(state) {
    const positions = relativePositions(mySeat);
    bindPlayerPanel("left", positions.left, state);
    bindPlayerPanel("top", positions.top, state);
    bindPlayerPanel("right", positions.right, state);

    const me = getPlayer(state, mySeat);
    const myCards = getMyCards(state);
    $("myName").textContent = me ? me.name : "我";
    if (state.phase === "koupai" && state.koupai_picker === mySeat && state.hole_cards?.length) {
        $("myCount").textContent = `${myCards.length} 张（含底牌）`;
    } else {
        $("myCount").textContent = `${me?.card_count || myCards.length || 0} 张`;
    }
    $("myBankerTag").classList.toggle("visible", Boolean(me?.is_banker));
    $("myBankerTag").textContent = me?.is_banker ? "庄" : "闲";
    $("myRole").textContent = me?.is_banker ? "庄家" : "闲家";
    $("myRole").classList.toggle("banker", Boolean(me?.is_banker));
}

function bindPlayerPanel(position, seat, state) {
    const player = getPlayer(state, seat);
    const area = $(`${position}Area`);
    area.classList.toggle("active-turn", state.current_turn === seat && state.phase === "playing");
    area.classList.toggle("banker", Boolean(player?.is_banker));
    $(`${position}Name`).textContent = player ? player.name : seatName(seat);
    $(`${position}Count`).textContent = `${player?.card_count || 0} 张`;
    $(`${position}Role`).textContent = player?.is_banker ? "庄家" : "闲家";
    $(`${position}Role`).classList.toggle("banker", Boolean(player?.is_banker));
}

function renderHoleCards(state) {
    const count = Math.max(4, state.hole_cards_count || 4);
    const visibleCards = state.hole_cards || [];
    const area = $("holeCardsArea");
    const label = area?.querySelector(".kitty-label");
    area?.classList.remove("has-visible-cards");
    if (state.huanpai_picked_cards?.length && state.liangzhu_player === mySeat) {
        area?.classList.add("has-visible-cards");
        $("holeCards").innerHTML = state.huanpai_picked_cards
            .map((card) => cardHtml(parseCard(card), { small: true }))
            .join("");
        if (label) {
            label.textContent = "你已拾取";
        }
        return;
    }
    if (state.phase === "huanpai" && state.liangzhu_player === mySeat && state.huanpai_offer?.length) {
        area?.classList.add("has-visible-cards");
        const offerCards = state.huanpai_offer || [];
        $("holeCards").innerHTML = offerCards
            .map((card, index) => cardHtml(parseCard(card), {
                small: true,
                selectable: true,
                selected: selectedHuanpaiOffers.has(index),
                cardType: card.card_type || card,
                offerIndex: index
            }))
            .join("");
        $("holeCards").querySelectorAll(".card").forEach((cardEl) => {
            cardEl.addEventListener("click", () => toggleHuanpaiOffer(Number(cardEl.dataset.offerIndex)));
        });
        if (label) {
            label.textContent = "可拾底牌";
        }
        return;
    }
    if (visibleCards.length) {
        area?.classList.add("has-visible-cards");
        $("holeCards").innerHTML = visibleCards
            .map((card) => cardHtml(parseCard(card), { small: true }))
            .join("");
        if (label) {
            label.textContent = "已拾取底牌";
        }
        return;
    }
    $("holeCards").innerHTML = Array.from({ length: count }, () => '<span class="card-back"></span>').join("");
    if (label) {
        label.textContent = "底牌区";
    }
}

function renderKoupaiCards(state) {
    const area = $("koupaiCardsArea");
    const list = $("koupaiCards");
    if (!area || !list) {
        return;
    }
    const cards = state.public_koupai_cards || [];
    area.hidden = !cards.length;
    if (!cards.length) {
        list.innerHTML = "";
        return;
    }
    list.innerHTML = cards
        .map((card) => cardHtml(parseCard(card), { small: true }))
        .join("");
}

function renderTrumpHud(state) {
    const color = state.now_color;
    if (!color) {
        return '<span class="trump-inline muted">待定</span>';
    }
    const red = color === "b" || color === "d";
    return `
        <span class="trump-inline ${red ? "red" : ""}">
            <span class="trump-suit">${escapeHtml(COLOR_SYMBOLS[color] || "")}</span>
            <span>${escapeHtml(COLOR_NAMES[color] || "--")}</span>
        </span>
    `;
}

function renderPlayedCards(state) {
    const positions = relativePositions(mySeat);
    const slots = {
        [positions.top]: "playedTop",
        [positions.left]: "playedLeft",
        [positions.right]: "playedRight",
        [mySeat]: "playedBottom"
    };

    Object.values(slots).forEach((id) => {
        if ($(id)) {
            $(id).innerHTML = '<span class="empty-zone">待出牌</span>';
        }
    });

    const players = state.epoch_players || [];
    const cards = state.epoch_cards || [];
    players.forEach((seat, index) => {
        const slotId = slots[seat];
        if (slotId && $(slotId)) {
            $(slotId).innerHTML = renderCards(cards[index] || [], { small: true, selectable: false });
        }
    });

    if (!players.length && state.last_epoch_players?.length) {
        state.last_epoch_players.forEach((seat, index) => {
            const slotId = slots[seat];
            if (slotId && $(slotId)) {
                $(slotId).innerHTML = renderCards(state.last_epoch_cards[index] || [], { small: true, selectable: false });
            }
        });
    }
}

function cardSelectionKey(cards, index) {
    const cardType = cards[index];
    if (!cardType) {
        return "";
    }
    let occurrence = 0;
    for (let i = 0; i <= index; i += 1) {
        if (cards[i] === cardType) {
            occurrence += 1;
        }
    }
    return `${cardType}#${occurrence}`;
}

function selectedIndexesFromKeys(cards) {
    const indexes = [];
    cards.forEach((_, index) => {
        if (selectedCardKeys.has(cardSelectionKey(cards, index))) {
            indexes.push(index);
        }
    });
    return indexes;
}

function syncSelectedCardsFromKeys(cards) {
    const validKeys = new Set(cards.map((_, index) => cardSelectionKey(cards, index)));
    selectedCardKeys = new Set([...selectedCardKeys].filter((key) => validKeys.has(key)));
    selectedCards = new Set(selectedIndexesFromKeys(cards));
}

function clearSelectedCards() {
    selectedCards.clear();
    selectedCardKeys.clear();
}

function renderHand(state) {
    const cards = getMyCards(state);
    const hand = $("myHand");
    const reconnectButton = $("btnNewRobotGame");
    syncSelectedCardsFromKeys(cards);
    cards.forEach((cardType, index) => {
        if (!isHandCardSelectable(state, cardType)) {
            selectedCardKeys.delete(cardSelectionKey(cards, index));
        }
    });
    syncSelectedCardsFromKeys(cards);
    updateHandPrompt(state);

    if (!cards.length) {
        const me = getPlayer(state, mySeat);
        const needsNewGame = state?.phase && state.phase !== "waiting" && (!me || mySeat < 0 || me.card_count === 0);
        const message = needsNewGame
            ? "当前未入座，请新开一局"
            : state?.phase && state.phase !== "waiting" && me?.card_count > 0
                ? "同步手牌中，请点刷新或重新进入"
                : "等待发牌";
        hand.innerHTML = `<span class="empty-zone">${escapeHtml(message)}</span>`;
        if (reconnectButton) {
            reconnectButton.hidden = !needsNewGame;
        }
        return;
    }
    if (reconnectButton) {
        reconnectButton.hidden = true;
    }

    hand.innerHTML = cards.map((cardType, index) => {
        const selectable = isHandCardSelectable(state, cardType);
        return cardHtml(parseCard(cardType), {
            selected: selectedCards.has(index),
            selectable,
            disabled: !selectable,
            cardType,
            index
        });
    }).join("");

    hand.querySelectorAll(".card").forEach((cardEl) => {
        cardEl.addEventListener("click", () => toggleCard(Number(cardEl.dataset.index)));
    });
    scheduleHandLayout();
}

function updateHandPrompt(state) {
    const prompt = $("handPrompt");
    if (!prompt) {
        return;
    }
    if (state?.phase === "koupai" && state.koupai_picker === mySeat) {
        const selected = selectedCards.size;
        const remaining = Math.max(0, 4 - selected);
        const pickedCards = state.hole_cards || [];
        const picked = pickedCards.length;
        const pickedText = pickedCards.length ? ` · 拾取：${plainCards(pickedCards)}` : "";
        prompt.textContent = `已拾取底牌${picked}张 · 已选${selected}张 · 还需扣${remaining}张${pickedText}`;
        prompt.classList.add("visible");
        return;
    }
    if (state?.phase === "huanpai" && state.liangzhu_player === mySeat) {
        const offer = state.huanpai_offer || [];
        const pickedCards = getSelectedHuanpaiOfferCards(state);
        const returnCandidates = getHuanpaiReturnCandidates(state);
        const need = pickedCards.length;
        const remaining = Math.max(0, need - selectedCards.size);
        const pickedText = pickedCards.length ? ` · 已拾：${plainCards(pickedCards)}` : "";
        const candidateText = returnCandidates.length ? ` · 可扣：${plainCards(returnCandidates)}` : "";
        const actionText = need ? `还需扣${remaining}张同点数牌` : "先选择要拾的底牌";
        prompt.textContent = `可拾${offer.length}张底牌 · 已选拾${need}张 · ${actionText}${pickedText}${candidateText}`;
        prompt.classList.add("visible");
        return;
    }
    prompt.textContent = "";
    prompt.classList.remove("visible");
}

function toggleHuanpaiOffer(index) {
    if (!Number.isInteger(index)) {
        return;
    }
    if (selectedHuanpaiOffers.has(index)) {
        selectedHuanpaiOffers.delete(index);
    } else {
        selectedHuanpaiOffers.add(index);
    }
    clearSelectedCards();
    renderHoleCards(currentState);
    renderHand(currentState);
    renderCenterMessage(currentState);
}

function getSelectedHuanpaiOfferCards(state) {
    const offer = state?.huanpai_offer || [];
    return [...selectedHuanpaiOffers]
        .sort((a, b) => a - b)
        .map((index) => offer[index])
        .filter(Boolean);
}

function getHuanpaiReturnCandidates(state) {
    const pickedCards = getSelectedHuanpaiOfferCards(state);
    if (!pickedCards.length) {
        return [];
    }
    const pickedRanks = new Set(pickedCards.map((card) => parseCard(card).rank));
    return getMyCards(state).filter((cardType) => pickedRanks.has(parseCard(cardType).rank));
}

function isHandCardSelectable(state, cardType) {
    if (!state) {
        return true;
    }
    if (state.phase === "huanpai" && state.liangzhu_player === mySeat) {
        const pickedCards = getSelectedHuanpaiOfferCards(state);
        if (!pickedCards.length) {
            return false;
        }
        const pickedRanks = new Set(pickedCards.map((card) => parseCard(card).rank));
        return pickedRanks.has(parseCard(cardType).rank);
    }
    return true;
}

function scheduleHandLayout() {
    cancelAnimationFrame(handLayoutFrame);
    handLayoutFrame = requestAnimationFrame(updateHandLayout);
}

function updateHandLayout() {
    const hand = $("myHand");
    if (!hand) {
        return;
    }

    const cards = hand.querySelectorAll(".card");
    const count = cards.length;
    if (hand.clientWidth < 40) {
        return;
    }
    if (!count) {
        hand.style.removeProperty("--hand-height");
        return;
    }

    const available = Math.max(180, hand.clientWidth - 16);
    const mobile = window.matchMedia("(max-width: 860px)").matches;
    const visualLandscape = window.innerWidth > window.innerHeight;
    const landscape = document.body.classList.contains("landscape-mode") && visualLandscape;
    const portraitMobile = mobile && !landscape;
    const shortPortrait = portraitMobile && window.innerHeight <= 720;
    const narrowLandscape = landscape && window.innerWidth < 620;
    const shortLandscape = landscape && window.innerHeight <= 520;
    const tinyLandscape = landscape && window.innerHeight <= 390;
    const densePortrait = portraitMobile && count >= 25;
    const cardWidth = tinyLandscape ? 31 : shortLandscape ? 34 : landscape ? (narrowLandscape ? 40 : 46) : densePortrait ? (shortPortrait ? 40 : 44) : shortPortrait ? 42 : portraitMobile ? (count >= 21 ? 44 : 46) : 58;
    const cardHeight = tinyLandscape ? 44 : shortLandscape ? 48 : landscape ? (narrowLandscape ? 56 : 64) : densePortrait ? (shortPortrait ? 56 : 62) : shortPortrait ? 58 : portraitMobile ? (count >= 21 ? 62 : 64) : 82;
    const minStep = tinyLandscape ? 24 : shortLandscape ? 24 : landscape ? (narrowLandscape ? 24 : 33) : densePortrait ? (shortPortrait ? 24 : 27) : shortPortrait ? 25 : portraitMobile ? 28 : 16;
    const maxRows = landscape ? (tinyLandscape || shortLandscape ? 2 : narrowLandscape ? 3 : 2) : portraitMobile ? (densePortrait ? 3 : 2) : 2;
    const rowConfig = chooseHandRows(count, available, cardWidth, minStep, maxRows);
    const { rows, rowCounts, step } = rowConfig;
    const rowGap = tinyLandscape ? 2 : shortLandscape ? 3 : landscape ? (narrowLandscape ? 5 : 6) : densePortrait ? (shortPortrait ? 3 : 4) : shortPortrait ? 4 : portraitMobile ? 5 : 10;
    const lift = tinyLandscape ? 6 : shortLandscape ? 7 : landscape ? (narrowLandscape ? 8 : 12) : densePortrait ? (shortPortrait ? 7 : 9) : shortPortrait ? 9 : portraitMobile ? 12 : 18;
    const handHeight = lift + (cardHeight * rows) + (rowGap * (rows - 1)) + 2;

    hand.style.setProperty("--hand-card-w", `${cardWidth}px`);
    hand.style.setProperty("--hand-card-h", `${cardHeight}px`);
    hand.style.setProperty("--hand-card-pad", tinyLandscape || shortLandscape ? "3px" : landscape ? "5px" : densePortrait ? "4px" : shortPortrait ? "4px" : portraitMobile ? "5px" : "7px");
    hand.style.setProperty("--hand-card-rank", tinyLandscape ? "0.64rem" : shortLandscape ? "0.68rem" : landscape ? "0.86rem" : densePortrait ? "0.8rem" : shortPortrait ? "0.78rem" : portraitMobile ? (count >= 21 ? "0.82rem" : "0.86rem") : "1.05rem");
    hand.style.setProperty("--hand-card-suit", tinyLandscape ? "0.68rem" : shortLandscape ? "0.72rem" : landscape ? "0.9rem" : densePortrait ? "0.84rem" : shortPortrait ? "0.82rem" : portraitMobile ? (count >= 21 ? "0.86rem" : "0.9rem") : "1.12rem");
    hand.style.setProperty("--hand-height", `${handHeight}px`);
    hand.style.setProperty("--selected-lift", `${lift}px`);

    let cardIndex = 0;
    for (let row = 0; row < rows; row += 1) {
        const rowCount = rowCounts[row] || 0;
        const rowWidth = rowCount <= 1 ? cardWidth : cardWidth + (step * (rowCount - 1));
        const startX = Math.max(0, (hand.clientWidth - rowWidth) / 2);
        for (let col = 0; col < rowCount; col += 1) {
            const card = cards[cardIndex];
            if (!card) {
                continue;
            }
            card.style.setProperty("--card-x", `${startX + (col * step)}px`);
            card.style.setProperty("--card-y", `${lift + (row * (cardHeight + rowGap))}px`);
            cardIndex += 1;
        }
    }
}

function chooseHandRows(count, available, cardWidth, minStep, maxRows) {
    for (let rows = 1; rows <= maxRows; rows += 1) {
        const columns = Math.ceil(count / rows);
        const step = columns > 1 ? (available - cardWidth) / (columns - 1) : cardWidth;
        if (step >= minStep || rows === maxRows) {
            return {
                rows,
                rowCounts: distributeHandRows(count, rows),
                step: Math.max(minStep, Math.min(cardWidth + 2, step || cardWidth))
            };
        }
    }
    return { rows: 1, rowCounts: [count], step: cardWidth };
}

function distributeHandRows(count, rows) {
    const base = Math.floor(count / rows);
    const extra = count % rows;
    return Array.from({ length: rows }, (_, index) => base + (index < extra ? 1 : 0));
}

function renderCenterMessage(state) {
    const phase = PHASE_LABELS[state.phase] || state.phase;
    if (state.phase === "playing") {
        const current = getPlayer(state, state.current_turn);
        $("centerMessage").textContent = current ? `轮到 ${current.name} 出牌` : "出牌中";
    } else if (state.phase === "liangzhu") {
        $("centerMessage").textContent = "亮主阶段：请选择亮主或不亮";
    } else if (state.phase === "chipai") {
        const needsChipaiAction = state.chipai_needs_action || state.can_chipai_pass || state.can_chipai_claim || state.can_liangzhu;
        $("centerMessage").textContent = needsChipaiAction
            ? "吃牌阶段：可选择更大的亮主牌型，或不吃"
            : "吃牌阶段：等待对方选择";
    } else if (state.phase === "koupai") {
        const need = Math.max(0, 4 - selectedCards.size);
        $("centerMessage").textContent = state.koupai_picker === mySeat
            ? `整理底牌阶段：请选择4张扣回底牌，还需扣${need}张`
            : `整理底牌阶段：等待${seatName(state.koupai_picker)}扣牌`;
    } else if (state.phase === "huanpai") {
        $("centerMessage").textContent = state.liangzhu_player === mySeat
            ? "拾牌阶段：先点底牌选择要拾的牌，再从手牌扣同点数牌"
            : `拾牌阶段：等待${seatName(state.liangzhu_player)}处理底牌`;
    } else {
        $("centerMessage").textContent = phase || "等待开始";
    }

    const liangCards = state.liangzhu_cards?.length ? ` · ${plainCards(state.liangzhu_cards)}` : "";
    $("liangzhuDisplay").textContent = state.liangzhu_player >= 0
        ? `${seatName(state.liangzhu_player)} 已亮主${liangCards}`
        : "";
    $("trumpHint").innerHTML = renderTrumpHint(state);
}

function renderTrumpHint(state) {
    const level = state.now_level || "A";
    const color = state.now_color;
    if (!color) {
        return `
            <span class="trump-title">当前主牌</span>
            <strong>待定</strong>
            <span class="trump-sub">王、${escapeHtml(level)}、5、3、2 为主牌</span>
        `;
    }
    const red = color === "b" || color === "d";
    return `
        <span class="trump-title">当前主牌</span>
        <strong class="${red ? "red" : ""}">${escapeHtml(COLOR_SYMBOLS[color] || "")} ${escapeHtml(COLOR_NAMES[color] || "")}</strong>
        <span class="trump-sub">王 · ${escapeHtml(level)} · 5 · 3 · 2 集中排列</span>
    `;
}

function setActionVisibility(state) {
    const phase = state.phase;
    const myTurn = phase === "playing" && state.current_turn === mySeat;
    const canLiang = Boolean(state.can_liangzhu);
    const canPass = phase === "liangzhu" || Boolean(state.chipai_needs_action || state.can_chipai_pass);
    const canKoupai = phase === "koupai" && state.koupai_picker === mySeat;
    const canHuanpai = phase === "huanpai" && state.liangzhu_player === mySeat;

    $("btnLiangzhu").textContent = phase === "chipai" ? "吃牌" : "亮主";
    $("btnPassLiang").textContent = phase === "chipai" ? "不吃" : "不亮";
    $("btnLiangzhu").hidden = !canLiang;
    $("btnPassLiang").hidden = !canPass;
    $("btnKoupai").hidden = !canKoupai;
    $("btnHuanpaiAccept").hidden = !canHuanpai;
    $("btnHuanpaiPass").hidden = !canHuanpai;
    $("btnAutoFill").hidden = !(myTurn || canKoupai || canHuanpai);
    $("btnPlay").hidden = !myTurn;
    if (!canLiang) {
        closeLiangzhuDialog();
    }
}

function toggleCard(index) {
    if (!Number.isInteger(index)) {
        return;
    }
    const cards = getMyCards(currentState);
    const cardType = cards[index];
    if (!isHandCardSelectable(currentState, cardType)) {
        if (currentState?.phase === "huanpai") {
            showToast("请先点选要拾的底牌，只能扣同点数牌");
        }
        return;
    }
    const key = cardSelectionKey(cards, index);
    if (selectedCardKeys.has(key)) {
        selectedCardKeys.delete(key);
    } else {
        selectedCardKeys.add(key);
    }
    syncSelectedCardsFromKeys(cards);
    renderHand(currentState);
    renderCenterMessage(currentState);
}

function getSelectedCardTypes() {
    const cards = getMyCards(currentState);
    syncSelectedCardsFromKeys(cards);
    return [...selectedCards]
        .sort((a, b) => a - b)
        .map((index) => cards[index])
        .filter(Boolean);
}

function selectCardTypes(cardTypes) {
    const remaining = [...cardTypes];
    const cards = getMyCards(currentState);
    selectedCardKeys = new Set();
    cards.forEach((cardType, index) => {
        const matchIndex = remaining.indexOf(cardType);
        if (matchIndex >= 0) {
            selectedCardKeys.add(cardSelectionKey(cards, index));
            remaining.splice(matchIndex, 1);
        }
    });
    syncSelectedCardsFromKeys(cards);
}

function playCards() {
    const cards = getSelectedCardTypes();
    if (!cards.length) {
        showToast("请先选择要出的牌");
        return;
    }
    send({ action: "play", cards });
    clearSelectedCards();
}

function showLiangzhuDialog() {
    const candidates = currentState?.liangzhu_available_types || [];
    if (!candidates.length) {
        showToast("当前没有可亮的牌");
        return;
    }

    const modal = $("liangzhuModal");
    const list = $("liangzhuOptions");
    const title = modal?.querySelector("h2");
    if (!modal || !list) {
        const choice = candidates[0];
        send({ action: "liangzhu", cards: choice.cards || [] });
        return;
    }
    if (title) {
        title.textContent = currentState?.phase === "chipai" ? "选择吃牌牌型" : "选择亮主花色";
    }

    list.innerHTML = candidates.map((candidate, index) => {
        const parsedCards = (candidate.cards || []).map(parseCard);
        const color = candidate.color || inferLiangzhuColor(candidate.cards || []);
        const label = candidate.label || LIANG_TYPE_LABELS[candidate.type] || "亮主";
        const suit = COLOR_SYMBOLS[color] || "★";
        const colorName = COLOR_NAMES[color] || "主";
        const red = color === "b" || color === "d";
        return `
            <button class="option-card" type="button" data-index="${index}">
                <span class="option-suit ${red ? "red" : ""}">${escapeHtml(suit)}</span>
                <span class="option-main">
                    <span class="option-title">${escapeHtml(colorName)} · ${escapeHtml(label)}</span>
                    <span class="option-subtitle">${escapeHtml(plainCards(candidate.cards || []))}</span>
                </span>
                <span class="option-preview">${parsedCards.map((card) => cardHtml(card, { small: true })).join("")}</span>
            </button>
        `;
    }).join("");

    list.querySelectorAll(".option-card").forEach((button) => {
        button.addEventListener("click", () => {
            const choice = candidates[Number(button.dataset.index)];
            closeLiangzhuDialog();
            clearSelectedCards();
            send({ action: "liangzhu", cards: choice?.cards || [], color: choice?.color || inferLiangzhuColor(choice?.cards || []) });
        });
    });
    modal.hidden = false;
}

function closeLiangzhuDialog() {
    const modal = $("liangzhuModal");
    if (modal) {
        modal.hidden = true;
    }
}

function passLiangzhu() {
    send({ action: "pass_liangzhu" });
}

function showKoupaiDialog() {
    const cards = getSelectedCardTypes();
    const need = 4;
    if (cards.length !== need) {
        showToast(`请选择 ${need} 张扣牌，当前已选 ${cards.length} 张`);
        return;
    }
    send({ action: "koupai", cards });
    clearSelectedCards();
}

function acceptHuanpai() {
    const cards = getSelectedCardTypes();
    const pickCards = getSelectedHuanpaiOfferCards(currentState).map((card) => card.card_type || card);
    const need = pickCards.length;
    if (!currentState?.huanpai_offer?.length) {
        showToast("当前没有可拾牌");
        return;
    }
    if (!need) {
        showToast("请先点选要拾的底牌");
        return;
    }
    if (cards.length !== need) {
        showToast(`请选择 ${need} 张同点数牌扣回`);
        return;
    }
    const pickedRanks = pickCards.map((cardType) => parseCard(cardType).rank).sort((a, b) => a - b);
    const returnRanks = cards.map((cardType) => parseCard(cardType).rank).sort((a, b) => a - b);
    if (pickedRanks.join(",") !== returnRanks.join(",")) {
        showToast("扣回的牌必须和拾起的牌点数一致，花色可以不同");
        return;
    }
    send({ action: "huanpai", accept: true, pick_cards: pickCards, cards });
    clearSelectedCards();
    selectedHuanpaiOffers.clear();
}

function passHuanpai() {
    send({ action: "huanpai", accept: false, cards: [] });
    clearSelectedCards();
    selectedHuanpaiOffers.clear();
}

function onAutoFillClick() {
    if (currentState?.phase === "playing") {
        send({ action: "suggest_play" });
        return;
    }
    if (currentState?.phase === "huanpai") {
        const offer = currentState?.huanpai_offer || [];
        selectedHuanpaiOffers = new Set(offer.map((_, index) => index));
        const hand = getMyCards(currentState);
        const used = new Set();
        const cards = [];
        offer.forEach((offerCard) => {
            const offerParsed = parseCard(offerCard.card_type || offerCard);
            const index = hand.findIndex((cardType, cardIndex) => {
                const card = parseCard(cardType);
                return !used.has(cardIndex) && card.color !== "z" && card.name === offerParsed.name;
            });
            if (index >= 0) {
                used.add(index);
                cards.push(hand[index]);
            }
        });
        if (cards.length < offer.length) {
            showToast("没有足够同点数返牌，可选择不拾");
        }
        selectCardTypes(cards);
        renderHoleCards(currentState);
        renderHand(currentState);
        return;
    }
    const cards = getMyCards(currentState).slice(0, currentState?.hole_cards_count || 4);
    selectCardTypes(cards);
    renderHand(currentState);
}

function applySuggestion(suggestions) {
    if (!suggestions.length) {
        showToast("暂无建议");
        return;
    }
    selectCardTypes(suggestions[0].cards || []);
    renderHand(currentState);
    showToast(`已选择：${suggestions[0].label || "推荐出牌"}`);
}

function startGame() {
    send({ action: "start_game" });
}

function selectSeat(seat) {
    send({ action: "select_seat", seat });
}

function leaveRoom() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        send({ action: "leave" });
        ws.close();
    }
    ws = null;
    currentState = null;
    clearSelectedCards();
    showScreen("lobby");
    loadRooms();
}

function backToLobby() {
    $("resultModal").hidden = true;
    leaveRoom();
}

function continueNextGame() {
    $("resultModal").hidden = true;
    clearSelectedCards();
    startGame();
}

function showResult(result) {
    $("resultTitle").textContent = "本局结束";
    const bankerText = (result.new_bankers || []).map(seatName).join("、") || "--";
    const winnerText = result.winner_team === "banker" ? "庄家" : "闲家";
    const nextLevel = result.next_level || "--";
    const multiplier = result.koudi_multiplier ?? 0;
    const bottomBaseScore = result.bottom_base_score ?? 0;
    const koudiDetail = multiplier > 0 ? `（底${bottomBaseScore} × ${multiplier}）` : "";
    $("resultBody").innerHTML = `
        <div class="result-summary">
            <div><span>闲家得分</span><strong>${escapeHtml(result.score_now ?? 0)}</strong></div>
            <div><span>闲家扣底分</span><strong>${escapeHtml(result.xianjia_koudi_score ?? 0)}${escapeHtml(koudiDetail)}</strong></div>
            <div><span>胜方</span><strong>${escapeHtml(winnerText)}</strong></div>
            <div><span>升级</span><strong>${escapeHtml(result.levels ?? 0)} 级</strong></div>
            <div><span>下局庄家</span><strong>${escapeHtml(bankerText)}</strong></div>
            <div><span>下局级牌</span><strong>${escapeHtml(nextLevel)}</strong></div>
        </div>
    `;
    $("resultModal").hidden = false;
}

function send(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        showToast("尚未连接房间");
        return;
    }
    ws.send(JSON.stringify(payload));
}

function getPlayer(state, seat) {
    return (state?.players || []).find((player) => player.seat === seat);
}

function getMyCards(state) {
    const me = getPlayer(state, mySeat);
    const cards = [...(me?.cards || [])];
    if (state?.phase === "koupai" && state.koupai_picker === mySeat && state.hole_cards?.length) {
        cards.push(...state.hole_cards.map((card) => card.card_type || card));
    }
    return sortCardsForDisplay(cards, state);
}

function sortCardsForDisplay(cardTypes, state) {
    const level = state?.now_level || currentState?.now_level || "A";
    const trumpColor = state?.now_color || currentState?.now_color || "";
    return [...cardTypes].sort((a, b) => compareCardsForDisplay(a, b, level, trumpColor));
}

function compareCardsForDisplay(a, b, level, trumpColor) {
    const cardA = parseCard(a);
    const cardB = parseCard(b);
    const groupA = displayGroup(cardA, level, trumpColor);
    const groupB = displayGroup(cardB, level, trumpColor);
    if (groupA !== groupB) {
        return groupA - groupB;
    }
    if (groupA === 0) {
        return trumpSortKey(cardA, level, trumpColor) - trumpSortKey(cardB, level, trumpColor)
            || rankSort(cardA, cardB)
            || colorSort(cardA, cardB);
    }
    return colorSort(cardA, cardB) || rankSort(cardA, cardB);
}

function displayGroup(card, level, trumpColor) {
    if (isDisplayTrump(card, level, trumpColor)) {
        return 0;
    }
    return 1 + (COLOR_ORDER[card.color] ?? 9);
}

function isDisplayTrump(card, level, trumpColor) {
    if (card.color === "z") {
        return true;
    }
    if (card.name === level) {
        return true;
    }
    if (FIXED_TRUMP_NAMES.has(card.name)) {
        return true;
    }
    return Boolean(trumpColor && card.color === trumpColor);
}

function trumpSortKey(card, level, trumpColor) {
    if (card.color === "z") {
        return card.name === "W" ? 0 : 1;
    }
    if (card.name === level) {
        return 10 + (COLOR_ORDER[card.color] ?? 9);
    }
    if (FIXED_TRUMP_NAMES.has(card.name)) {
        return 20 + (FIXED_TRUMP_ORDER[card.name] ?? 9) * 10 + (COLOR_ORDER[card.color] ?? 9);
    }
    if (trumpColor && card.color === trumpColor) {
        return 100 + numericRank(card);
    }
    return 200 + numericRank(card);
}

function numericRank(card) {
    return Number(card.rank || String(card.cardType || "").split("-")[1]) || 0;
}

function colorSort(cardA, cardB) {
    return (COLOR_ORDER[cardA.color] ?? 9) - (COLOR_ORDER[cardB.color] ?? 9);
}

function rankSort(cardA, cardB) {
    return numericRank(cardA) - numericRank(cardB);
}

function relativePositions(seat) {
    if (seat < 0) {
        return { left: 1, top: 2, right: 3 };
    }
    return {
        left: (seat + 1) % 4,
        top: (seat + 2) % 4,
        right: (seat + 3) % 4
    };
}

function getClientId() {
    const key = "chaoyang_client_id";
    try {
        let value = localStorage.getItem(key);
        if (!value) {
            const randomPart = crypto?.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
            value = `client-${randomPart}`;
            localStorage.setItem(key, value);
        }
        return value;
    } catch (error) {
        return `client-${Date.now()}-${Math.random()}`;
    }
}

function seatName(seat) {
    return ["南", "西", "北", "东"][seat] || "玩家";
}

function parseCard(cardTypeOrDict) {
    if (typeof cardTypeOrDict === "object" && cardTypeOrDict !== null) {
        return {
            cardType: cardTypeOrDict.card_type,
            name: cardTypeOrDict.name,
            color: cardTypeOrDict.color,
            rank: cardTypeOrDict.rank
        };
    }

    const [name, rank, color] = String(cardTypeOrDict).split("-");
    return { cardType: cardTypeOrDict, name, color, rank: Number(rank) || 0 };
}

function renderCards(cards, options = {}) {
    if (!cards.length) {
        return '<span class="empty-zone">未出</span>';
    }
    return cards.map((card) => cardHtml(parseCard(card), options)).join("");
}

function plainCards(cards) {
    return cards.map((card) => {
        const parsed = parseCard(card);
        const rank = parsed.name === "w" ? "小王" : parsed.name === "W" ? "大王" : parsed.name;
        return `${rank}${COLOR_SYMBOLS[parsed.color] || ""}`;
    }).join(" ");
}

const LIANG_TYPE_LABELS = {
    danlian: "单对子",
    shuanglian: "双连对",
    threeking: "三王",
    duolian: "多连对"
};

function inferLiangzhuColor(cardTypes) {
    for (const cardType of cardTypes) {
        const card = parseCard(cardType);
        if (card.color && card.color !== "z") {
            return card.color;
        }
    }
    return "z";
}

function cardHtml(card, options = {}) {
    const isRed = card.color === "b" || card.color === "d";
    const rank = card.name === "w" ? "小王" : card.name === "W" ? "大王" : card.name;
    const classNames = [
        "card",
        options.small ? "small" : "",
        options.selectable ? "selectable" : "",
        options.selected ? "selected" : "",
        String(rank).length > 1 ? "long-rank" : "",
        options.disabled ? "disabled" : "",
        isRed ? "red" : ""
    ].filter(Boolean).join(" ");
    const suit = COLOR_SYMBOLS[card.color] || "";
    const data = options.cardType ? ` data-card="${escapeHtml(options.cardType)}"` : "";
    const indexData = Number.isInteger(options.index) ? ` data-index="${options.index}"` : "";
    const offerIndexData = Number.isInteger(options.offerIndex) ? ` data-offer-index="${options.offerIndex}"` : "";
    const style = Number.isInteger(options.index) ? ` style="--card-index: ${options.index + 1}"` : "";
    const tagName = options.selectable ? "button" : "span";
    const typeAttr = options.selectable ? ' type="button"' : ' role="img"';
    return `
        <${tagName} class="${classNames}"${typeAttr}${data}${indexData}${offerIndexData}${style} aria-label="${escapeHtml(rank + suit)}">
            <span class="rank">${escapeHtml(rank)}</span>
            <span class="suit">${escapeHtml(suit)}</span>
        </${tagName}>
    `;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

window.__chaoyangDebug = () => ({
    currentRoomId,
    mySeat,
    currentState,
    selectedCards: [...selectedCards],
    selectedCardKeys: [...selectedCardKeys]
});
