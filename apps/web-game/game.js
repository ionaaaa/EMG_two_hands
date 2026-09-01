"use strict";

/* ===== 手势协议（与 emg_api.py 固定一致） ===== */
const GESTURE_SET = new Set(["rest", "fist", "open-palm", "pinch"]);
const NOTE_GESTURES = ["fist", "open-palm", "pinch"];
const HANDS = ["left", "right"];
const HINT_LEAD = 1600;

const GESTURE_COLOR = {
  fist: "#ff4d6d",
  "open-palm": "#36d1ff",
  pinch: "#ffd23f",
  rest: "#5ad17a"
};
const GESTURE_LABEL = { rest: "REST", fist: "FIST", "open-palm": "OPEN", pinch: "PINCH" };
const GESTURE_CN = { rest: "放松", fist: "握拳", "open-palm": "摊掌", pinch: "捏合" };
const HAND_CN = { left: "左", right: "右" };
const GRADE_CN = { perfect: "完美", good: "良好" };
const DEFAULT_RUNTIME_MAPPING = {
  commands: {
    A: { display_name_zh: "红色音符/指令 A", game_gesture: "fist" },
    B: { display_name_zh: "蓝色音符/指令 B", game_gesture: "open-palm" },
    none: { display_name_zh: "无操作", game_gesture: "rest" }
  },
  resolved_mapping: { rest: "none", fist: "A", "open-palm": "B", pinch: "none" }
};
const REASON_CN = {
  timeout: "超时未击打",
  wrong: "手势错误",
  released: "长条中途松手",
  changed: "长条中途换手势"
};

/* ===== 小星星旋律（C大调）：[音名, 拍数] ===== */
const NOTE_FREQ = { C: 261.63, D: 293.66, E: 329.63, F: 349.23, G: 392.00, A: 440.00 };
const TWINKLE_MELODY = [
  ["C",1],["C",1],["G",1],["G",1],["A",1],["A",1],["G",2],
  ["F",1],["F",1],["E",1],["E",1],["D",1],["D",1],["C",2],
  ["G",1],["G",1],["F",1],["F",1],["E",1],["E",1],["D",2],
  ["G",1],["G",1],["F",1],["F",1],["E",1],["E",1],["D",2]
];
// 音高 → 手势：只保留红(fist)/蓝(open-palm)，黄(pinch)已移除
function pitchToGesture(pitch) {
  if (pitch === "C" || pitch === "D") return "fist";
  if (pitch === "E" || pitch === "F") return "open-palm";
  // G, A 原为黄(pinch)，现并入红蓝
  if (pitch === "G") return "fist";
  return "open-palm"; // A
}

/* ===== 游戏状态 ===== */
const state = {
  running: false,
  ended: false,
  mode: "both",
  score: 0,
  combo: 0,
  maxCombo: 0,
  energy: 0,
  energyMax: 100,
  perfectCount: 0,
  goodCount: 0,
  missCount: 0,
  hands: { left: "rest", right: "rest" },
  confidence: { left: 0, right: 0 },
  lastGestureAt: { left: 0, right: 0 },
  live: {
    left: { gesture: "rest", confidence: 0, probs: {}, gameControl: false, modelType: "demo", connected: false, source: "", lastAt: 0 },
    right: { gesture: "rest", confidence: 0, probs: {}, gameControl: false, modelType: "demo", connected: false, source: "", lastAt: 0 }
  },
  bridgeOnline: false,
  bridgeConnecting: false,
  holding: { left: null, right: null },
  holdMissCount: { left: 0, right: 0 },  // hold期间手势不匹配的连续帧计数，超过阈值才判失手
  holdMissThreshold: 5,  // 连续5帧（≈0.5s实时）不匹配才判released/changed，容忍瞬时抖动
  handStats: {
    left: { perfect: 0, good: 0, miss: 0, score: 0, combo: 0, maxCombo: 0 },
    right: { perfect: 0, good: 0, miss: 0, score: 0, combo: 0, maxCombo: 0 }
  },
  chart: [],
  startTime: 0,
  lastTime: 0,
  effects: [],
  particles: [],
  stars: [],
  flash: null,
  hint: { left: null, right: null },
  burstUntil: 0,
  bpm: 80,
  musicOn: true,     // 击中音效开关（开则击中时合成播放该音）
  message: "红=握拳 蓝=摊掌 黄=捏合。击中音符即奏响对应音高，漏音则静音。",
  runtimeMapping: DEFAULT_RUNTIME_MAPPING,

  // 判定参数
  fallDuration: 7000,        // ms：音符从顶部落到判定线的时间（放慢，给更多反应时间）
  judgeWindowPerfect: 100,
  judgeWindowGood: 1600,     // 前段命中窗口：到达判定线前约3.2cm（≈1600ms）即可开始命中（tap用）
  judgeWindowHoldHead: 300,  // hold头命中窗口：长条头部需接近判定线（±300ms）才命中，避免长条提前消失
  judgeWindowMiss: 800,      // 后段：音符彻底离开判定线（过线后800ms）前仍可命中

  // 画布几何
  judgeY: 560,
  leftX: 0,
  rightX: 0,
  trackW: 0,
  noteW: 0,
  pulse: 0
};

const el = {
  canvas: document.querySelector("#gameCanvas"),
  score: document.querySelector("#score"),
  combo: document.querySelector("#combo"),
  bestCombo: document.querySelector("#bestCombo"),
  energyLabel: document.querySelector("#energyLabel"),
  energyPct: document.querySelector("#energyPct"),
  energyFill: document.querySelector("#energyFill"),
  gameMessage: document.querySelector("#gameMessage"),
  bridgeStatus: document.querySelector("#bridgeStatus"),
  bpmSlider: document.querySelector("#bpmSlider"),
  bpmVal: document.querySelector("#bpmVal"),
  musicToggle: document.querySelector("#musicToggle"),
  musicState: document.querySelector("#musicState"),
  livePanels: [...document.querySelectorAll(".live-recog")],
  scoreboards: [...document.querySelectorAll(".scoreboard")],
  buttons: [...document.querySelectorAll(".gesture-btn")],
  modeBtns: [...document.querySelectorAll(".mode-btn")],
  leftBox: document.querySelector("#leftHandBox"),
  rightBox: document.querySelector("#rightHandBox"),
  eventLog: document.querySelector("#eventLog"),
  startBtn: document.querySelector("#startBtn")
};

const ctx = el.canvas.getContext("2d");

/* ===== 画布尺寸 ===== */
function resizeCanvas() {
  const rect = el.canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  el.canvas.width = Math.round(rect.width * scale);
  el.canvas.height = Math.round(rect.height * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.imageSmoothingEnabled = false;

  const w = rect.width;
  state.judgeY = computeJudgeY(rect);
  state.trackW = (w - 16) / 2;
  state.leftX = 8 + state.trackW / 2;
  state.rightX = 8 + state.trackW + state.trackW / 2;
  state.noteW = Math.max(60, state.trackW * 0.46);
  initStars(w, rect.height);
}

/* ===== 判定线对齐右侧手势块 ===== */
function computeJudgeY(rect) {
  if (window.innerWidth <= 920) return rect.height - 96;
  const lh = el.leftBox.getBoundingClientRect();
  const rh = el.rightBox.getBoundingClientRect();
  let targetY = 0;
  let have = false;
  if (state.mode === "both") {
    if (lh.height > 0 && rh.height > 0) { targetY = (lh.bottom + rh.top) / 2; have = true; }
    else if (lh.height > 0) { targetY = lh.bottom; have = true; }
    else if (rh.height > 0) { targetY = rh.top; have = true; }
  } else if (state.mode === "left") {
    if (lh.height > 0) { targetY = (lh.top + lh.bottom) / 2; have = true; }
  } else {
    if (rh.height > 0) { targetY = (rh.top + rh.bottom) / 2; have = true; }
  }
  let jy = have ? targetY - rect.top : rect.height - 96;
  return Math.max(rect.height * 0.28, Math.min(rect.height - 40, jy));
}

function updateJudgeY() {
  const rect = el.canvas.getBoundingClientRect();
  state.judgeY = computeJudgeY(rect);
}

function initStars(w, h) {
  state.stars = [];
  for (let i = 0; i < 70; i++) {
    state.stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 1.6 + 0.4,
      tw: Math.random() * Math.PI * 2,
      sp: Math.random() * 0.02 + 0.005
    });
  }
}

/* ===== 工具 ===== */
function normalizeGesture(gesture) {
  const value = String(gesture || "").trim().toLowerCase().replaceAll("_", "-");
  const aliases = {
    relax: "rest", idle: "rest",
    open: "open-palm", palm: "open-palm", openpalm: "open-palm",
    "open-hand": "open-palm", handopen: "open-palm",
    fist: "fist", grip: "fist", jump: "fist",
    pinch: "pinch", eat: "pinch", bite: "pinch",
    "index-thumb-pinch": "pinch", thumbindexpinch: "pinch"
  };
  return aliases[value] || value;
}

function normalizeHand(value) {
  const h = String(value || "").trim().toLowerCase();
  if (h === "l" || h === "left") return "left";
  if (h === "r" || h === "right") return "right";
  return "left";
}

function applyRuntimeMappingConfig(payload) {
  const commands = payload && payload.commands;
  const mapping = payload && payload.resolved_mapping;
  if (!commands || !mapping) return false;
  if (mapping.rest !== "none" || mapping.pinch !== "none") return false;
  if (new Set([mapping.fist, mapping["open-palm"]]).size !== 2) return false;
  if (![mapping.fist, mapping["open-palm"]].every((command) => command === "A" || command === "B")) return false;
  for (const command of ["A", "B", "none"]) {
    if (!commands[command] || !GESTURE_SET.has(normalizeGesture(commands[command].game_gesture))) return false;
  }
  state.runtimeMapping = { commands, resolved_mapping: mapping };
  return true;
}

function mapRealtimeGesture(gesture) {
  const normalized = normalizeGesture(gesture);
  const runtime = state.runtimeMapping || DEFAULT_RUNTIME_MAPPING;
  const command = runtime.resolved_mapping[normalized] || "none";
  const target = runtime.commands[command] && runtime.commands[command].game_gesture;
  return normalizeGesture(target || "rest");
}

async function loadRuntimeMappingConfig() {
  try {
    const response = await fetch("runtime-config.json", { cache: "no-store" });
    if (!response.ok) return false;
    return applyRuntimeMappingConfig(await response.json());
  } catch (_error) {
    return false;
  }
}

function logEvent(label, detail) {
  const row = document.createElement("div");
  row.className = "event-row";
  row.innerHTML = `<strong>${label}</strong><span>${detail}</span>`;
  el.eventLog.prepend(row);
  while (el.eventLog.children.length > 9) el.eventLog.lastElementChild.remove();
}

function setBridgeStatus(status) {
  if (el.bridgeStatus) el.bridgeStatus.textContent = status;
  state.bridgeOnline = status === "online";
  state.bridgeConnecting = status === "connecting";
  refreshLiveRecog();
}

/* ===== 击中音效：Web Audio 合成（钟琴音色） ===== */
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) {
    try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { audioCtx = null; }
  }
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}
function playTone(pitch, durationMs) {
  if (!state.musicOn) return;
  const ac = ensureAudio();
  if (!ac) return;
  const freq = NOTE_FREQ[pitch] || 440;
  const dur = Math.max(0.18, (durationMs || 400) / 1000);
  const t0 = ac.currentTime;
  const osc = ac.createOscillator();
  const gain = ac.createGain();
  osc.type = "triangle";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, t0);
  gain.gain.exponentialRampToValueAtTime(0.32, t0 + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  osc.connect(gain).connect(ac.destination);
  osc.start(t0);
  osc.stop(t0 + dur + 0.05);
}

/* ===== 谱面生成：基于小星星旋律，一音一块 ===== */
function generateChart() {
  const chart = [];
  const beat = Math.max(200, Math.round(60000 / state.bpm));
  const startT = state.fallDuration + 200; // 第一个音落到判定线的时刻
  const hands = state.mode === "both" ? ["left", "right"] : [state.mode];
  const gap = beat * 3.7;        // 音符间间隔，避免色块粘连
  const phraseGap = beat * 2.7;  // 乐句间停顿（句末长音后）
  let t = startT;
  let lastGesture = null;  // 用于交替打乱，避免相邻同色
  for (const [pitch, beats] of TWINKLE_MELODY) {
    const dur = beats * beat;
    let gesture = pitchToGesture(pitch);
    // 相邻音符同色时翻转红蓝，保证颜色不连着
    if (lastGesture && gesture === lastGesture) {
      gesture = gesture === "fist" ? "open-palm" : "fist";
    }
    lastGesture = gesture;
    const isLong = beats >= 2; // 长音(≥2拍)用长条hold，短音用tap
    for (const hand of hands) {
      if (isLong) {
        chart.push({
          id: chart.length, time: t, hand, gesture, type: "hold",
          tailTime: t + dur, pitch, dur, judged: false, holding: false, tailJudged: false, headGrade: null
        });
      } else {
        chart.push({ id: chart.length, time: t, hand, gesture, type: "tap", pitch, dur, judged: false });
      }
    }
    t += dur;
    // 句末长音后加乐句停顿，普通音之间加间隔，保证色块不粘连
    t += isLong ? phraseGap : gap;
  }
  return chart;
}

/* ===== 时间 ===== */
function songTime() {
  return state.running ? performance.now() - state.startTime : 0;
}
function noteY(time) {
  const progress = (time - songTime()) / state.fallDuration;
  return state.judgeY - progress * state.judgeY;
}

/* ===== 判定 ===== */
function applyGrade(note, grade) {
  const pts = grade === "perfect" ? 100 : 50;
  const gain = pts + state.combo * 2;
  state.score += gain;
  state.combo += 1;
  state.maxCombo = Math.max(state.maxCombo, state.combo);
  state.energy = Math.min(state.energyMax, state.energy + (grade === "perfect" ? 10 : 7));
  if (grade === "perfect") state.perfectCount++; else state.goodCount++;

  // 按手统计
  const hs = state.handStats[note.hand];
  hs.score += gain;
  hs.combo += 1;
  hs.maxCombo = Math.max(hs.maxCombo, hs.combo);
  if (grade === "perfect") hs.perfect++; else hs.good++;

  // 击中才奏响对应音高（漏音则静音）
  if (note.pitch) playTone(note.pitch, note.dur);

  const x = note.hand === "left" ? state.leftX : state.rightX;
  spawnEffect(x, state.judgeY, grade);
  spawnParticles(x, state.judgeY, GESTURE_COLOR[note.gesture], grade === "perfect" ? 16 : 10);

  const fcolor = grade === "perfect" ? "#36d1ff" : "#5ad17a";
  spawnFlash(GRADE_CN[grade] + "!", `${HAND_CN[note.hand]}手 ${GESTURE_CN[note.gesture]} · +${gain} · 连击 ${state.combo}`, fcolor);
  state.message = `✓ 命中 ${GRADE_CN[grade]} · ${HAND_CN[note.hand]}手 ${GESTURE_CN[note.gesture]} · +${gain} · 连击 ${state.combo}`;

  if (state.energy >= state.energyMax && state.burstUntil < performance.now()) triggerBurst();
}

function missNote(note, reason) {
  if (note.judged && note.type === "tap") return;
  note.judged = true;
  note.tailJudged = true;
  for (const h of HANDS) if (state.holding[h] === note) state.holding[h] = null;
  state.combo = 0;
  state.missCount++;
  const hs = state.handStats[note.hand];
  hs.miss++;
  hs.combo = 0;
  const x = note.hand === "left" ? state.leftX : state.rightX;
  spawnEffect(x, state.judgeY, "miss");
  spawnParticles(x, state.judgeY, "#8a8f9c", 6);

  const reasonText = REASON_CN[reason] || "失误";
  spawnFlash("失误", `${HAND_CN[note.hand]}手 · ${reasonText} · 连击中断`, "#ff4d6d");
  state.message = `✗ 失误 · ${HAND_CN[note.hand]}手 · ${reasonText} · 连击清零`;
}

function handleGesture(hand, gesture) {
  const now = songTime();
  if (gesture === "rest") {
    const held = state.holding[hand];
    if (held && !held.tailJudged) {
      state.holdMissCount[hand] += 1;
      if (state.holdMissCount[hand] >= state.holdMissThreshold) missNote(held, "released");
    }
    return;
  }
  const held = state.holding[hand];
  if (held && !held.tailJudged && gesture !== held.gesture) {
    state.holdMissCount[hand] += 1;
    if (state.holdMissCount[hand] >= state.holdMissThreshold) missNote(held, "changed");
    // hold中途手势不匹配时，不继续找新的tap命中，直接返回
    return;
  }
  if (held && !held.tailJudged && gesture === held.gesture) {
    state.holdMissCount[hand] = 0;  // 恢复正确手势，清零
  }

  let best = null, bestDt = Infinity;
  for (const n of state.chart) {
    if (n.judged) continue;
    if (n.hand !== hand) continue;
    if (n.type === "hold" && n.holding) continue;
    const diff = n.time - now;
    if (diff < HINT_LEAD && diff > -state.judgeWindowMiss && Math.abs(diff) < bestDt) { best = n; bestDt = Math.abs(diff); }
  }
  if (!best) return;
  // tap和hold头部都用同一宽窗口（前1cm即可命中），保证长条也能提前接住
  if (bestDt > state.judgeWindowGood) return;  // 音符未进入判定窗口，忽略，避免实时手势提前误判
  if (best.gesture !== gesture) return;  // 手势不匹配：不立刻miss，等音符自然超时由update判定timeout

  best.judged = true;
  const grade = bestDt <= state.judgeWindowPerfect ? "perfect" : "good";
  applyGrade(best, grade);
  if (best.type === "hold") {
    best.holding = true;
    best.headGrade = grade;
    state.holding[hand] = best;
  }
}

/* ===== 反馈：中央大字 + 消息条 ===== */
function spawnFlash(text, sub, color) {
  state.flash = { text, sub, color, life: 52, maxLife: 52 };
}

/* ===== 效果 / 粒子 ===== */
function spawnEffect(x, y, grade) {
  const text = grade === "perfect" ? "PERFECT" : grade === "good" ? "GOOD" : "MISS";
  const color = grade === "perfect" ? "#36d1ff" : grade === "good" ? "#5ad17a" : "#ff4d6d";
  state.effects.push({ x, y, text, color, life: 48, vy: -0.8 });
}
function spawnParticles(x, y, color, n) {
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const s = Math.random() * 4 + 1.5;
    state.particles.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 1, life: 40, color, size: Math.random() * 3 + 2 });
  }
}
function triggerBurst() {
  state.burstUntil = performance.now() + 800;
  state.energy = 0;
  state.score += 300;
  state.message = "能量满载！全屏爆发 +300";
  spawnFlash("能量爆发!", "全屏 +300", "#ffd23f");
  logEvent("Burst", "energy full");
  for (let i = 0; i < 40; i++) {
    state.particles.push({
      x: state.leftX + (state.rightX - state.leftX) * Math.random(),
      y: state.judgeY,
      vx: (Math.random() - 0.5) * 10,
      vy: -Math.random() * 8 - 2,
      life: 60, color: ["#ff4d6d", "#36d1ff", "#ffd23f"][i % 3], size: Math.random() * 4 + 2
    });
  }
}

/* ===== 核心接入：dispatchGesture ===== */
function dispatchGesture(gesture, confidence = 0.85, hand = "left") {
  gesture = normalizeGesture(gesture);
  hand = normalizeHand(hand);
  if (!GESTURE_SET.has(gesture)) return;

  state.hands[hand] = gesture;
  state.confidence[hand] = Math.max(0, Math.min(1, confidence));
  state.lastGestureAt[hand] = performance.now();
  updateHandUi(hand, gesture);
  refreshLiveRecog();

  if (state.running && !state.ended) handleGesture(hand, gesture);
  logEvent(`${hand.toUpperCase()} ${GESTURE_LABEL[gesture]}`, `${Math.round(confidence * 100)}%`);
}

function updateLiveRecog() {
  el.livePanels.forEach((panel) => {
    const hand = panel.dataset.hand;
    const live = state.live[hand];
    const gesture = state.hands[hand] || "rest";
    const conf = state.confidence[hand] || 0;
    const color = GESTURE_COLOR[gesture] || "#9a9a9a";

    const big = panel.querySelector(".live-gesture-big");
    const cn = panel.querySelector(".live-gesture-cn");
    const fill = panel.querySelector(".live-conf-fill");
    const num = panel.querySelector(".live-conf-num");
    const main = panel.querySelector(".live-main");
    const probs = panel.querySelector(".live-probs");
    const ctrl = panel.querySelector(".live-ctrl-state");

    if (big) big.textContent = GESTURE_LABEL[gesture] || gesture;
    if (cn) cn.textContent = GESTURE_CN[gesture] || gesture;
    const pct = Math.round(conf * 100);
    if (fill) fill.style.width = pct + "%";
    if (num) num.textContent = pct + "%";

    if (main) {
      main.style.setProperty("--live-color", color);
      main.style.setProperty("--live-glow", color);
      main.style.setProperty("--live-glow-op", gesture === "rest" ? "0.12" : "0.42");
    }

    const ordered = ["rest", "fist", "open-palm", "pinch"];
    const liveProbs = live.probs || {};
    let topGesture = gesture, topProb = -1;
    for (const g of ordered) {
      const p = Number(liveProbs[g] || 0);
      if (p > topProb) { topProb = p; topGesture = g; }
    }
    if (probs) {
      probs.querySelectorAll(".prob-row").forEach((row) => {
        const g = row.dataset.gesture;
        const p = Math.max(0, Math.min(1, Number(liveProbs[g] || 0)));
        const c = GESTURE_COLOR[g] || "#9a9a9a";
        row.style.setProperty("--pcolor", c);
        row.style.setProperty("--pglow", c);
        row.querySelector(".prob-fill").style.width = Math.round(p * 100) + "%";
        row.querySelector(".prob-val").textContent = Math.round(p * 100) + "%";
        row.classList.toggle("is-top", g === topGesture && topProb > 0);
      });
    }

    if (ctrl) {
      ctrl.textContent = live.gameControl ? "● 游戏控制已开启" : "";
    }

    const pill = panel.querySelector(".conn-pill");
    const ctext = panel.querySelector(".conn-text");
    const src = panel.querySelector(".live-source");
    let label, pillOnline, pillConnecting, sourceText;
    if (!state.bridgeOnline && !state.bridgeConnecting) {
      label = "未连接";
      pillOnline = false; pillConnecting = false;
      sourceText = "等待 emg_live_marker 启动…";
    } else if (state.bridgeConnecting && !live.connected) {
      label = "连接中";
      pillOnline = false; pillConnecting = true;
      sourceText = "正在连接…";
    } else if (live.connected) {
      label = "已连接";
      pillOnline = true; pillConnecting = false;
      sourceText = `${HAND_CN[hand]}手 · emg_live_marker 实时数据`;
    } else {
      label = "等待数据";
      pillOnline = false; pillConnecting = true;
      sourceText = `等待${HAND_CN[hand]}手数据…`;
    }
    if (pill) {
      pill.classList.toggle("online", pillOnline);
      pill.classList.toggle("connecting", pillConnecting);
    }
    if (ctext) ctext.textContent = label;
    if (src) src.textContent = sourceText;
    panel.classList.toggle("offline", !pillOnline);
  });
}

function refreshLiveRecog() { updateLiveRecog(); }

function applyLiveGesture(hand) {
  ensureAudio();  // 实时数据可能在用户无交互时到达，确保音频上下文已激活
  const live = state.live[hand];
  const recognizedGesture = live.gesture;
  const gesture = mapRealtimeGesture(recognizedGesture);
  const prev = state.hands[hand];
  state.hands[hand] = gesture;
  state.confidence[hand] = live.confidence;
  state.lastGestureAt[hand] = performance.now();
  updateHandUi(hand, gesture);
  refreshLiveRecog();
  if (live.gameControl && state.running && !state.ended) {
    handleGesture(hand, gesture);
    if (gesture !== prev) {
      const command = state.runtimeMapping.resolved_mapping[recognizedGesture] || "none";
      logEvent(`LIVE ${HAND_CN[hand]} ${GESTURE_LABEL[recognizedGesture]}`, `→ 指令 ${command} · ${Math.round(live.confidence * 100)}%`);
    }
  }
}

function updateHandUi(hand, gesture) {
  el.buttons.forEach((b) => {
    if (b.dataset.hand === hand) b.classList.toggle("active", b.dataset.gesture === gesture);
  });
}

/* ===== 实时 API（SSE） ===== */
function connectRealtimeGestureApi(primary = "http://127.0.0.1:8766/events", fallback = "http://127.0.0.1:8765/events") {
  if (!window.EventSource) { setBridgeStatus("unsupported"); return null; }
  let openedOnce = false;
  let triedFallback = false;
  let source = null;
  const onGesture = (event) => {
    try {
      const p = JSON.parse(event.data);
      const hand = normalizeHand(p.hand);
      const live = state.live[hand];
      live.gesture = normalizeGesture(p.gesture);
      live.confidence = Math.max(0, Math.min(1, Number(p.confidence ?? 0)));
      live.probs = p.probs || {};
      live.gameControl = !!p.game_control;
      live.modelType = p.model_type || "demo";
      live.source = p.source || (triedFallback ? "emg_api" : "emg_live_marker");
      live.lastAt = performance.now();
      if (typeof p.connected === "boolean") live.connected = p.connected;
      updateLiveRecog();
      applyLiveGesture(hand);
    } catch (e) { logEvent("LIVE error", "bad gesture payload"); }
  };
  const onHandStatus = (event) => {
    try {
      const p = JSON.parse(event.data);
      const hand = normalizeHand(p.hand);
      const live = state.live[hand];
      if (typeof p.connected === "boolean") live.connected = p.connected;
      if (typeof p.game_control === "boolean") live.gameControl = p.game_control;
      refreshLiveRecog();
    } catch (e) { /* ignore malformed status */ }
  };
  const onMapping = (event) => {
    try {
      if (applyRuntimeMappingConfig(JSON.parse(event.data))) {
        for (const hand of HANDS) if (state.live[hand].connected) applyLiveGesture(hand);
        logEvent("MAPPING", "游戏指令映射已更新");
      }
    } catch (e) { /* ignore malformed mapping */ }
  };
  const onMappingTest = (event) => {
    try {
      const p = JSON.parse(event.data);
      if (p.test === true) logEvent("MAPPING TEST", p.message || "映射层测试");
    } catch (e) { /* ignore malformed test feedback */ }
  };
  const open = (url, isFallback) => {
    if (source) source.close();
    source = new EventSource(url);
    setBridgeStatus("connecting");
    source.onopen = () => {
      openedOnce = true;
      setBridgeStatus("online");
      logEvent("LIVE", isFallback ? "emg_api bridge connected" : "emg_live_marker connected");
    };
    source.onerror = () => {
      setBridgeStatus("offline");
      if (!openedOnce && !isFallback && !triedFallback) {
        triedFallback = true;
        source.close();
        setTimeout(() => open(fallback, true), 1200);
      }
    };
    source.addEventListener("gesture", onGesture);
    source.addEventListener("hand_status", onHandStatus);
    source.addEventListener("mapping", onMapping);
    source.addEventListener("mapping_test", onMappingTest);
    source.addEventListener("status", (event) => setBridgeStatus(event.data || "online"));
  };
  open(primary, false);
  return source;
}

/* ===== 模式 ===== */
function setMode(mode) {
  state.mode = mode;
  el.modeBtns.forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  el.leftBox.style.display = (mode === "both" || mode === "left") ? "" : "none";
  el.rightBox.style.display = (mode === "both" || mode === "right") ? "" : "none";
  logEvent("Mode", mode);
  startGame();
  requestAnimationFrame(updateJudgeY);
}

/* ===== 按钮提示：该按哪个 ===== */
function updateHints() {
  const hint = { left: null, right: null };
  if (state.running && !state.ended) {
    const now = songTime();
    const lead = HINT_LEAD;
    for (const n of state.chart) {
      if (n.judged) continue;
      const dt = n.time - now;
      if (dt < lead && dt > -state.judgeWindowMiss && !hint[n.hand]) hint[n.hand] = n.gesture;
    }
    for (const hand of HANDS) {
      const h = state.holding[hand];
      if (h && !h.tailJudged) hint[hand] = h.gesture;
    }
  }
  state.hint = hint;
  el.buttons.forEach((b) => {
    const g = hint[b.dataset.hand];
    b.classList.toggle("hint", g != null && g !== "rest" && b.dataset.gesture === g);
  });
}

/* ===== 更新循环 ===== */
function update() {
  state.pulse += 0.05;
  if (state.flash) { state.flash.life -= 1; if (state.flash.life <= 0) state.flash = null; }
  updateHints();
  if (!state.running || state.ended) { updateEffects(); updateParticles(); return; }
  const now = songTime();

  for (const n of state.chart) {
    if (n.judged) continue;
    if (now > n.time + state.judgeWindowMiss) missNote(n, "timeout");
  }
  for (const hand of HANDS) {
    const h = state.holding[hand];
    if (!h || h.tailJudged) continue;
    // hold的失手判定统一由handleGesture按实时手势频率计数，update只负责长条完成判定
    if (now >= h.tailTime) {
      h.tailJudged = true;
      state.holding[hand] = null;
      state.score += 60;
      state.handStats[hand].score += 60;
      const x = hand === "left" ? state.leftX : state.rightX;
      spawnEffect(x, state.judgeY, "perfect");
      spawnParticles(x, state.judgeY, GESTURE_COLOR[h.gesture], 12);
      spawnFlash("长条完成!", `${HAND_CN[hand]}手 ${GESTURE_CN[h.gesture]} · +60`, "#5ad17a");
      state.message = `✓ 长条完成 · ${HAND_CN[hand]}手 ${GESTURE_CN[h.gesture]} · +60`;
    }
  }
  const last = state.chart[state.chart.length - 1];
  if (last && now > (last.tailTime || last.time) + 2200) endGame();

  updateEffects();
  updateParticles();
}
function updateEffects() {
  state.effects = state.effects.filter((e) => e.life > 0);
  state.effects.forEach((e) => { e.y += e.vy; e.life -= 1; });
}
function updateParticles() {
  state.particles = state.particles.filter((p) => p.life > 0);
  state.particles.forEach((p) => { p.x += p.vx; p.y += p.vy; p.vy += 0.18; p.life -= 1; });
}

/* ===== 渲染（霓虹风） ===== */
function drawBackground(w, h) {
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "#0a0c1a");
  grad.addColorStop(1, "#12162b");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  state.stars.forEach((s) => {
    s.tw += s.sp;
    const a = 0.35 + Math.abs(Math.sin(s.tw)) * 0.55;
    ctx.globalAlpha = a;
    ctx.fillStyle = "#9fb4ff";
    ctx.fillRect(s.x, s.y, s.r, s.r);
  });
  ctx.globalAlpha = 1;

  const tw = state.trackW;
  drawTrack(8, tw, h, "#1a2a4a", "#0e1530");
  drawTrack(8 + tw, tw, h, "#3a1a4a", "#1a0e30");
  ctx.strokeStyle = "rgba(120,140,200,0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(8 + tw, 0); ctx.lineTo(8 + tw, h);
  ctx.stroke();

  const glow = 8 + Math.sin(state.pulse) * 4;
  ctx.shadowColor = "#36d1ff";
  ctx.shadowBlur = glow;
  ctx.fillStyle = "#36d1ff";
  ctx.fillRect(0, state.judgeY, w, 3);
  ctx.shadowBlur = 0;

  drawJudgeRing(state.leftX, state.judgeY, state.hands.left);
  drawJudgeRing(state.rightX, state.judgeY, state.hands.right);

  ctx.fillStyle = "rgba(160,180,230,0.5)";
  ctx.font = "700 18px 'Courier New', monospace";
  ctx.textAlign = "center";
  ctx.fillText("L", state.leftX, 26);
  ctx.fillText("R", state.rightX, 26);
  ctx.textAlign = "left";

  if (performance.now() < state.burstUntil) {
    ctx.fillStyle = "rgba(54,209,255,0.14)";
    ctx.fillRect(0, 0, w, h);
  }
}

function drawTrack(x, w, h, c1, c2) {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, c2);
  g.addColorStop(1, c1);
  ctx.fillStyle = g;
  ctx.fillRect(x, 0, w, h);
  const gg = ctx.createLinearGradient(0, 0, 0, 120);
  gg.addColorStop(0, "rgba(120,160,255,0.10)");
  gg.addColorStop(1, "rgba(120,160,255,0)");
  ctx.fillStyle = gg;
  ctx.fillRect(x, 0, w, 120);
}

function drawJudgeRing(x, y, gesture) {
  const color = GESTURE_COLOR[gesture] || "#9a9a9a";
  ctx.shadowColor = color;
  ctx.shadowBlur = 16;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, 28, 0, Math.PI * 2);
  ctx.stroke();
  if (gesture !== "rest") {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fill();
}

function drawNote(note) {
  const x = (note.hand === "left" ? state.leftX : state.rightX) - state.noteW / 2;
  const color = GESTURE_COLOR[note.gesture];
  ctx.shadowColor = color;
  ctx.shadowBlur = 18;

  if (note.type === "hold") {
    const headY = noteY(note.time);
    const tailY = noteY(note.tailTime);
    if (note.holding && !note.tailJudged) {
      const remTop = Math.min(tailY, state.judgeY);
      const remH = state.judgeY - remTop;
      ctx.shadowColor = color;
      ctx.shadowBlur = 22;
      ctx.fillStyle = color;
      if (remH > 2) roundRect(x, remTop, state.noteW, remH, 7);
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      if (remH > 6) roundRect(x + 5, remTop + 4, state.noteW - 10, 4, 2);
      const doneH = headY - state.judgeY;
      ctx.fillStyle = "rgba(130,135,150,0.3)";
      if (doneH > 2) roundRect(x, state.judgeY, state.noteW, doneH, 6);
    } else {
      const top = Math.min(headY, tailY);
      const hgt = Math.max(10, Math.abs(tailY - headY));
      ctx.fillStyle = color + "44";
      roundRect(x, top, state.noteW, hgt, 6);
      ctx.fillStyle = color;
      roundRect(x, headY - 11, state.noteW, 18, 6);
    }
  } else {
    const y = noteY(note.time) - 12;
    ctx.fillStyle = color;
    roundRect(x, y, state.noteW, 24, 7);
    ctx.shadowBlur = 0;
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    roundRect(x + 5, y + 4, state.noteW - 10, 4, 2);
  }
  ctx.shadowBlur = 0;
}

function drawHolding() {
  for (const hand of HANDS) {
    const h = state.holding[hand];
    if (!h || h.tailJudged) continue;
    const xc = hand === "left" ? state.leftX : state.rightX;
    const color = GESTURE_COLOR[h.gesture];
    ctx.shadowColor = color;
    ctx.shadowBlur = 18;
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(xc, state.judgeY, 28 + Math.sin(state.pulse * 2) * 3, 0, Math.PI * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

function drawEffects() {
  ctx.textAlign = "center";
  state.effects.forEach((e) => {
    ctx.globalAlpha = Math.max(0, e.life / 48);
    ctx.shadowColor = e.color;
    ctx.shadowBlur = 12;
    ctx.fillStyle = e.color;
    ctx.font = "700 20px 'Courier New', monospace";
    ctx.fillText(e.text, e.x, e.y);
  });
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;
  ctx.textAlign = "left";
}

function drawFlash(w) {
  if (!state.flash) return;
  const f = state.flash;
  const a = Math.max(0, f.life / f.maxLife);
  const scale = 1 + (1 - a) * 0.25;
  ctx.save();
  ctx.globalAlpha = a;
  ctx.textAlign = "center";
  ctx.shadowColor = f.color;
  ctx.shadowBlur = 24;
  ctx.fillStyle = f.color;
  ctx.translate(w / 2, 150);
  ctx.scale(scale, scale);
  ctx.font = "800 44px 'Courier New', monospace";
  ctx.fillText(f.text, 0, 0);
  if (f.sub) {
    ctx.shadowBlur = 8;
    ctx.font = "600 18px 'Courier New', monospace";
    ctx.fillStyle = "#e6e9f5";
    ctx.fillText(f.sub, 0, 34);
  }
  ctx.restore();
  ctx.textAlign = "left";
}

function drawParticles() {
  state.particles.forEach((p) => {
    ctx.globalAlpha = Math.max(0, p.life / 40);
    ctx.fillStyle = p.color;
    ctx.fillRect(p.x, p.y, p.size, p.size);
  });
  ctx.globalAlpha = 1;
}

function drawEndScreen(w, h) {
  if (!state.ended) return;
  ctx.fillStyle = "rgba(10,12,26,0.84)";
  ctx.fillRect(0, 0, w, h);
  ctx.shadowColor = "#36d1ff";
  ctx.shadowBlur = 16;
  ctx.fillStyle = "#e6e9f5";
  ctx.font = "700 34px 'Courier New', monospace";
  ctx.textAlign = "center";
  ctx.fillText("R U N   E N D E D", w / 2, h / 2 - 64);
  ctx.shadowBlur = 0;

  const la = handAcc("left"), ra = handAcc("right");
  if (state.mode === "both") {
    const lh = state.handStats.left, rh = state.handStats.right;
    ctx.font = "600 19px 'Courier New', monospace";
    ctx.fillStyle = "#36d1ff";
    ctx.fillText(`左手  Acc ${Math.round(la * 100)}%  Score ${lh.score}  Combo ${lh.maxCombo}`, w / 2, h / 2 - 22);
    ctx.fillStyle = "#ff4d6d";
    ctx.fillText(`右手  Acc ${Math.round(ra * 100)}%  Score ${rh.score}  Combo ${rh.maxCombo}`, w / 2, h / 2 + 6);

    let winText, winColor;
    if (Math.abs(la - ra) < 0.0005) { winText = "平 局"; winColor = "#ffd23f"; }
    else if (la > ra) { winText = "左手玩家获胜!"; winColor = "#36d1ff"; }
    else { winText = "右手玩家获胜!"; winColor = "#ff4d6d"; }
    ctx.font = "700 26px 'Courier New', monospace";
    ctx.shadowColor = winColor;
    ctx.shadowBlur = 14;
    ctx.fillStyle = winColor;
    ctx.fillText(winText, w / 2, h / 2 + 48);
    ctx.shadowBlur = 0;
  } else {
    const total = state.perfectCount + state.goodCount + state.missCount;
    const acc = total ? Math.round((state.perfectCount + state.goodCount * 0.5) / total * 100) : 0;
    ctx.font = "600 18px 'Courier New', monospace";
    ctx.fillStyle = "#e6e9f5";
    ctx.fillText(`Score ${state.score}   Combo ${state.maxCombo}   Acc ${acc}%`, w / 2, h / 2);
    ctx.fillText(`P ${state.perfectCount}   G ${state.goodCount}   M ${state.missCount}`, w / 2, h / 2 + 28);
  }

  ctx.font = "500 15px 'Courier New', monospace";
  ctx.fillStyle = "#9fb4ff";
  ctx.fillText("按 Start / 空格 再来一次", w / 2, h / 2 + 88);
  ctx.textAlign = "left";
}

function render() {
  const rect = el.canvas.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  drawBackground(w, h);
  drawHolding();
  for (const n of state.chart) if (!n.judged || (n.type === "hold" && n.holding && !n.tailJudged)) drawNote(n);
  drawParticles();
  drawEffects();
  drawFlash(w);
  drawEndScreen(w, h);
}

/* ===== 正确率 / 计分板 ===== */
function handAcc(hand) {
  const hs = state.handStats[hand];
  const total = hs.perfect + hs.good + hs.miss;
  return total ? (hs.perfect + hs.good * 0.5) / total : 0;
}

function updateScoreboards() {
  el.scoreboards.forEach((sb) => {
    const hand = sb.dataset.hand;
    const hs = state.handStats[hand];
    const total = hs.perfect + hs.good + hs.miss;
    const acc = total ? Math.round((hs.perfect + hs.good * 0.5) / total * 100) : 0;
    sb.querySelector(".sb-score").textContent = hs.score;
    sb.querySelector(".sb-acc").textContent = acc + "%";
    sb.querySelector(".sb-combo").textContent = hs.combo;
  });
}

function updateHud() {
  el.score.textContent = state.score;
  el.combo.textContent = state.combo;
  el.bestCombo.textContent = state.maxCombo;
  const pct = Math.round((state.energy / state.energyMax) * 100);
  el.energyLabel.textContent = pct + "%";
  el.energyPct.textContent = pct + "%";
  el.energyFill.style.width = pct + "%";
  el.gameMessage.textContent = state.message;
  updateScoreboards();
}

function loop(time) {
  state.lastTime = time;
  update();
  render();
  updateHud();
  requestAnimationFrame(loop);
}

/* ===== 流程控制 ===== */
function startGame() {
  ensureAudio(); // 预初始化音频上下文（用户交互时）
  state.chart = generateChart();
  state.running = true;
  state.ended = false;
  state.score = 0;
  state.combo = 0;
  state.maxCombo = 0;
  state.energy = 0;
  state.perfectCount = 0;
  state.goodCount = 0;
  state.missCount = 0;
  state.holding = { left: null, right: null };
  state.holdMissCount = { left: 0, right: 0 };
  state.handStats = {
    left: { perfect: 0, good: 0, miss: 0, score: 0, combo: 0, maxCombo: 0 },
    right: { perfect: 0, good: 0, miss: 0, score: 0, combo: 0, maxCombo: 0 }
  };
  state.hands = { left: "rest", right: "rest" };
  state.confidence = { left: 0, right: 0 };
  state.lastGestureAt = { left: 0, right: 0 };
  state.effects = [];
  state.particles = [];
  state.flash = null;
  state.startTime = performance.now();
  state.message = state.mode === "both"
    ? "Go! 双人对战 — 同谱对决，正确率高者获胜"
    : `Go! 单手模式（${state.mode === "left" ? "左" : "右"}手）`;
  updateHandUi("left", "rest");
  updateHandUi("right", "rest");
  refreshLiveRecog();
  updateScoreboards();
  el.startBtn.textContent = "Restart";
  logEvent("Start", `${state.mode} · bpm ${state.bpm}`);
}

function endGame() {
  state.ended = true;
  state.running = false;
  const la = handAcc("left"), ra = handAcc("right");
  if (state.mode === "both") {
    let result;
    if (Math.abs(la - ra) < 0.0005) result = "平局";
    else result = la > ra ? "左手玩家获胜" : "右手玩家获胜";
    state.message = `结束！${result}（左 ${Math.round(la * 100)}% vs 右 ${Math.round(ra * 100)}%）按 Start / 空格 再来一次。`;
  } else {
    state.message = "结束！按 Start / 空格 再来一次。";
  }
  logEvent("End", `L ${Math.round(la * 100)}% vs R ${Math.round(ra * 100)}% · score ${state.score}`);
}

/* ===== 事件绑定 ===== */
el.buttons.forEach((button) => {
  const press = (e) => {
    e.preventDefault();
    const g = button.dataset.gesture;
    if (!state.running && !state.ended && g !== "rest") startGame();
    dispatchGesture(g, 0.9 + Math.random() * 0.08, button.dataset.hand);
  };
  const release = () => {
    if (button.dataset.gesture === "rest") return;
    dispatchGesture("rest", 0.9, button.dataset.hand);
  };
  button.addEventListener("mousedown", press);
  button.addEventListener("touchstart", press, { passive: false });
  button.addEventListener("mouseup", release);
  button.addEventListener("mouseleave", release);
  button.addEventListener("touchend", release);
  button.addEventListener("touchcancel", release);
});

el.modeBtns.forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
el.startBtn.addEventListener("click", startGame);

if (el.bpmSlider) {
  el.bpmSlider.addEventListener("input", () => {
    state.bpm = parseInt(el.bpmSlider.value, 10);
    if (el.bpmVal) el.bpmVal.textContent = state.bpm;
  });
}
if (el.musicToggle) {
  el.musicToggle.addEventListener("click", () => {
    state.musicOn = !state.musicOn;
    if (el.musicState) el.musicState.textContent = state.musicOn ? "开" : "关";
    ensureAudio();
    if (state.musicOn) playTone("G", 500); // 试听
  });
}

const KEY_MAP = {
  a: ["left", "fist"], s: ["left", "open-palm"], d: ["left", "pinch"], f: ["left", "rest"],
  j: ["right", "fist"], k: ["right", "open-palm"], l: ["right", "pinch"], ";": ["right", "rest"]
};

window.addEventListener("keydown", (event) => {
  if (event.key === " " || event.code === "Space") {
    event.preventDefault();
    startGame();
    return;
  }
  if (event.repeat) return;
  const m = KEY_MAP[event.key.toLowerCase()];
  if (!m) return;
  event.preventDefault();
  const [hand, gesture] = m;
  if (!state.running && !state.ended && gesture !== "rest") startGame();
  dispatchGesture(gesture, 0.84 + Math.random() * 0.14, hand);
});

window.addEventListener("keyup", (event) => {
  const m = KEY_MAP[event.key.toLowerCase()];
  if (!m) return;
  const [hand, gesture] = m;
  if (gesture !== "rest") dispatchGesture("rest", 0.9, hand);
});

window.addEventListener("resize", resizeCanvas);

/* ===== 暴露接入点 ===== */
window.dispatchGesture = dispatchGesture;
window.connectRealtimeGestureApi = connectRealtimeGestureApi;
window.setBPM = (v) => {
  state.bpm = v;
  if (el.bpmSlider) el.bpmSlider.value = v;
  if (el.bpmVal) el.bpmVal.textContent = v;
};
window.playTone = playTone;
window.state = state;
window.debugChart = () => {
  const summary = state.chart.map(n => `${n.hand[0].toUpperCase()}${n.type==='hold'?'H':'T'}:${n.gesture}:${n.judged?'j':'-'}`);
  console.log("谱面颜色分布:", summary.join(" "));
  const vis = state.chart.filter(n => !n.judged || (n.type==='hold' && n.holding && !n.tailJudged));
  console.log("当前可见块:", vis.map(n => `${n.hand[0].toUpperCase()} ${n.gesture} judged=${n.judged}`).join(" | "));
  console.log("实时手势:", JSON.stringify({left: state.hands.left, right: state.hands.right}));
};

resizeCanvas();
refreshLiveRecog();
loadRuntimeMappingConfig().finally(() => connectRealtimeGestureApi());
requestAnimationFrame(loop);
