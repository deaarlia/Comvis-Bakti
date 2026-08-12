/* ==========================================================================
   NeoPuzzle — app.js
   Gesture-controlled 3×3 puzzle photobooth
   ========================================================================== */

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* ── Landmark indices ────────────────────────────────────────────────── */
const LM = {
  WRIST: 0, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_TIP: 8,
  MIDDLE_TIP: 12, MIDDLE_MCP: 9,
  RING_TIP: 16, RING_MCP: 13,
  PINKY_TIP: 20, PINKY_MCP: 17,
};

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

/* ── Tuning constants ────────────────────────────────────────────────── */
const PINCH_THRESHOLD     = 0.055;
const FRAME_PADDING       = 28;
const FREEZE_HOLD_MS      = 250;
const COUNTDOWN_SECONDS   = 3;
const FIST_HOLD_FRAMES    = 12;
const SNAP_DISTANCE_RATIO = 0.80;
const GRID                = 3;
const LOAD_TIMEOUT_MS     = 20000;
const FRAME_GRACE_MS      = 450;
const STRIP_MAX_PHOTOS    = 3;
const MIN_FRAME_RATIO     = 0.30; // Minimum frame threshold (30% of canvas dimensions)

/* ── Shatter effect ──────────────────────────────────────────────────── */
const SHATTER_COLS        = 6;
const SHATTER_ROWS        = 6;
const SHATTER_DURATION_MS = 850;
const DISPLACE_ANIM_MS    = 220;

/* ── DOM Elements ────────────────────────────────────────────────────── */
const videoEl         = document.getElementById("webcam");
const canvas          = document.getElementById("sceneCanvas");
const ctx             = canvas.getContext("2d");

const statusDot       = document.getElementById("statusDot");
const statusText      = document.getElementById("statusText");
const loadingOverlay  = document.getElementById("loadingOverlay");
const loaderText      = document.getElementById("loaderText");
const loaderRetry     = document.getElementById("loaderRetry");
const errorBanner     = document.getElementById("errorBanner");
const progressBadge   = document.getElementById("progressBadge");
const progressText    = document.getElementById("progressText");
const filterSelect    = document.getElementById("filterSelect");
const instructionToast = document.getElementById("instructionToast");
const instructionText = document.getElementById("instructionText");

const settingsBtn     = document.getElementById("settingsBtn");
const settingsPanel   = document.getElementById("settingsPanel");
const timerBadge      = document.getElementById("timerBadge");
const timerTextEl     = document.getElementById("timerText");
const solvedPopup     = document.getElementById("solvedPopup");
const timeupPopup     = document.getElementById("timeupPopup");

const galleryStrip    = document.getElementById("galleryStrip");
const galleryEmpty    = document.getElementById("galleryEmpty");
const galleryCount    = document.getElementById("galleryCount");
const downloadStripBtn = document.getElementById("downloadStripBtn");
const resetAllBtn     = document.getElementById("resetAllBtn");
const stripCompleteMsg = document.getElementById("stripCompleteMsg");

/* ── Application State ───────────────────────────────────────────────── */
let appState = "tracking"; // tracking | countdown | puzzle | shattering
let handLandmarker = null;
let activeFilter = "color";
let handLossGraceCounter = 0;
const MAX_HAND_LOSS_GRACE_FRAMES = 3;

const puzzle = {
  boardBox: null,
  pieces: [],
  solved: false,
  tileW: 0,
  tileH: 0,
  fullCapturedCanvas: null,
};

const shatter = {
  active: false,
  startedAt: 0,
  fragments: [],
  pendingCanvas: null,
};

const freezeGate = { holding: false, since: 0, box: null };
const lastSeenFrame = { box: null, at: 0 };
const countdown = { active: false, startedAt: 0 };
const drag = { activeHand: null, piece: null, offsetX: 0, offsetY: 0 };

const galleryEntries = [];
let fistHoldCounter = 0;

/* ── Timer state ─────────────────────────────────────────────────────── */
let timerLimitSec = 0;          // 0 = no limit
let puzzleStartedAt = 0;
let timerExpired = false;
const TIMEUP_RESET_DELAY_MS = 3000;

/* ═══════════════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════════════ */

async function init() {
  try {
    loaderText.textContent = "Mengunduh MediaPipe HandLandmarker…";

    const vision = await withTimeout(
      FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      ),
      LOAD_TIMEOUT_MS,
      "Timeout saat memuat runtime MediaPipe (WASM). Periksa koneksi internet."
    );

    loaderText.textContent = "Mengunduh model hand tracking (~10 MB)…";

    handLandmarker = await initHandLandmarker(vision);

    loaderText.textContent = "Mengakses kamera…";
    await initWebcam();

    loadingOverlay.classList.add("hidden");
    statusDot.className = "status-dot live";
    statusText.textContent = "kamera aktif";
    showInstruction("Tunjukkan kedua tangan, lalu cubit untuk membuat bingkai");
    updateStripDownload();

    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error("Init error:", err);
    loaderText.textContent = `Gagal: ${err.message}`;
    loaderText.style.color = "#e0533d";
    loaderRetry.classList.remove("hidden");
  }
}

let detectionConfidenceThreshold = 0.65;

function updateConfidenceThreshold(newVal) {
  detectionConfidenceThreshold = parseFloat(newVal) || 0.65;
  if (handLandmarker && handLandmarker.setOptions) {
    try {
      handLandmarker.setOptions({
        minHandDetectionConfidence: detectionConfidenceThreshold,
        minHandPresenceConfidence: detectionConfidenceThreshold,
        minTrackingConfidence: Math.max(0.50, detectionConfidenceThreshold - 0.05),
      });
    } catch (err) {
      console.warn("[NeoPuzzle] setOptions error:", err);
    }
  }
}

async function initHandLandmarker(vision) {
  // Try GPU first
  try {
    return await withTimeout(
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: detectionConfidenceThreshold,
        minHandPresenceConfidence: detectionConfidenceThreshold,
        minTrackingConfidence: Math.max(0.50, detectionConfidenceThreshold - 0.05),
      }),
      LOAD_TIMEOUT_MS,
      "Timeout saat mengunduh model HandLandmarker dengan GPU."
    );
  } catch (gpuErr) {
    console.warn("[PuzzleCam] GPU gagal, mencoba CPU…", gpuErr);
  }

  // Fallback CPU
  return await withTimeout(
    HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: detectionConfidenceThreshold,
      minHandPresenceConfidence: detectionConfidenceThreshold,
      minTrackingConfidence: Math.max(0.50, detectionConfidenceThreshold - 0.05),
    }),
    LOAD_TIMEOUT_MS,
    "Timeout saat mengunduh model HandLandmarker dengan CPU."
  );
}

let selectedDeviceId = null;

async function initWebcam(deviceId = null) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Browser tidak mendukung getUserMedia.");
  }

  if (videoEl.srcObject) {
    videoEl.srcObject.getTracks().forEach(track => track.stop());
  }

  const videoConstraints = deviceId
    ? { deviceId: { exact: deviceId } }
    : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" };

  const stream = await navigator.mediaDevices.getUserMedia({
    video: videoConstraints,
    audio: false,
  });
  videoEl.srcObject = stream;

  await new Promise((resolve) => {
    videoEl.onloadedmetadata = () => { videoEl.play(); resolve(); };
  });

  canvas.width = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  fitCanvasToWindow();

  await populateCameraList();
}

async function populateCameraList() {
  const cameraSelect = document.getElementById("cameraSelect");
  if (!cameraSelect || !navigator.mediaDevices?.enumerateDevices) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");

    const currentTrack = videoEl.srcObject?.getVideoTracks()[0];
    const activeDeviceId = currentTrack?.getSettings()?.deviceId;

    cameraSelect.innerHTML = "";

    if (videoDevices.length === 0) {
      cameraSelect.innerHTML = '<option value="">Kamera tidak ditemukan</option>';
      return;
    }

    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `Kamera ${index + 1}`;
      if (device.deviceId === activeDeviceId || (selectedDeviceId && device.deviceId === selectedDeviceId)) {
        option.selected = true;
      }
      cameraSelect.appendChild(option);
    });
  } catch (err) {
    console.warn("[NeoPuzzle] Gagal memuat daftar kamera:", err);
  }
}

function fitCanvasToWindow() {
  const stageEl = document.getElementById("stage");
  const wrapperEl = document.getElementById("canvasWrapper");
  if (!stageEl || !wrapperEl || !canvas) return;

  const videoAspect = (canvas.width && canvas.height) ? (canvas.width / canvas.height) : (16 / 9);
  wrapperEl.style.setProperty("--canvas-aspect", `${videoAspect}`);

  // Gunakan getBoundingClientRect untuk presisi piksel tinggi/lebar
  const rect = stageEl.getBoundingClientRect();
  const maxW = Math.max(100, rect.width - 32);
  const maxH = Math.max(100, rect.height - 32);
  const containerAspect = maxW / maxH;

  let cssW, cssH;
  if (containerAspect > videoAspect) {
    cssH = maxH;
    cssW = maxH * videoAspect;
  } else {
    cssW = maxW;
    cssH = maxW / videoAspect;
  }

  const roundedW = `${Math.round(cssW)}px`;
  const roundedH = `${Math.round(cssH)}px`;

  canvas.style.width = roundedW;
  canvas.style.height = roundedH;
  wrapperEl.style.width = roundedW;
  wrapperEl.style.height = roundedH;
}

let resizeTimer;
window.addEventListener("resize", () => {
  fitCanvasToWindow();
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(fitCanvasToWindow, 100);
});

window.addEventListener("orientationchange", () => {
  setTimeout(fitCanvasToWindow, 150);
});

function withTimeout(promise, ms, msg) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(msg)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/* ═══════════════════════════════════════════════════════════════════════
   GESTURE UTILS
   ═══════════════════════════════════════════════════════════════════════ */

function dist2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function isPinching(landmarks) {
  return dist2D(landmarks[LM.THUMB_TIP], landmarks[LM.INDEX_TIP]) < PINCH_THRESHOLD;
}

function isFist(landmarks) {
  const wrist = landmarks[LM.WRIST];
  const pairs = [
    [LM.INDEX_TIP, LM.INDEX_MCP],
    [LM.MIDDLE_TIP, LM.MIDDLE_MCP],
    [LM.RING_TIP, LM.RING_MCP],
    [LM.PINKY_TIP, LM.PINKY_MCP],
  ];
  let curled = 0;
  for (const [tipIdx, mcpIdx] of pairs) {
    if (dist2D(landmarks[tipIdx], wrist) < dist2D(landmarks[mcpIdx], wrist)) curled++;
  }
  return curled >= 4;
}

function toPixel(lmNorm) {
  return { x: lmNorm.x * canvas.width, y: lmNorm.y * canvas.height };
}

function mirrorLandmarkX(lm) {
  return { x: 1 - lm.x, y: lm.y };
}

function computeHandFrame(indexTipA, indexTipB) {
  const a = toPixel(indexTipA);
  const b = toPixel(indexTipB);

  const minX = Math.min(a.x, b.x) - FRAME_PADDING;
  const maxX = Math.max(a.x, b.x) + FRAME_PADDING;
  const minY = Math.min(a.y, b.y) - FRAME_PADDING;
  const maxY = Math.max(a.y, b.y) + FRAME_PADDING;

  const x = Math.max(0, minX);
  const y = Math.max(0, minY);
  const width = Math.min(canvas.width, maxX) - x;
  const height = Math.min(canvas.height, maxY) - y;

  return { x, y, width, height };
}

/* ═══════════════════════════════════════════════════════════════════════
   FILTER
   ═══════════════════════════════════════════════════════════════════════ */

function applyFilter(imageData, filterType) {
  if (filterType === "color") return imageData;
  const d = imageData.data;
  const len = d.length;

  for (let i = 0; i < len; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2];

    if (filterType === "vintage_warm") {
      // Warm Film (Portra Creamy Warmth)
      const nr = r * 1.15 + 10;
      const ng = g * 1.05 + 6;
      const nb = b * 0.88 + 4;
      d[i]   = Math.min(255, Math.max(0, nr));
      d[i+1] = Math.min(255, Math.max(0, ng));
      d[i+2] = Math.min(255, Math.max(0, nb));
    }
    else if (filterType === "mono_chrome") {
      // Noir Mono (Cinematic High Contrast B&W)
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      const highContrast = (gray - 128) * 1.4 + 128;
      const clamped = Math.min(255, Math.max(0, highContrast));
      d[i] = d[i+1] = d[i+2] = clamped;
    }
    else if (filterType === "vintage_mono") {
      // Vintage Noir (Cool Espresso Brown / Neutral Taupe Sepia)
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      const contrastGray = (gray - 128) * 1.28 + 128;
      const cG = Math.min(255, Math.max(0, contrastGray));
      const nr = cG * 1.12 + 14;  // Red base for subtle brown tone
      const ng = cG * 0.98 + 6;   // Subdued green (removes yellow cast)
      const nb = cG * 0.92 + 8;   // Cool blue balance (prevents golden/yellow warmth)
      d[i]   = Math.min(255, Math.max(0, nr));
      d[i+1] = Math.min(255, Math.max(0, ng));
      d[i+2] = Math.min(255, Math.max(0, nb));
    }
  }
  return imageData;
}



/* ═══════════════════════════════════════════════════════════════════════
   COUNTDOWN & CAPTURE
   ═══════════════════════════════════════════════════════════════════════ */

function startCountdown(frameBox) {
  puzzle.boardBox = { ...frameBox };
  appState = "countdown";
  countdown.active = true;
  countdown.startedAt = performance.now();
  hideInstruction();
}

function drawCountdownOverlay(box) {
  const elapsed = (performance.now() - countdown.startedAt) / 1000;
  const remaining = COUNTDOWN_SECONDS - elapsed;

  if (remaining <= 0) {
    finishCountdownAndCapture(box);
    return;
  }

  ctx.save();
  // Neo Blue frame
  ctx.strokeStyle = "#3b5bdb";
  ctx.lineWidth = 3;
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  // Darken inside
  ctx.fillStyle = "rgba(10, 14, 26, 0.55)";
  ctx.fillRect(box.x, box.y, box.width, box.height);

  // Countdown number
  const n = Math.ceil(remaining);
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const fontSize = Math.max(48, Math.min(box.width, box.height) * 0.4);
  ctx.font = `800 ${fontSize}px 'Plus Jakarta Sans', sans-serif`;
  ctx.fillStyle = "#3b5bdb";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(n), cx, cy);
  ctx.restore();

  statusText.innerHTML = `<i class="ph ph-camera"></i> Mengambil foto dalam ${n}…`;
}

function finishCountdownAndCapture(box) {
  countdown.active = false;

  // Create mirrored full frame from video
  const mirroredFrame = document.createElement("canvas");
  mirroredFrame.width = canvas.width;
  mirroredFrame.height = canvas.height;
  const mirroredCtx = mirroredFrame.getContext("2d");
  mirroredCtx.save();
  mirroredCtx.translate(mirroredFrame.width, 0);
  mirroredCtx.scale(-1, 1);
  mirroredCtx.drawImage(videoEl, 0, 0, mirroredFrame.width, mirroredFrame.height);
  mirroredCtx.restore();

  // Crop the framed area
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = Math.max(1, Math.round(box.width));
  cropCanvas.height = Math.max(1, Math.round(box.height));
  const cropCtx = cropCanvas.getContext("2d");
  cropCtx.drawImage(
    mirroredFrame,
    box.x, box.y, box.width, box.height,
    0, 0, cropCanvas.width, cropCanvas.height
  );

  // Apply selected filter
  if (activeFilter !== "color") {
    const imgData = cropCtx.getImageData(0, 0, cropCanvas.width, cropCanvas.height);
    applyFilter(imgData, activeFilter);
    cropCtx.putImageData(imgData, 0, 0);
  }

  puzzle.fullCapturedCanvas = cropCanvas;

  // Create puzzle pieces
  createPuzzlePieces(cropCanvas, box);
}

/* ═══════════════════════════════════════════════════════════════════════
   PUZZLE CREATION
   ═══════════════════════════════════════════════════════════════════════ */

function createPuzzlePieces(cropCanvas, box) {
  const tileW = Math.floor(cropCanvas.width / GRID);
  const tileH = Math.floor(cropCanvas.height / GRID);
  const pieces = [];

  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      const sx = col * tileW;
      const sy = row * tileH;
      const w = col === GRID - 1 ? cropCanvas.width - sx : tileW;
      const h = row === GRID - 1 ? cropCanvas.height - sy : tileH;

      const pieceCanvas = document.createElement("canvas");
      pieceCanvas.width = w;
      pieceCanvas.height = h;
      pieceCanvas.getContext("2d").drawImage(cropCanvas, sx, sy, w, h, 0, 0, w, h);

      pieces.push({
        row, col,
        canvas: pieceCanvas,
        w, h,
        x: 0, y: 0,
        placed: false,
        dragging: false,
        displacing: false,
      });
    }
  }

  // Create shuffled slot positions within the board
  const slots = [];
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      slots.push({ x: box.x + col * tileW, y: box.y + row * tileH });
    }
  }
  shuffle(slots);

  pieces.forEach((piece, i) => {
    piece.x = slots[i].x;
    piece.y = slots[i].y;
    // If randomly placed in own cell, check and snap
    if (isNearOwnCell(piece, box, tileW, tileH)) {
      snapPieceToCell(piece, box, tileW, tileH);
    }
  });

  puzzle.boardBox = box;
  puzzle.pieces = pieces;
  puzzle.tileW = tileW;
  puzzle.tileH = tileH;
  puzzle.solved = pieces.every(p => p.placed);
  appState = "puzzle";
  fistHoldCounter = 0;
  startPuzzleTimer();
  updateProgressBadge();

  showInstruction(" Cubit dengan satu tangan untuk menggeser kepingan");
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/* ═══════════════════════════════════════════════════════════════════════
   PUZZLE DRAG & SNAP
   ═══════════════════════════════════════════════════════════════════════ */

function isNearOwnCell(piece, box, tileW, tileH) {
  const correctX = box.x + piece.col * tileW;
  const correctY = box.y + piece.row * tileH;
  const dx = piece.x - correctX;
  const dy = piece.y - correctY;
  const tolerance = Math.min(tileW, tileH) * SNAP_DISTANCE_RATIO;
  return Math.sqrt(dx * dx + dy * dy) < tolerance;
}

function reconcilePlacedState(box, tileW, tileH) {
  if (!box || !puzzle.pieces.length) return false;
  for (const piece of puzzle.pieces) {
    if (piece.placed) continue;           // already locked — don't touch
    if (piece.displacing || piece.dragging) continue;
    piece.placed = isNearOwnCell(piece, box, tileW, tileH);
  }
  return puzzle.pieces.every(p => p.placed);
}

function snapPieceToCell(piece, box, tileW, tileH) {
  displaceCellOccupant(piece, piece.row, piece.col, box, tileW, tileH);
  piece.x = box.x + piece.col * tileW;
  piece.y = box.y + piece.row * tileH;
  piece.placed = true;
}

function displaceCellOccupant(piece, targetRow, targetCol, box, tileW, tileH) {
  const cellX = box.x + targetCol * tileW;
  const cellY = box.y + targetRow * tileH;

  const occupant = puzzle.pieces.find((p) => {
    if (p === piece || p.displacing || p.placed) return false;  // placed pieces are locked
    const cx = p.x + p.w / 2;
    const cy = p.y + p.h / 2;
    return cx >= cellX && cx < cellX + tileW && cy >= cellY && cy < cellY + tileH;
  });
  if (!occupant) return;

  // Don't displace if occupant is correctly placed in its own cell
  if (occupant.row === targetRow && occupant.col === targetCol && occupant.placed) return;

  occupant.placed = false;

  // Find a free cell
  const freeCells = [];
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      if (row === targetRow && col === targetCol) continue;
      const cx0 = box.x + col * tileW;
      const cy0 = box.y + row * tileH;
      const taken = puzzle.pieces.some((p) => {
        if (p === occupant || p === piece || p.displacing) return false;
        const pcx = p.x + p.w / 2;
        const pcy = p.y + p.h / 2;
        return pcx >= cx0 && pcx < cx0 + tileW && pcy >= cy0 && pcy < cy0 + tileH;
      });
      if (!taken) freeCells.push({ row, col });
    }
  }

  let targetSlot;
  if (freeCells.length > 0) {
    targetSlot = freeCells[Math.floor(Math.random() * freeCells.length)];
  } else {
    targetSlot = { row: occupant.row, col: occupant.col };
  }

  const targetX = box.x + targetSlot.col * tileW;
  const targetY = box.y + targetSlot.row * tileH;

  animateDisplacement(occupant, targetX, targetY);
}

function animateDisplacement(piece, targetX, targetY) {
  const startX = piece.x;
  const startY = piece.y;
  const startedAt = performance.now();
  piece.displacing = true;

  function step() {
    const t = Math.min(1, (performance.now() - startedAt) / DISPLACE_ANIM_MS);
    const eased = 1 - Math.pow(1 - t, 3);
    piece.x = startX + (targetX - startX) * eased;
    piece.y = startY + (targetY - startY) * eased;

    if (t < 1) {
      requestAnimationFrame(step);
    } else {
      piece.x = targetX;
      piece.y = targetY;
      piece.displacing = false;
      clampPieceToBoard(piece);
    }
  }
  requestAnimationFrame(step);
}

function clampPieceToBoard(piece) {
  const box = puzzle.boardBox;
  piece.x = Math.min(Math.max(piece.x, box.x), box.x + box.width - piece.w);
  piece.y = Math.min(Math.max(piece.y, box.y), box.y + box.height - piece.h);
}

function findNearestPiece(px, py) {
  let best = null;
  let bestDist = Infinity;
  for (const piece of puzzle.pieces) {
    if (piece.displacing || piece.placed) continue;  // placed pieces are locked
    const cx = piece.x + piece.w / 2;
    const cy = piece.y + piece.h / 2;
    const d = Math.hypot(px - cx, py - cy);
    if (d < Math.max(piece.w, piece.h) * 0.75 && d < bestDist) {
      best = piece;
      bestDist = d;
    }
  }
  return best;
}

const SMOOTHING_ALPHA = 0.38; // Exponential Moving Average (EMA) low-pass filter factor for zero-jitter drag
const smoothedLandmarksMap = new Map();

function getSmoothedLandmarks(handLabel, rawLandmarksPx, alpha = SMOOTHING_ALPHA) {
  let prev = smoothedLandmarksMap.get(handLabel);
  if (!prev || prev.length !== rawLandmarksPx.length) {
    const initial = rawLandmarksPx.map(pt => ({ x: pt.x, y: pt.y }));
    smoothedLandmarksMap.set(handLabel, initial);
    return initial;
  }
  const smoothed = rawLandmarksPx.map((pt, i) => ({
    x: prev[i].x + (pt.x - prev[i].x) * alpha,
    y: prev[i].y + (pt.y - prev[i].y) * alpha,
  }));
  smoothedLandmarksMap.set(handLabel, smoothed);
  return smoothed;
}

function handleDragForHand(handLabel, pinching, indexPx) {
  if (timerExpired) return;  // no interaction after time's up
  if (pinching) {
    if (drag.activeHand === null) {
      const candidate = findNearestPiece(indexPx.x, indexPx.y);
      if (candidate) {
        drag.activeHand = handLabel;
        drag.piece = candidate;
        drag.offsetX = indexPx.x - candidate.x;
        drag.offsetY = indexPx.y - candidate.y;
        candidate.dragging = true;
        candidate.placed = false;
      }
    } else if (drag.activeHand === handLabel && drag.piece) {
      const targetX = indexPx.x - drag.offsetX;
      const targetY = indexPx.y - drag.offsetY;

      // Exponential Moving Average (EMA) Low-Pass Filter eliminates finger jitter / shakiness
      drag.piece.x += (targetX - drag.piece.x) * SMOOTHING_ALPHA;
      drag.piece.y += (targetY - drag.piece.y) * SMOOTHING_ALPHA;
    }
  } else {
    if (drag.activeHand === handLabel && drag.piece) {
      const piece = drag.piece;
      piece.dragging = false;
      if (isNearOwnCell(piece, puzzle.boardBox, puzzle.tileW, puzzle.tileH)) {
        snapPieceToCell(piece, puzzle.boardBox, puzzle.tileW, puzzle.tileH);
      } else {
        const box = puzzle.boardBox;
        const cx = piece.x + piece.w / 2;
        const cy = piece.y + piece.h / 2;
        const dropCol = Math.min(GRID - 1, Math.max(0, Math.floor((cx - box.x) / puzzle.tileW)));
        const dropRow = Math.min(GRID - 1, Math.max(0, Math.floor((cy - box.y) / puzzle.tileH)));

        // Snap edge-to-edge directly into dropped cell slot
        piece.x = box.x + dropCol * puzzle.tileW;
        piece.y = box.y + dropRow * puzzle.tileH;

        displaceCellOccupant(piece, dropRow, dropCol, box, puzzle.tileW, puzzle.tileH);
      }
      drag.activeHand = null;
      drag.piece = null;
      puzzle.solved = reconcilePlacedState(puzzle.boardBox, puzzle.tileW, puzzle.tileH);
      updateProgressBadge();
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   RENDERING
   ═══════════════════════════════════════════════════════════════════════ */

function getGpuCanvasFilter(filterType) {
  if (filterType === "vintage_warm") {
    return "sepia(40%) contrast(110%) brightness(105%) saturate(115%)";
  } else if (filterType === "mono_chrome") {
    return "grayscale(100%) contrast(135%) brightness(95%)";
  } else if (filterType === "vintage_mono") {
    return "grayscale(100%) sepia(30%) contrast(125%) brightness(100%)";
  }
  return "none";
}

function drawVideoFrame() {
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  if (activeFilter !== "color") {
    ctx.filter = getGpuCanvasFilter(activeFilter);
  }
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  ctx.filter = "none";
  ctx.restore();
}

function drawLiveFrameOverlay(box, isTooSmall = false) {
  ctx.save();
  const mainColor = isTooSmall ? "#ef4444" : "#3b5bdb";
  ctx.strokeStyle = mainColor;
  ctx.lineWidth = 3;
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  // Corner handles
  const cornerLen = 18;
  ctx.lineWidth = 4;
  const corners = [
    [box.x, box.y, 1, 1],
    [box.x + box.width, box.y, -1, 1],
    [box.x, box.y + box.height, 1, -1],
    [box.x + box.width, box.y + box.height, -1, -1],
  ];
  for (const [cx, cy, dx, dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(cx, cy + cornerLen * dy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + cornerLen * dx, cy);
    ctx.stroke();
  }

  // Dimension & warning label
  const labelText = isTooSmall
    ? "Bingkai terlalu kecil. Perluas area pengambilan gambar."
    : `${Math.round(box.width)}×${Math.round(box.height)}  AR ${(box.width / box.height).toFixed(2)}`;
  ctx.font = "700 12px 'Plus Jakarta Sans', sans-serif";
  ctx.fillStyle = mainColor;
  ctx.textAlign = "left";
  ctx.fillText(labelText, box.x + 4, Math.max(14, box.y - 8));

  ctx.restore();
}

function drawBoardAndPieces() {
  const box = puzzle.boardBox;

  // Board background
  ctx.save();
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(box.x, box.y, box.width, box.height);
  ctx.restore();

  // Grid lines
  ctx.save();
  ctx.strokeStyle = "rgba(59, 91, 219, 0.22)";
  ctx.lineWidth = 1;
  for (let i = 1; i < GRID; i++) {
    ctx.beginPath();
    ctx.moveTo(box.x + i * puzzle.tileW, box.y);
    ctx.lineTo(box.x + i * puzzle.tileW, box.y + box.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(box.x, box.y + i * puzzle.tileH);
    ctx.lineTo(box.x + box.width, box.y + i * puzzle.tileH);
    ctx.stroke();
  }
  ctx.restore();

  // Pieces (Layer 0: placed/locked -> Layer 1: unplaced/freed -> Layer 2: dragging)
  const sorted = [...puzzle.pieces].sort((a, b) => {
    const layerA = a.dragging ? 2 : (a.placed ? 0 : 1);
    const layerB = b.dragging ? 2 : (b.placed ? 0 : 1);
    return layerA - layerB;
  });
  for (const piece of sorted) {
    ctx.save();
    ctx.drawImage(piece.canvas, piece.x, piece.y, piece.w, piece.h);
    ctx.strokeStyle = piece.placed ? "#10b981" : "rgba(148, 163, 184, 0.5)";
    ctx.lineWidth = piece.dragging ? 3 : 1.5;
    ctx.strokeRect(piece.x, piece.y, piece.w, piece.h);

    // Locked indicator on placed pieces
    if (piece.placed) {
      const badgeR = Math.max(8, Math.min(piece.w, piece.h) * 0.09);
      const bx = piece.x + piece.w - badgeR - 4;
      const by = piece.y + badgeR + 4;
      ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
      ctx.beginPath();
      ctx.arc(bx, by, badgeR, 0, Math.PI * 2);
      ctx.fill();
      // Checkmark
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(bx - badgeR * 0.35, by);
      ctx.lineTo(bx - badgeR * 0.05, by + badgeR * 0.3);
      ctx.lineTo(bx + badgeR * 0.4, by - badgeR * 0.3);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Board border
  ctx.save();
  ctx.strokeStyle = puzzle.solved ? "#10b981" : "#3b5bdb";
  ctx.lineWidth = 3;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
  ctx.restore();

  // Solved overlay — use HTML popup instead of canvas text
  if (puzzle.solved) {
    ctx.save();
    ctx.fillStyle = "rgba(95, 174, 110, 0.15)";
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.restore();
    showSolvedPopup();
  } else {
    hideSolvedPopup();
  }
}

function drawHandSkeleton(landmarksPx) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 3;

  for (const [iA, iB] of HAND_CONNECTIONS) {
    const a = landmarksPx[iA];
    const b = landmarksPx[iB];
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  ctx.fillStyle = "white";
  for (const p of landmarksPx) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawHandSkeletonsOverBoard(handsLandmarks, box) {
  if (!box || !handsLandmarks || handsLandmarks.length === 0) return;
  handsLandmarks.forEach((lm, i) => {
    const label = i === 0 ? "A" : "B";
    const rawPx = lm.map(pt => toPixel(mirrorLandmarkX(pt)));
    const smoothedPx = getSmoothedLandmarks(label, rawPx, 0.42);
    const overBoard = smoothedPx.some(p =>
      p.x >= box.x && p.x <= box.x + box.width &&
      p.y >= box.y && p.y <= box.y + box.height
    );
    if (overBoard) drawHandSkeleton(smoothedPx);
  });
}

/* ═══════════════════════════════════════════════════════════════════════
   SHATTER EFFECT
   ═══════════════════════════════════════════════════════════════════════ */

function startShatter(sourceCanvas, box) {
  const fragW = sourceCanvas.width / SHATTER_COLS;
  const fragH = sourceCanvas.height / SHATTER_ROWS;
  const fragments = [];

  for (let row = 0; row < SHATTER_ROWS; row++) {
    for (let col = 0; col < SHATTER_COLS; col++) {
      const sx = col * fragW;
      const sy = row * fragH;

      const fragCanvas = document.createElement("canvas");
      fragCanvas.width = Math.ceil(fragW);
      fragCanvas.height = Math.ceil(fragH);
      fragCanvas.getContext("2d").drawImage(sourceCanvas, sx, sy, fragW, fragH, 0, 0, fragCanvas.width, fragCanvas.height);

      const cx = box.x + sx + fragW / 2;
      const cy = box.y + sy + fragH / 2;
      const boardCx = box.x + box.width / 2;
      const boardCy = box.y + box.height / 2;
      const dirX = cx - boardCx;
      const dirY = cy - boardCy;
      const dirLen = Math.max(1, Math.hypot(dirX, dirY));
      const speed = 90 + Math.random() * 160;

      fragments.push({
        canvas: fragCanvas, x: cx, y: cy, w: fragW, h: fragH,
        vx: (dirX / dirLen) * speed + (Math.random() - 0.5) * 40,
        vy: (dirY / dirLen) * speed + (Math.random() - 0.5) * 40 - 60,
        rotation: 0,
        rotationSpeed: (Math.random() - 0.5) * 6,
        gravity: 220 + Math.random() * 80,
      });
    }
  }

  shatter.fragments = fragments;
  shatter.active = true;
  shatter.startedAt = performance.now();
  appState = "shattering";
}

function updateAndDrawShatter() {
  const elapsedMs = performance.now() - shatter.startedAt;
  const t = Math.min(1, elapsedMs / SHATTER_DURATION_MS);

  if (t >= 1) {
    finishShatter();
    return;
  }

  const dt = 1 / 60;
  const fadeStart = 0.45;

  ctx.save();
  for (const frag of shatter.fragments) {
    frag.x += frag.vx * dt;
    frag.y += frag.vy * dt;
    frag.vy += frag.gravity * dt;
    frag.rotation += frag.rotationSpeed * dt;

    const alpha = t < fadeStart ? 1 : Math.max(0, 1 - (t - fadeStart) / (1 - fadeStart));
    const scale = 1 - t * 0.25;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(frag.x, frag.y);
    ctx.rotate(frag.rotation);
    ctx.scale(scale, scale);
    ctx.drawImage(frag.canvas, -frag.w / 2, -frag.h / 2, frag.w, frag.h);
    ctx.restore();
  }
  ctx.restore();
}

function finishShatter() {
  shatter.active = false;
  shatter.fragments = [];
  if (shatter.pendingCanvas) {
    addToGallery(shatter.pendingCanvas);
    statusText.innerHTML = '<i class="ph ph-check-circle"></i> Tersimpan di photo strip!';
    shatter.pendingCanvas = null;
  }
  resetPuzzleOnly();
}

/* ═══════════════════════════════════════════════════════════════════════
   FIST RESET / SAVE
   ═══════════════════════════════════════════════════════════════════════ */

function handleFistReset() {
  if (appState !== "puzzle" || !puzzle.solved) return;

  const reallySolved = reconcilePlacedState(puzzle.boardBox, puzzle.tileW, puzzle.tileH);
  puzzle.solved = reallySolved;

  if (reallySolved && puzzle.fullCapturedCanvas) {
    shatter.pendingCanvas = puzzle.fullCapturedCanvas;
    startShatter(puzzle.fullCapturedCanvas, puzzle.boardBox);
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   GALLERY & 5:4 STRIP CROPPING
   ═══════════════════════════════════════════════════════════════════════ */

function cropCanvasToRatio(srcCanvas, aspectW = 5, aspectH = 4) {
  const targetAspect = aspectW / aspectH;
  const srcW = srcCanvas.width;
  const srcH = srcCanvas.height;
  const srcAspect = srcW / srcH;

  let cropW, cropH, cropX, cropY;

  if (srcAspect > targetAspect) {
    cropH = srcH;
    cropW = Math.round(srcH * targetAspect);
    cropX = Math.round((srcW - cropW) / 2);
    cropY = 0;
  } else {
    cropW = srcW;
    cropH = Math.round(srcW / targetAspect);
    cropX = 0;
    cropY = Math.round((srcH - cropH) / 2);
  }

  const outCanvas = document.createElement("canvas");
  outCanvas.width = cropW;
  outCanvas.height = cropH;
  const outCtx = outCanvas.getContext("2d");
  outCtx.drawImage(srcCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
  return outCanvas;
}

function addToGallery(snapshotCanvas) {
  if (galleryEntries.length >= STRIP_MAX_PHOTOS) return;

  // Save original uncropped snapshot canvas (exact pinched frame ratio)
  galleryEntries.push({ canvas: snapshotCanvas, time: Date.now() });
  renderGalleryThumb(snapshotCanvas, galleryEntries.length);
  galleryCount.textContent = `${galleryEntries.length} / ${STRIP_MAX_PHOTOS}`;
  if (galleryEmpty) galleryEmpty.style.display = "none";

  if (galleryEntries.length >= STRIP_MAX_PHOTOS) {
    showStripComplete();
  }
  updateStripDownload();
  updateSettingsAvailability();
}

function renderGalleryThumb(snapshotCanvas, index) {
  const print = document.createElement("div");
  print.className = "print";

  const thumbCanvas = document.createElement("canvas");
  const THUMB_W = 220;
  const scale = THUMB_W / snapshotCanvas.width;
  thumbCanvas.width = THUMB_W;
  thumbCanvas.height = Math.round(snapshotCanvas.height * scale);
  thumbCanvas.getContext("2d").drawImage(snapshotCanvas, 0, 0, thumbCanvas.width, thumbCanvas.height);

  const label = document.createElement("div");
  label.className = "print-label";
  label.textContent = `#${String(index).padStart(2, "0")} | NeoPuzzle`;

  print.appendChild(thumbCanvas);
  print.appendChild(label);
  galleryStrip.insertBefore(print, galleryStrip.firstChild);
}

function showStripComplete() {
  if (stripCompleteMsg) stripCompleteMsg.classList.add("visible");
}

function hideStripComplete() {
  if (stripCompleteMsg) stripCompleteMsg.classList.remove("visible");
}

function updateStripDownload() {
  const hasPhotos = galleryEntries.length > 0;
  const isPuzzleActive = appState === "puzzle" || puzzle.boardBox !== null;
  const canReset = hasPhotos || isPuzzleActive;

  if (downloadStripBtn) downloadStripBtn.disabled = !hasPhotos;
  if (resetAllBtn) resetAllBtn.disabled = !canReset;
}

function isStripFull() {
  return galleryEntries.length >= STRIP_MAX_PHOTOS;
}

function downloadPhotoStrip() {
  if (galleryEntries.length === 0) return;

  // Crop each entry to 3:2 aspect ratio for the downloaded strip
  const cropped32Entries = galleryEntries.map(e => cropCanvasToRatio(e.canvas, 3, 2));

  const targetW = cropped32Entries[0].width;
  const scaledHeights = cropped32Entries.map(c => Math.round(c.height * (targetW / c.width)));
  const border = 24, gap = 16;
  const totalH = border * 2 + scaledHeights.reduce((s, h) => s + h, 0) + gap * (cropped32Entries.length - 1);
  const totalW = targetW + border * 2;

  const stripCanvas = document.createElement("canvas");
  stripCanvas.width = totalW;
  stripCanvas.height = totalH;
  const sCtx = stripCanvas.getContext("2d");

  sCtx.fillStyle = document.documentElement.getAttribute("data-theme") === "light" ? "#f8fafc" : "#0a0c10";
  sCtx.fillRect(0, 0, totalW, totalH);

  let cursorY = border;
  cropped32Entries.forEach((cCanvas) => {
    const h = Math.round(cCanvas.height * (targetW / cCanvas.width));
    sCtx.drawImage(cCanvas, border, cursorY, targetW, h);
    cursorY += h + gap;
  });

  stripCanvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `NeoPuzzle_Strip_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }, "image/png");
}

/* ═══════════════════════════════════════════════════════════════════════
   RESET
   ═══════════════════════════════════════════════════════════════════════ */

function resetPuzzleOnly() {
  puzzle.boardBox = null;
  puzzle.pieces = [];
  puzzle.solved = false;
  puzzle.fullCapturedCanvas = null;
  appState = "tracking";
  countdown.active = false;
  drag.activeHand = null;
  drag.piece = null;
  shatter.active = false;
  shatter.fragments = [];
  shatter.pendingCanvas = null;
  fistHoldCounter = 0;
  freezeGate.holding = false;
  freezeGate.since = 0;
  freezeGate.box = null;
  lastSeenFrame.box = null;
  lastSeenFrame.at = 0;
  timerExpired = false;
  puzzleStartedAt = 0;
  updateProgressBadge();
  hideTimerBadge();
  hideSolvedPopup();
  hideTimeupPopup();
  showInstruction("Tunjukkan kedua tangan, lalu cubit untuk membuat bingkai");
}

function resetEverything() {
  galleryEntries.length = 0;
  galleryStrip.innerHTML = "";
  galleryCount.textContent = `0 / ${STRIP_MAX_PHOTOS}`;
  if (galleryEmpty) {
    galleryEmpty.style.display = "flex";
    galleryStrip.appendChild(galleryEmpty);
  }
  hideStripComplete();
  updateStripDownload();
  resetPuzzleOnly();
  statusText.innerHTML = '<i class="ph ph-trash"></i> Semua direset';
  updateSettingsAvailability();
}

/* ═══════════════════════════════════════════════════════════════════════
   UI HELPERS
   ═══════════════════════════════════════════════════════════════════════ */

function updateProgressBadge() {
  if (appState !== "puzzle") {
    progressBadge.classList.remove("visible", "solved");
    return;
  }
  const placedCount = puzzle.pieces.filter(p => p.placed).length;
  progressText.textContent = `${placedCount} / ${puzzle.pieces.length} terpasang`;
  progressBadge.classList.add("visible");
  progressBadge.classList.toggle("solved", puzzle.solved);
}

function showInstruction(text) {
  instructionText.textContent = text;
  instructionToast.classList.remove("hidden");
}

function hideInstruction() {
  instructionToast.classList.add("hidden");
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.style.display = "block";
}

function showSolvedPopup() {
  if (solvedPopup) solvedPopup.classList.add("visible");
}

function hideSolvedPopup() {
  if (solvedPopup) solvedPopup.classList.remove("visible");
}

function showTimeupPopup() {
  if (timeupPopup) timeupPopup.classList.add("visible");
}

function hideTimeupPopup() {
  if (timeupPopup) timeupPopup.classList.remove("visible");
}

/* ═══════════════════════════════════════════════════════════════════════
   TIMER
   ═══════════════════════════════════════════════════════════════════════ */

function startPuzzleTimer() {
  puzzleStartedAt = performance.now();
  timerExpired = false;
  if (timerLimitSec > 0) {
    showTimerBadge();
  } else {
    hideTimerBadge();
  }
}

function showTimerBadge() {
  timerBadge.classList.add("visible");
}

function hideTimerBadge() {
  timerBadge.classList.remove("visible", "warning", "danger");
}

function updateTimer() {
  if (timerLimitSec <= 0 || !puzzleStartedAt || timerExpired) return;

  const elapsed = (performance.now() - puzzleStartedAt) / 1000;
  const remaining = Math.max(0, timerLimitSec - elapsed);
  const minutes = Math.floor(remaining / 60);
  const seconds = Math.ceil(remaining % 60);

  timerTextEl.textContent = `${minutes}:${String(seconds).padStart(2, "0")}`;

  // Warning at 30s, danger at 10s
  timerBadge.classList.remove("warning", "danger");
  if (remaining <= 10) {
    timerBadge.classList.add("danger");
  } else if (remaining <= 30) {
    timerBadge.classList.add("warning");
  }

  if (remaining <= 0) {
    timerExpired = true;
    handleTimeUp();
  }
}

function handleTimeUp() {
  // Show time's up popup
  showTimeupPopup();
  hideInstruction();
  statusText.innerHTML = '<i class="ph ph-alarm"></i> Waktu habis!';

  // Auto-reset after delay
  setTimeout(() => {
    hideTimeupPopup();
    resetPuzzleOnly();
  }, TIMEUP_RESET_DELAY_MS);
}

/* ═══════════════════════════════════════════════════════════════════════
   MAIN RENDER LOOP
   ═══════════════════════════════════════════════════════════════════════ */

function updateFilterAvailability() {
  const isSessionActive = appState === "countdown" || appState === "puzzle" || appState === "shattering";
  if (filterSelect) filterSelect.disabled = isSessionActive;
}

let lastDetectionTime = 0;
const DETECTION_INTERVAL_MS = 50; // Max ~20 FPS AI inference throttling to free up Main Thread CPU cycles
let cachedResult = null;

function renderLoop() {
  if (videoEl.readyState >= 2 && handLandmarker) {
    drawVideoFrame();
    const nowMs = performance.now();
    
    // AI Neural Net inference throttled to ~20 FPS while canvas renders at smooth 60 FPS
    if (nowMs - lastDetectionTime >= DETECTION_INTERVAL_MS) {
      cachedResult = handLandmarker.detectForVideo(videoEl, nowMs);
      lastDetectionTime = nowMs;
    }
    
    if (cachedResult) {
      processResults(cachedResult);
    }
  }
  requestAnimationFrame(renderLoop);
}

function updateSettingsAvailability() {
  const isPuzzleActive = appState === "countdown" || appState === "puzzle" || appState === "shattering";
  const hasPhotos = galleryEntries.length > 0;
  const timerDisabled = isPuzzleActive || hasPhotos;

  // Timer limit radios lock only during active photo session
  document.querySelectorAll('input[name="timerLimit"]').forEach((radio) => {
    radio.disabled = timerDisabled;
    const label = radio.closest(".radio-option");
    if (label) {
      label.classList.toggle("disabled", timerDisabled);
    }
  });

  // Theme radios are ALWAYS enabled
  document.querySelectorAll('input[name="themeMode"]').forEach((radio) => {
    radio.disabled = false;
    const label = radio.closest(".radio-option");
    if (label) {
      label.classList.remove("disabled");
    }
  });

  const settingsNotice = document.getElementById("settingsNotice");
  if (settingsNotice) {
    settingsNotice.style.display = timerDisabled ? "flex" : "none";
  }
}

function isForegroundHand(landmarks) {
  if (!landmarks || landmarks.length === 0) return false;
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const pt of landmarks) {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }
  const width = maxX - minX;
  const height = maxY - minY;
  // Filter out background hands/noise smaller than 5% of screen
  return width >= 0.05 && height >= 0.05;
}

const cancelFrameBtn = document.getElementById("cancelFrameBtn");

function updateCancelFrameBtnVisibility() {
  if (!cancelFrameBtn) return;
  const canCancel = freezeGate.holding || appState === "countdown" || (appState === "puzzle" && !puzzle.solved);
  cancelFrameBtn.classList.toggle("hidden", !canCancel);
}

function processResults(result) {
  updateCancelFrameBtnVisibility();

  // Shattering state — just animate
  if (appState === "shattering") {
    updateAndDrawShatter();
    statusText.innerHTML = '<i class="ph ph-floppy-disk"></i> Menyimpan foto…';
    showInstruction('Menyimpan ke photo strip…');
    return;
  }

  // Countdown state — FRAME IS LOCKED. IGNORE GESTURES AND HAND LOSS.
  if (appState === "countdown") {
    drawCountdownOverlay(puzzle.boardBox);
    return;
  }

  const rawLandmarks = result.landmarks || [];
  const handsLandmarks = rawLandmarks.filter(isForegroundHand);
  const rawNoHands = handsLandmarks.length === 0;

  if (rawNoHands) {
    handLossGraceCounter++;
  } else {
    handLossGraceCounter = 0;
  }

  // Only consider hand lost if absent for more than 3 consecutive frames (~100ms)
  const noHands = rawNoHands && (handLossGraceCounter >= MAX_HAND_LOSS_GRACE_FRAMES);

  /* ── No Hands Detected ── */
  if (noHands) {
    statusDot.className = puzzle.solved ? "status-dot solved" : "status-dot";
    fistHoldCounter = 0;
    freezeGate.holding = false;

    // Release any dragged piece only after 3 consecutive empty frames
    if (drag.activeHand && drag.piece) {
      handleDragForHand(drag.activeHand, false, { x: drag.piece.x, y: drag.piece.y });
    }

    if (appState === "tracking") {
      const sinceLast = performance.now() - lastSeenFrame.at;
      if (lastSeenFrame.box && sinceLast < FRAME_GRACE_MS) {
        drawLiveFrameOverlay(lastSeenFrame.box);
      }
      if (isStripFull()) {
        statusText.innerHTML = '<i class="ph ph-film-strip"></i> Strip lengkap!';
        showInstruction('Unduh strip foto atau klik Bersihkan untuk mulai lagi');
      } else {
        statusText.innerHTML = '<i class="ph ph-magnifying-glass"></i> Mencari tangan…';
        showInstruction('Tunjukkan kedua tangan, lalu cubit untuk membuat bingkai');
      }
      return;
    }

    if (appState === "puzzle") {
      if (!timerExpired) updateTimer();
      puzzle.solved = reconcilePlacedState(puzzle.boardBox, puzzle.tileW, puzzle.tileH);
      updateProgressBadge();
      drawBoardAndPieces();
      if (puzzle.solved) {
        statusText.innerHTML = '<i class="ph ph-check-circle"></i> Puzzle selesai!';
        showInstruction('Kepalkan tangan untuk menyimpan ke strip');
      } else {
        statusText.innerHTML = '<i class="ph ph-puzzle-piece"></i> Sesi puzzle aktif';
        showInstruction('Cubit dengan satu tangan untuk menggeser kepingan');
      }
      return;
    }

    return;
  }

  /* ── Hands Detected ── */
  statusDot.className = puzzle.solved ? "status-dot solved" : "status-dot live";

  // Draw hand skeletons in tracking mode AND in solved puzzle final state
  if (appState === "tracking" || (appState === "puzzle" && puzzle.solved)) {
    handsLandmarks.forEach((lm, i) => {
      const label = i === 0 ? "A" : "B";
      const rawPx = lm.map(pt => toPixel(mirrorLandmarkX(pt)));
      const smoothedPx = getSmoothedLandmarks(label, rawPx, 0.42);
      drawHandSkeleton(smoothedPx);
    });
  }

  // Fist gesture detection (ONLY active when puzzle is SOLVED to save photo to strip)
  const anyFist = handsLandmarks.some(lm => isFist(lm));
  const draggingNow = drag.activeHand !== null && drag.piece !== null;
  if (appState === "puzzle" && puzzle.solved && anyFist && !draggingNow) {
    fistHoldCounter++;
    if (fistHoldCounter >= FIST_HOLD_FRAMES) {
      fistHoldCounter = 0;
      handleFistReset();
      return;
    }
  } else {
    fistHoldCounter = 0;
  }

  /* ── TRACKING STATE ── */
  if (appState === "tracking") {
    if (isStripFull()) {
      statusText.innerHTML = '<i class="ph ph-film-strip"></i> Strip lengkap!';
      showInstruction('Unduh strip foto atau klik Bersihkan untuk mulai lagi');
      return;
    }

    // 1. If currently in freezeGate holding state, hold & lock the frame box
    if (freezeGate.holding && freezeGate.box) {
      drawLiveFrameOverlay(freezeGate.box);
      lastSeenFrame.box = freezeGate.box;
      lastSeenFrame.at = performance.now();

      statusDot.className = "status-dot armed";
      statusText.innerHTML = '<i class="ph ph-camera"></i> Membingkai foto…';
      showInstruction('Membingkai foto… Tahan posisi');

      if (performance.now() - freezeGate.since > FREEZE_HOLD_MS) {
        const lockedBox = { ...freezeGate.box };
        freezeGate.holding = false;
        freezeGate.box = null;
        startCountdown(lockedBox);
        updateSettingsAvailability();
      }
      return;
    }

    // 2. Evaluate 2 hands framing
    if (handsLandmarks.length === 2) {
      const [handA, handB] = handsLandmarks;
      const indexA = mirrorLandmarkX(handA[LM.INDEX_TIP]);
      const indexB = mirrorLandmarkX(handB[LM.INDEX_TIP]);
      const frameBox = computeHandFrame(indexA, indexB);

      const minW = canvas.width * MIN_FRAME_RATIO;
      const minH = canvas.height * MIN_FRAME_RATIO;
      const isTooSmall = frameBox.width < minW || frameBox.height < minH;

      if (frameBox.width > 4 && frameBox.height > 4) {
        drawLiveFrameOverlay(frameBox, isTooSmall);
        lastSeenFrame.box = frameBox;
        lastSeenFrame.at = performance.now();
      }

      const bothPinching = isPinching(handA) && isPinching(handB);
      if (bothPinching && !isTooSmall) {
        freezeGate.holding = true;
        freezeGate.since = performance.now();
        freezeGate.box = { ...frameBox };

        statusDot.className = "status-dot armed";
        statusText.innerHTML = '<i class="ph ph-camera"></i> Membingkai foto…';
        showInstruction('Membingkai foto… Tahan posisi');

        if (performance.now() - freezeGate.since > FREEZE_HOLD_MS) {
          const lockedBox = { ...freezeGate.box };
          freezeGate.holding = false;
          freezeGate.box = null;
          startCountdown(lockedBox);
          updateSettingsAvailability();
        }
      } else if (bothPinching && isTooSmall) {
        statusText.innerHTML = '<i class="ph ph-warning"></i> Bingkai terlalu kecil';
        showInstruction('Bingkai terlalu kecil. Perluas area pengambilan gambar.');
      } else {
        statusText.innerHTML = '<i class="ph ph-hand"></i> 2 Tangan terdeteksi';
        showInstruction('Cubit dengan kedua tangan untuk membuat bingkai');
      }
    } else {
      const sinceLast = performance.now() - lastSeenFrame.at;
      if (lastSeenFrame.box && sinceLast < FRAME_GRACE_MS) {
        drawLiveFrameOverlay(lastSeenFrame.box);
        statusText.innerHTML = '<i class="ph ph-hand"></i> Tangan terdeteksi';
      } else {
        statusText.innerHTML = '<i class="ph ph-hand"></i> Tangan terdeteksi (butuh 2 tangan)';
      }
      showInstruction('Tunjukkan kedua tangan, lalu cubit untuk membuat bingkai');
    }
    return;
  }

  /* ── PUZZLE STATE ── */
  if (appState === "puzzle") {
    const labelsPresent = new Set();
    handsLandmarks.forEach((lm, i) => {
      const label = i === 0 ? "A" : "B";
      labelsPresent.add(label);
      const pinching = isPinching(lm);
      const indexPx = toPixel(mirrorLandmarkX(lm[LM.INDEX_TIP]));
      handleDragForHand(label, pinching, indexPx);
    });

    // If the active drag hand disappeared, release
    if (drag.activeHand && !labelsPresent.has(drag.activeHand) && drag.piece) {
      handleDragForHand(drag.activeHand, false, { x: drag.piece.x, y: drag.piece.y });
    }

    if (!drag.piece) {
      puzzle.solved = reconcilePlacedState(puzzle.boardBox, puzzle.tileW, puzzle.tileH);
      updateProgressBadge();
    }

    drawBoardAndPieces();
    drawHandSkeletonsOverBoard(handsLandmarks, puzzle.boardBox);

    if (!timerExpired) updateTimer();

    if (puzzle.solved) {
      if (fistHoldCounter > 0) {
        const remainingFrames = Math.max(1, FIST_HOLD_FRAMES - fistHoldCounter);
        const holdSec = Math.ceil(remainingFrames / 10);
        statusText.innerHTML = '<i class="ph ph-floppy-disk"></i> Menyimpan foto…';
        showInstruction(`Tahan kepalan tangan Anda. Menyimpan… (${holdSec}s)`);
      } else {
        statusText.innerHTML = '<i class="ph ph-check-circle"></i> Puzzle selesai!';
        showInstruction('Kepalkan tangan untuk menyimpan ke strip');
      }
    } else {
      statusText.innerHTML = '<i class="ph ph-puzzle-piece"></i> Sesi puzzle aktif';
      showInstruction('Cubit dengan satu tangan untuk geser kepingan');
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   EVENT LISTENERS
   ═══════════════════════════════════════════════════════════════════════ */

filterSelect.addEventListener("change", (e) => {
  activeFilter = e.target.value;
});

downloadStripBtn.addEventListener("click", downloadPhotoStrip);
resetAllBtn.addEventListener("click", resetEverything);

if (cancelFrameBtn) {
  cancelFrameBtn.addEventListener("click", () => {
    resetPuzzleOnly();
  });
}

// Mobile touch & pointer drag support on canvas
let pointerDragPiece = null;

function getCanvasPointerCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY
  };
}

canvas.addEventListener("pointerdown", (e) => {
  if (appState !== "puzzle" || puzzle.solved) return;
  const pt = getCanvasPointerCoords(e);
  const target = findNearestPiece(pt.x, pt.y);
  if (target && !target.placed) {
    pointerDragPiece = target;
    target.dragging = true;
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!pointerDragPiece || appState !== "puzzle") return;
  const pt = getCanvasPointerCoords(e);
  pointerDragPiece.x = pt.x - pointerDragPiece.w / 2;
  pointerDragPiece.y = pt.y - pointerDragPiece.h / 2;
});

const endPointerDrag = () => {
  if (!pointerDragPiece) return;
  const p = pointerDragPiece;
  pointerDragPiece = null;
  p.dragging = false;

  if (isNearOwnCell(p, puzzle.boardBox, puzzle.tileW, puzzle.tileH)) {
    snapPieceToCell(p, puzzle.boardBox, puzzle.tileW, puzzle.tileH);
  } else {
    const box = puzzle.boardBox;
    const cx = p.x + p.w / 2;
    const cy = p.y + p.h / 2;
    const dropCol = Math.min(GRID - 1, Math.max(0, Math.floor((cx - box.x) / puzzle.tileW)));
    const dropRow = Math.min(GRID - 1, Math.max(0, Math.floor((cy - box.y) / puzzle.tileH)));

    p.x = box.x + dropCol * puzzle.tileW;
    p.y = box.y + dropRow * puzzle.tileH;

    displaceCellOccupant(p, dropRow, dropCol, box, puzzle.tileW, puzzle.tileH);
  }

  puzzle.solved = reconcilePlacedState(puzzle.boardBox, puzzle.tileW, puzzle.tileH);
  updateProgressBadge();
};

canvas.addEventListener("pointerup", endPointerDrag);
canvas.addEventListener("pointercancel", endPointerDrag);

// Settings panel toggle function (called from inline onclick on #settingsBtn)
window.toggleSettings = function(e) {
  if (e) {
    if (e.stopPropagation) e.stopPropagation();
  }
  updateSettingsAvailability();
  const panel = document.getElementById("settingsPanel");
  if (panel) {
    panel.classList.toggle("open");
  }
};

// Prevent clicks inside the panel from closing it
const settingsPanelEl = document.getElementById("settingsPanel");
if (settingsPanelEl) {
  settingsPanelEl.addEventListener("click", (e) => {
    e.stopPropagation();
  });
}

// Close settings when clicking outside
document.addEventListener("click", (e) => {
  const panel = document.getElementById("settingsPanel");
  const btn = document.getElementById("settingsBtn");
  if (panel && btn && !panel.contains(e.target) && !btn.contains(e.target)) {
    panel.classList.remove("open");
  }
});

// Gallery sidebar toggle
const gallerySidebarEl = document.getElementById("gallerySidebar");
const galleryToggleBtnEl = document.getElementById("galleryToggleBtn");
if (galleryToggleBtnEl && gallerySidebarEl) {
  galleryToggleBtnEl.addEventListener("click", () => {
    gallerySidebarEl.classList.toggle("collapsed");
    setTimeout(fitCanvasToWindow, 300);
  });
}

// Timer limit radio buttons
document.querySelectorAll('input[name="timerLimit"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    if (radio.disabled) return;
    timerLimitSec = parseInt(e.target.value, 10);
    if (timerLimitSec <= 0) {
      hideTimerBadge();
    }
  });
});

// Theme mode radio buttons
document.querySelectorAll('input[name="themeMode"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    const theme = e.target.value;
    document.documentElement.setAttribute("data-theme", theme);
    const logoEl = document.getElementById("brandLogo");
    if (logoEl) {
      logoEl.src = theme === "light" ? "assets/logo-neo.png" : "assets/logo-neo-white.png";
    }
  });
});

// Confidence sensitivity range slider
const confidenceSliderEl = document.getElementById("confidenceSlider");
const confValueBadgeEl = document.getElementById("confValueBadge");

if (confidenceSliderEl) {
  confidenceSliderEl.addEventListener("input", (e) => {
    const valPct = parseInt(e.target.value, 10);
    if (confValueBadgeEl) confValueBadgeEl.textContent = `${valPct}%`;
    updateConfidenceThreshold(valPct / 100);
  });
}

// More / Advanced Settings accordion toggle
const moreSettingsToggleEl = document.getElementById("moreSettingsToggle");
const moreSettingsContentEl = document.getElementById("moreSettingsContent");

if (moreSettingsToggleEl && moreSettingsContentEl) {
  moreSettingsToggleEl.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = moreSettingsToggleEl.classList.toggle("open");
    moreSettingsContentEl.classList.toggle("hidden", !isOpen);
  });
}

// Camera selection dropdown
const cameraSelectEl = document.getElementById("cameraSelect");
if (cameraSelectEl) {
  cameraSelectEl.addEventListener("change", async (e) => {
    const deviceId = e.target.value;
    if (!deviceId) return;
    selectedDeviceId = deviceId;
    try {
      await initWebcam(deviceId);
    } catch (err) {
      console.error("[NeoPuzzle] Gagal mengganti kamera:", err);
      showError("Gagal mengganti ke kamera terpilih.");
    }
  });
}

window.addEventListener("DOMContentLoaded", init);
