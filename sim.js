// sim.js — One Layer at a Time
// Layer-streaming inference simulator (airllm-style). Pure physics + scene animation.
// All constants are measured anchors or declared assumptions (see provenance dialog).

(function (global) {
  'use strict';

  // ---------- measured / declared constants ----------
  var BYTES_PER_PARAM_FP16 = 2;              // fp16: 2 bytes/param (README table math)
  var QUANT_DIV = { 'fp16': 1, '4bit': 4 };  // 4bit block-wise weights: bytes/4 (airllm compression)
  var DISK_BW = {                            // typical sequential read, GB/s (declared)
    'NVMe SSD': 2.5,
    'SATA SSD': 0.5,
    'HDD 7200': 0.12
  };
  var GPU_BW = 1000;        // GB/s effective weight ingest, RTX-4090-class (declared)
  var VRAM_OVERHEAD = 1.2;  // GB reserved for activations + runtime (declared)

  var MODELS = {
    '8B':   { label: 'Qwen3 8B · dense fp16',        layers: 36,  total_gb: 8e9 * 2 / 1e9,   streamed_gb_per_layer_src: 'dense', vram_ref: '~1-2 GB' },
    '70B':  { label: 'Llama 3.x 70B · fp16',          layers: 80,  total_gb: 70e9 * 2 / 1e9,  streamed_gb_per_layer_src: 'dense', vram_ref: '~4 GB' },
    '405B': { label: 'Llama 3.1 405B · fp16',         layers: 126, total_gb: 405e9 * 2 / 1e9, streamed_gb_per_layer_src: 'dense', vram_ref: '~8 GB' },
    '671B': { label: 'DeepSeek-V3 671B · MoE ~37B active', layers: 61, total_gb: 37e9 * 2 / 1e9, streamed_gb_per_layer_src: 'moe', vram_ref: '~12 GB' }
  };

  var SENTENCE = ['The', ' model', ' fits', ' in', ' memory', '.',
                  ' The', ' time', ' is', ' the', ' price', '.',
                  ' No', ' GPU', ' upgrade', '.'];

  // ---------- pure physics ----------
  // layer bytes that must travel per layer, in GB
  function layerGB(modelKey, quant) {
    var m = MODELS[modelKey];
    // MoE: only routed experts travel per token (~active params); dense: full weights
    var totalGB = m.total_gb / (QUANT_DIV[quant] || 1);
    return totalGB / m.layers;
  }

  function residentLayers(modelKey, quant, vramGB) {
    var m = MODELS[modelKey];
    var usable = Math.max(0.1, vramGB - VRAM_OVERHEAD);
    var perLayer = layerGB(modelKey, quant);
    return Math.max(1, Math.min(m.layers, Math.floor(usable / perLayer)));
  }

  // seconds per token for a given config
  function tokenSeconds(modelKey, quant, vramGB, disk) {
    var m = MODELS[modelKey];
    var perLayer = layerGB(modelKey, quant);
    var resident = residentLayers(modelKey, quant, vramGB);
    var travel = m.layers - resident;
    var loadPerLayer = perLayer / DISK_BW[disk];   // disk round-trip dominates
    var computePerLayer = perLayer / GPU_BW;       // resident ingest/compute
    return travel * loadPerLayer + m.layers * computePerLayer;
  }

  // GB read from disk per token (the non-resident layers travel every token)
  function diskGBPerToken(modelKey, quant, vramGB) {
    var m = MODELS[modelKey];
    var resident = residentLayers(modelKey, quant, vramGB);
    return (m.layers - resident) * layerGB(modelKey, quant);
  }

  function sentenceStats(modelKey, quant, vramGB, disk) {
    var t = tokenSeconds(modelKey, quant, vramGB, disk);
    return {
      token_s: t,
      sentence_s: t * SENTENCE.length,
      sentence_gb: diskGBPerToken(modelKey, quant, vramGB) * SENTENCE.length,
      trips_per_token: MODELS[modelKey].layers - residentLayers(modelKey, quant, vramGB),
      resident: residentLayers(modelKey, quant, vramGB),
      layers: MODELS[modelKey].layers
    };
  }

  // ---------- scene ----------
  function Scene(canvas, opts) {
    this.cv = canvas;
    this.cx = canvas.getContext('2d');
    this.cfg = opts;             // {model, quant, vram, disk, speed}
    this.reset();
  }

  Scene.prototype.reset = function () {
    this.tokenIdx = 0;
    this.layerIdx = 0;
    this.phase = 'idle';         // idle | toslot | compute | done-token
    this.tPhase = 0;
    this.elapsed = 0;            // real seconds (simulated)
    this.gbRead = 0;
    this.trips = 0;
    this.done = false;
    this.fly = null;             // {fromY, t}
    this.pulse = 0;
  };

  Scene.prototype.step = function (dtMs) {
    if (this.done) return;
    var st = sentenceStats(this.cfg.model, this.cfg.quant, this.cfg.vram, this.cfg.disk);
    var realDt = (dtMs / 1000) * this.cfg.speed;   // simulated seconds advanced
    this.elapsed += realDt;
    var m = MODELS[this.cfg.model];
    var resident = st.resident;
    var loadPerLayer = layerGB(this.cfg.model, this.cfg.quant) / DISK_BW[this.cfg.disk];
    var computePerLayer = layerGB(this.cfg.model, this.cfg.quant) / GPU_BW;
    // advance through layers consuming their TRUE time share (animation == sim time)
    var budget = realDt;
    while (budget > 0 && this.tokenIdx < SENTENCE.length) {
      var isResident = this.layerIdx < resident;
      var need = isResident ? computePerLayer : (loadPerLayer + computePerLayer);
      if (budget < need) break;
      budget -= need;
      if (!isResident) { this.gbRead += layerGB(this.cfg.model, this.cfg.quant); this.trips += 1; this.pulse = 0; }
      this.layerIdx += 1;
      if (this.layerIdx >= m.layers) {
        this.layerIdx = 0;
        this.tokenIdx += 1;
        if (this.tokenIdx >= SENTENCE.length) { this.done = true; }
      }
    }
    this.pulse = (this.pulse + dtMs) % 1000;
  };

  Scene.prototype.draw = function () {
    var cx = this.cx, W = this.cv.width, H = this.cv.height;
    var st = sentenceStats(this.cfg.model, this.cfg.quant, this.cfg.vram, this.cfg.disk);
    var m = MODELS[this.cfg.model];
    cx.clearRect(0, 0, W, H);
    var N = m.layers;
    var stackX = 30, stackW = 70, top = 30, bot = H - 30;
    var barH = (bot - top) / N;
    var slotX = W - 190, slotW = 150, slotTop = 60, slotBot = H - 60;
    var resident = st.resident;

    // disk stack
    cx.fillStyle = '#8a93a5';
    cx.font = '11px ui-monospace, monospace';
    cx.fillText('DISK (' + this.cfg.disk + ', ' + DISK_BW[this.cfg.disk] + ' GB/s)', stackX, top - 10);
    for (var i = 0; i < N; i++) {
      var y = bot - (i + 1) * barH;
      if (i < this.layerIdx && this.tokenIdx < SENTENCE.length) {
        cx.fillStyle = '#3d4657'; // already walked this token
      } else {
        cx.fillStyle = '#565f73';
      }
      cx.fillRect(stackX, y + 1, stackW, Math.max(1.5, barH - 2));
      if (i === this.layerIdx && !this.done) {
        cx.fillStyle = '#4c8dff';
        cx.fillRect(stackX - 4, y - 1, stackW + 8, Math.max(2, barH + 1));
      }
    }
    cx.fillStyle = '#8a93a5';
    cx.fillText(N + ' layers · ' + (m.total_gb).toFixed(0) + ' GB' + (this.cfg.quant === '4bit' ? ' → 4bit' : ''), stackX, bot + 16);

    // GPU slot
    cx.strokeStyle = '#2a3242';
    cx.strokeRect(slotX, slotTop, slotW, slotBot - slotTop);
    cx.fillStyle = '#8a93a5';
    cx.fillText('GPU ' + this.cfg.vram + ' GB', slotX, slotTop - 10);
    var slotH = (slotBot - slotTop - 10) / Math.max(resident, 1);
    for (var r = 0; r < resident; r++) {
      var yy = slotBot - 5 - (r + 1) * slotH;
      var flash = (!this.done && r === 0 && this.pulse < 120) ? '#ffd166' : '#3fb96f';
      cx.fillStyle = flash;
      cx.fillRect(slotX + 5, yy + 1, slotW - 10, Math.max(2, slotH - 2));
    }
    if (resident < N) {
      cx.fillStyle = '#5a6377';
      cx.fillText(resident + ' of ' + N + ' resident', slotX, slotBot + 16);
      cx.fillText((N - resident) + ' travel per token', slotX, slotBot + 30);
    } else {
      cx.fillStyle = '#3fb96f';
      cx.fillText('all ' + N + ' layers resident', slotX, slotBot + 16);
    }

    // token arrow between disk and slot
    cx.fillStyle = this.done ? '#3fb96f' : '#4c8dff';
    cx.fillText(this.done ? 'sentence complete' : 'token ' + (this.tokenIdx + 1) + ' / ' + SENTENCE.length,
                (stackX + stackW + slotX) / 2 - 46, H / 2 - 4);
  };

  // ---------- exports ----------
  var API = {
    MODELS: MODELS,
    SENTENCE: SENTENCE,
    DISK_BW: DISK_BW,
    QUANT_DIV: QUANT_DIV,
    VRAM_OVERHEAD: VRAM_OVERHEAD,
    layerGB: layerGB,
    residentLayers: residentLayers,
    tokenSeconds: tokenSeconds,
    diskGBPerToken: diskGBPerToken,
    sentenceStats: sentenceStats,
    Scene: Scene
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = API; }
  else { global.Sim = API; }
})(typeof window !== 'undefined' ? window : globalThis);
