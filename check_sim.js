// check_sim.js — node smoke check: physics anchors vs measured README/issue values.
// Run: node check_sim.js   (0 deps; exits non-zero on first failure)
'use strict';
const S = require('./sim.js');

let n = 0;
function ok(name, cond) {
  n += 1;
  if (!cond) { console.error('FAIL: ' + name); process.exit(1); }
  console.log('ok  ' + name);
}
function approx(name, got, want, tol) {
  ok(name + ' (got ' + got.toFixed(3) + ', want ' + want + ' ±' + tol + ')', Math.abs(got - want) <= tol);
}

// 1. dense layer math: 70B fp16 = 140 GB / 80 layers
approx('70B layerGB', S.layerGB('70B', 'fp16'), 1.75, 0.01);

// 2. README anchor: 70B fits ~4 GB -> exactly 1 resident layer
ok('70B resident @4GB === 1', S.residentLayers('70B', 'fp16', 4) === 1);

// 3. token time @4GB/NVMe: 79 travel layers x (1.75/2.5) + compute ~55.4 s
approx('70B token seconds @4GB NVMe', S.tokenSeconds('70B', 'fp16', 4, 'NVMe SSD'), 55.44, 1.0);

// 4. the hook number: disk read per token @4GB = 79 x 1.75 = 138.25 GB
approx('70B disk GB/token @4GB', S.diskGBPerToken('70B', 'fp16', 4), 138.25, 0.5);

// 5. 8B on 24GB: all layers resident, tokens fly
ok('8B all resident @24GB', S.residentLayers('8B', 'fp16', 24) === 36);
ok('8B fast when resident (<30ms/token)', S.tokenSeconds('8B', 'fp16', 24, 'NVMe SSD') < 0.03);

// 6. 4bit compression: layer bytes /4 -> ~13 s/token @4GB (README claims 3x end-to-end)
approx('70B 4bit token seconds @4GB', S.tokenSeconds('70B', '4bit', 4, 'NVMe SSD'), 12.99, 0.5);

// 7. MoE: only active experts travel; 671B layer = 74/61 = 1.213 GB; @16GB -> 12 resident
approx('671B MoE layerGB', S.layerGB('671B', 'fp16'), 1.2131, 0.01);
ok('671B resident @16GB === 12', S.residentLayers('671B', 'fp16', 16) === 12);

// 8. 405B @8GB: 1 resident layer (6.43 GB), ~321 s/token
approx('405B layerGB', S.layerGB('405B', 'fp16'), 6.4286, 0.02);
approx('405B token seconds @8GB', S.tokenSeconds('405B', 'fp16', 8, 'NVMe SSD'), 321.4, 5.0);

// 9. sentence stats consistency: totals = 16 x per-token
var st = S.sentenceStats('70B', 'fp16', 4, 'NVMe SSD');
approx('sentence GB = 16 x per-token', st.sentence_gb, 16 * 138.25, 1.0);
approx('sentence seconds = 16 x per-token', st.sentence_s, 16 * 55.44, 2.0);
ok('trips per token === 79', st.trips_per_token === 79);

// 10. disk hierarchy: HDD is ~20.8x slower on travel than NVMe
var r = S.tokenSeconds('70B', 'fp16', 4, 'HDD 7200') / S.tokenSeconds('70B', 'fp16', 4, 'NVMe SSD');
approx('HDD/NVMe slowdown ratio', r, 2.5 / 0.12, 0.3);

console.log('ALL PASS (' + n + ' checks)');
