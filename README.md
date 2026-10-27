---
title: One Layer at a Time
emoji: 🎞️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Run a 70B in 4 GB of VRAM. Watch what each token costs.
---

**▶ Try it live:** https://iamdgarcia-one-layer-at-a-time.static.hf.space
**Code on GitHub:** https://github.com/iamdgarcia/one-layer-at-a-time

# One Layer at a Time

A 70B model fits in 4 GB of VRAM. This page shows what that costs.

Layer-streaming inference (the [airllm](https://github.com/lyogavin/airllm) trick) keeps **one
layer of the model on the GPU at a time** and reads the rest from disk, for every token. VRAM
becomes a function of layer size, not model size. The price: every layer that does not fit makes
a disk round-trip, per token.

Pick a GPU (4-24 GB), a model (8B-671B MoE), a disk (NVMe/SATA/HDD) and optional 4bit weights.
Press Generate and watch the layers travel while a sentence appears at the speed the hardware
dictates. Meters count tokens, simulated elapsed time, bytes read from disk and layer trips.

**The hook number:** one 16-token sentence with a 70B at 4 GB on an NVMe reads
**~2.2 TB from the disk** and takes **~15 minutes**. With every layer resident, the same sentence
takes ~0.3 seconds. Same model, same GPU-compute; only the memory story changed.

## What is measured vs modeled

- Measured anchors: the airllm README VRAM table (70B ~4 GB, 27B 3.33 GB, 180B MoE 5.95 GB,
  671B ~12 GB) and real user reports in issues
  [#364](https://github.com/lyogavin/airllm/issues/364) (28.6 s/token, 3B, Windows) and
  [#374](https://github.com/lyogavin/airllm/issues/374) (30 min/query, 20B).
- Declared model: time/token = (non-resident layers) x (layer GB / disk GB/s) + (all layers) x
  (layer GB / 1000 GB/s GPU ingest). Disk bandwidths and the 1.2 GB VRAM overhead are declared
  typicals. The full constant table is in the page's "how this works" dialog.

This is a simulation with measured anchors, not a running model.

## Attribution

- [lyogavin/airllm](https://github.com/lyogavin/airllm) (Apache-2.0), the library this page
  visualizes; compression math from block-wise quantization ([arXiv 2212.09720](https://arxiv.org/abs/2212.09720)).
- Code: [iamdgarcia/one-layer-at-a-time](https://github.com/iamdgarcia/one-layer-at-a-time).
