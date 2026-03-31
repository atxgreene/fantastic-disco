---
title: TurboQuant Research Notes
project: Mnemosyne
type: research
source: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
created: 2026-03-31
tags:
  - research
  - compression
  - quantization
  - turboquant
  - kv-cache
---

# TurboQuant Research Notes

> Google Research, ICLR 2026. Authors: Zandieh, Daliri, Hadian, Mirrokni.

## What It Is

KV cache compression for LLM inference. Compresses attention memory to **3 bits per coordinate** with near-zero accuracy loss. Not weight quantization — targets the memory that grows linearly with sequence length.

At 32K tokens on 8B params, KV cache = ~4.6 GB in FP16. TurboQuant → ~770 MB.

## Two-Stage Algorithm

### Stage 1: PolarQuant
1. Multiply KV vector by **randomized Hadamard matrix** — O(d log d)
2. After rotation, each coordinate follows **Beta((d-1)/2, (d-1)/2)** distribution
3. Apply **Lloyd-Max scalar quantizer** with precomputed centroids from known Beta distribution
4. **No calibration data needed** — distribution is analytically known

### Stage 2: QJL (Quantized Johnson-Lindenstrauss)
1. Compute residual: original - quantized
2. Project through random Gaussian matrix
3. Keep only **sign bits**: `sign(S * residual)`
4. This provides **unbiased inner product estimation**

## Key Properties

| Property | Traditional | TurboQuant |
|---|---|---|
| Per-block constants | Required (1-2 bits overhead) | None |
| Calibration | Usually required | Data-oblivious |
| Training | Often required | Training-free |
| Inner products | Biased at low bits | Unbiased (via QJL) |
| Theory | Ad hoc | Near Shannon bound (~2.7x) |

## K4/V2 Mixed Precision

Keys determine which tokens to attend to → need precision (4-bit).
Values are content that gets averaged → tolerate compression (2-bit).
Average = 3 bits, but allocated where it matters.

## How We Use It in Mnemosyne

- **SDI index embeddings** = Keys → 4-bit (routing accuracy)
- **Memory content embeddings** = Values → 2-bit (bulk storage)
- **High-importance entries** → 4-bit regardless
- Implementation: `mnemosyne/compression/turboquant.py`

## Performance Numbers

| Config | Result |
|---|---|
| 4-bit, H100 | 8x attention speedup |
| 3.5-bit, LongBench | Matches FP16 baseline (50.06) |
| 3-bit, Needle-in-Haystack | 0.997 recall |
| KV cache reduction | At least 6x |

## Papers

- [TurboQuant (arXiv)](https://arxiv.org/html/2504.19874v1)
- [PolarQuant (arXiv)](https://arxiv.org/abs/2502.02617)
