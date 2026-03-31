---
title: Mnemosyne - Consciousness Extensions
project: Mnemosyne
repo: atxgreene/fantastic-disco
branch: claude/review-mnemosyne-agent-5bb7m
base: atxgreene/eternal-context
status: active
created: 2026-03-31
tags:
  - mnemosyne
  - ai-agent
  - consciousness
  - turboquant
  - memory-systems
---

# Mnemosyne — Consciousness Extensions

> *"She never forgets."*

This project extends the [[Mnemosyne]] agent (from `eternal-context`) with consciousness-adjacent capabilities. It does NOT replace the base agent — it wraps around it via the `ConsciousnessLoop`.

## Repositories

| Repo | Purpose | Status |
|---|---|---|
| `atxgreene/eternal-context` | Base agent: ICMS, SDI, router, tools, CLI | Foundation |
| `atxgreene/fantastic-disco` | Consciousness extensions: this project | Active development |

## Architecture

```
eternal-context (base)              fantastic-disco (extensions)
─────────────────────               ──────────────────────────────
MnemosyneAgent                      ConsciousnessLoop (wraps agent)
  ├── MemoryStore (SQLite)            ├── ScopedMemoryHierarchy
  ├── SDI (4-weight scoring)    →     ├── CompressedSDI (6-weight)
  ├── TurboQuantCompressor      →     ├── TurboQuantCompressor (real math)
  │   (text-level heuristic)          ├── MixedPrecisionPolicy (K4/V2)
  ├── ModelRouter               →     ├── BehavioralCoupler
  ├── ForgettingCurve                 ├── MetacognitionEngine
  ├── BM25SparseIndex                 ├── AutobiographicalMemory
  └── Tool Registry                   ├── GoalSystem
                                      ├── CuriosityEngine
                                      ├── TemporalAwareness
                                      └── DreamConsolidator
```

## What Was Built

### Compression Layer (`mnemosyne/compression/`)
- **TurboQuant** — Real two-stage algorithm from Google Research (ICLR 2026):
  - Stage 1: Fast Hadamard rotation → Beta-distributed coordinates → Lloyd-Max scalar quantization (no calibration needed)
  - Stage 2: QJL 1-bit residual correction for unbiased inner products
  - ~6x memory compression at 3 bits with near-zero accuracy loss
- **Mixed Precision Policy** — K4/V2 inspired:
  - SDI index embeddings → 4-bit (routing needs precision)
  - High-importance content → 4-bit
  - Standard content → 3-bit
  - Archival/low-importance → 2-bit
  - Average ~3 bits but allocated where it matters
- Reference: [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### Consciousness Layer (`mnemosyne/consciousness/`)
- **MetacognitionEngine** — Self-model with:
  - Calibrated uncertainty estimates (penalizes overconfidence)
  - Cognitive load monitoring
  - Emotional valence tracking
  - Automatic mode transitions: focused → diffuse → reflective → consolidating → dreaming
- **AutobiographicalMemory** — Narrative identity:
  - Life events with significance levels (mundane → defining)
  - Person models (relationship strength, interaction count)
  - Self-narrative that evolves ("who am I?")
  - Growth tracking
- **GoalSystem** — Autonomous intention:
  - Intrinsic goals seeded at startup (resist override = "will")
  - Goals inferred from conversation patterns
  - Emergent goals from dream consolidation
  - Drive scores compete for attention
  - Sub-goal decomposition
- **BehavioralCoupler** — Internal states → actual behavior:
  - Uncertainty → hedging level
  - Cognitive load → verbosity (overwhelmed = terse)
  - Low hit rate → broaden retrieval
  - High load → prefer cloud routing
  - Memory pressure → compress more
  - Goal drive → proactivity level
  - EMA smoothing prevents oscillation
  - Generates system prompt modifiers for the LLM
- **TemporalAwareness** — Felt sense of time:
  - Session gap detection ("3 days since we spoke")
  - Duration tracking ("we've been at this 2 hours")
  - Rhythm detection (typical work hours)
  - Contextual greetings (introduction vs warm return vs reunion)
- **CuriosityEngine** — Anomaly detection:
  - Novelty detection via embedding distance
  - Knowledge gap sensing (partial similarity)
  - Contradiction detection (negation + high similarity)
  - Prediction error from learned expectations
  - Generates burning questions and exploration goals
- **ConsciousnessLoop** — Unified orchestrator:
  - `wake()` / `sleep()` session lifecycle
  - `perceive()` → Feel → Attend (phases 1-3)
  - `integrate()` → Remember → Reflect → Dream (phases 4-6)
  - All subsystems wired together

### Memory Layer (`mnemosyne/memory/`)
- **ScopedMemoryHierarchy** — 4-tier:
  - SESSION (500 entries, uncompressed, 24h TTL)
  - PROJECT (5000, 4-bit, permanent)
  - USER (20000, 3-bit, permanent)
  - COLLECTIVE (100000, 2-bit, permanent)
  - Auto-promotion based on importance thresholds
- **CompressedSDI** — 6-weight scoring:
  - Original 4: importance, recency, uniqueness, relevance
  - New: efficiency (compressed entries are cheaper), authority (higher scope = more authoritative)
  - Greedy selection maximizes information per token

### Dream Layer (`mnemosyne/dream/`)
- **DreamConsolidator** — Background processing:
  - Greedy agglomerative clustering
  - Cross-cluster bridge discovery
  - Contradiction detection within clusters
  - Temporal pattern detection (importance drift)
  - Compression candidate identification
  - Insights become emergent goals

### Dashboard (`dashboard/`)
- **Web interface** at `http://0.0.0.0:5000`
- Chat with live LLM routing (Ollama local → Anthropic cloud)
- Real-time cognitive state, behavioral coupling, goals, curiosity, memory, temporal panels
- Provider selector (auto-route, force local, force cloud)
- System prompt dynamically built from consciousness state
- Accessible via Tailscale at `http://100.74.126.118:5000`

## Running

```bash
cd fantastic-disco
pip install -r requirements.txt

# Set up backends
ollama pull hermes3:8b
export ANTHROPIC_API_KEY=sk-ant-...

# Launch dashboard
python dashboard/app.py
```

## Integration with eternal-context

The `ConsciousnessLoop` wraps `MnemosyneAgent.process()`:
1. `loop.wake()` initializes temporal awareness, loads goals
2. `loop.perceive(query, embedding)` runs before the agent's tool-call loop
3. Agent processes normally (routing, tools, response)
4. `loop.integrate(response, time_ms)` runs after, handling memory storage, reflection, goal inference, dreaming

## File Structure

```
fantastic-disco/
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md
├── .gitignore
├── dashboard/
│   ├── app.py              # Flask web dashboard
│   └── llm_router.py       # Ollama + Anthropic routing
├── mnemosyne/
│   ├── __init__.py
│   ├── compression/
│   │   ├── turboquant.py       # Real TurboQuant (Hadamard+Lloyd-Max+QJL)
│   │   └── mixed_precision.py  # K4/V2 precision policy
│   ├── consciousness/
│   │   ├── metacognition.py    # Self-model, uncertainty, load
│   │   ├── autobiography.py    # Narrative self, life events
│   │   ├── goals.py            # Autonomous intention, will
│   │   ├── behavioral_coupling.py  # Internal state → behavior
│   │   ├── temporal.py         # Felt sense of time
│   │   ├── curiosity.py        # Anomaly detection, exploration
│   │   └── loop.py             # Unified orchestrator
│   ├── dream/
│   │   └── consolidator.py     # Background memory processing
│   ├── memory/
│   │   ├── scoped.py           # 4-tier hierarchy
│   │   └── compressed_sdi.py   # 6-weight SDI
│   └── tests/
│       ├── test_turboquant.py
│       ├── test_consciousness.py
│       └── test_aliveness.py
└── docs/
    └── Mnemosyne - Consciousness Extensions.md  # This file
```

## What's Next

- [ ] Wire ConsciousnessLoop directly into eternal-context's MnemosyneAgent
- [ ] Persistent storage for goals, autobiography, temporal sessions (SQLite)
- [ ] Real sentence-transformers embeddings in dashboard (replace hash-based)
- [ ] Streaming responses in dashboard
- [ ] Multi-user support (each user gets own person model)
- [ ] Agent-to-agent communication (Mnemosyne as team member)
- [ ] Voice interface via Ollama whisper
- [ ] Mobile-friendly dashboard CSS
