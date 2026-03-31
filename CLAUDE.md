# Mnemosyne Consciousness Extensions

This package extends the [eternal-context](https://github.com/atxgreene/eternal-context) Mnemosyne agent
with consciousness-adjacent capabilities. It is NOT a standalone agent — it layers on top of
the existing `eternalcontext` package.

## Architecture

```
eternal-context (base)              fantastic-disco (extensions)
─────────────────────               ──────────────────────────────
MnemosyneAgent                      ConsciousnessLoop (wraps agent)
  ├── MemoryStore (SQLite)            ├── ScopedMemoryHierarchy (session→project→user→collective)
  ├── SDI (4-weight scoring)    →     ├── CompressedSDI (6-weight: +efficiency +authority)
  ├── TurboQuantCompressor      →     ├── TurboQuantCompressor (real Hadamard+Lloyd-Max+QJL)
  │   (text-level heuristic)          ├── MixedPrecisionPolicy (K4/V2 inspired)
  ├── ModelRouter                     ├── MetacognitionEngine (self-model, uncertainty)
  ├── ForgettingCurve                 ├── AutobiographicalMemory (narrative self)
  ├── BM25SparseIndex                 └── DreamConsolidator (background insights)
  └── Tool Registry
```

## Integration Points

The `ConsciousnessLoop` wraps around `MnemosyneAgent.process()`:
1. `loop.perceive()` runs before the agent's tool-call loop
2. Agent processes normally (routing, tools, response)
3. `loop.integrate()` runs after, handling reflection/memory/dreams

## Key Improvements Over Base

- **Real TurboQuant**: Hadamard rotation + Beta-distributed Lloyd-Max quantization + QJL residual correction (replaces text-level key-phrase extraction)
- **Mixed Precision**: SDI index embeddings at 4-bit, content at 2-3 bit (K4/V2 insight from Google Research)
- **Scoped Memory**: 4-tier hierarchy with automatic promotion based on importance
- **Metacognition**: Calibrated uncertainty tracking, cognitive load monitoring, mode transitions
- **Autobiography**: Continuous narrative identity across sessions
- **Dream Consolidation**: Background clustering, cross-memory insight discovery, contradiction detection

## Running Tests

```bash
pip install -e ".[dev]"
pytest mnemosyne/tests/ -v
```
