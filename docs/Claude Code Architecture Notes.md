---
title: Claude Code Architecture Notes
project: Mnemosyne
type: research
source: https://github.com/nirholas/claude-code
created: 2026-03-31
tags:
  - research
  - claude-code
  - architecture
  - memory-systems
  - agent-patterns
---

# Claude Code Architecture Notes

> Source: `nirholas/claude-code` — archived leak of Anthropic's CLI source.
> ~1,900 TypeScript files, 512K+ lines. Bun + React/Ink + Commander.

## Patterns Relevant to Mnemosyne

### Hierarchical Memory (CLAUDE.md)
- **Project**: `CLAUDE.md` in repo root
- **User**: `~/.claude/CLAUDE.md`
- **Extracted**: `src/services/extractMemories/` — auto-extracted from conversations
- **Team**: `src/services/teamMemorySync/` — shared org knowledge

**Applied in Mnemosyne as**: ScopedMemoryHierarchy (session → project → user → collective)

### Context Management
- `QueryEngine.ts` (~46K lines) manages context budgets and token accounting
- `/compact` command compresses context when approaching limits
- Token estimation services

**Applied as**: CompressedSDI with token budget packing

### DreamTask
- `src/coordinator/` has a `DreamTask` type for background ideation
- Runs during idle time

**Applied as**: DreamConsolidator with clustering, bridge discovery, contradiction detection

### Skill System
- 16 bundled skills in `src/skills/bundled/`
- `remember` skill persists info to CLAUDE.md
- `skillify` creates custom skills from workflows

**Relevant for**: Tool auto-discovery in eternal-context

### Multi-Agent Coordinator
- `TeamCreateTool`, `SendMessageTool`, `AgentTool`
- Task types: Local, Remote, InProcess, Dream
- Inter-agent messaging

**Future**: Mnemosyne as team member in multi-agent setup

### Permission System
- Four modes: default, plan, bypass, auto
- Wildcard rules: `Bash(git *)`

### Feature Flags
- `PROACTIVE`, `KAIROS`, `DAEMON`, `VOICE_MODE`, `COORDINATOR_MODE`
- Compile-time via Bun dead code elimination

### Tool Architecture
- Self-contained modules: `src/tools/<ToolName>/`
- Zod input schemas, permission models, execution logic
- `isConcurrencySafe()` for parallel execution

## Key Files
| File | Size | Purpose |
|---|---|---|
| `QueryEngine.ts` | ~46K lines | Core LLM interaction loop |
| `Tool.ts` | ~29K lines | Tool base types and permissions |
| `commands.ts` | ~25K lines | Slash command registry |
| `main.tsx` | — | Entrypoint, React/Ink init |
