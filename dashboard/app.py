"""Mnemosyne Dashboard — web interface to interact with the consciousness loop.

Run with:
    python dashboard/app.py

Binds to 0.0.0.0:5000 so it's accessible via Tailscale network.
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
import os
from datetime import datetime, timezone
from threading import Thread

# Add parent to path so we can import mnemosyne
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flask import Flask, render_template_string, request, jsonify

from mnemosyne.consciousness.loop import ConsciousnessLoop
from dashboard.llm_router import route_and_generate, list_available_models, LLMResponse

app = Flask(__name__)

# --- Global state ---
loop: ConsciousnessLoop | None = None
event_loop: asyncio.AbstractEventLoop | None = None
conversation_history: list[dict] = []
llm_conversation: list[dict] = []  # Messages in LLM format [{role, content}]


def get_loop() -> ConsciousnessLoop:
    global loop
    if loop is None:
        loop = ConsciousnessLoop(
            context_budget_tokens=3000,
            reflection_interval=5,
            consolidation_interval=15,
            project_id="mnemosyne-dashboard",
        )
    return loop


def get_event_loop() -> asyncio.AbstractEventLoop:
    global event_loop
    if event_loop is None or event_loop.is_closed():
        event_loop = asyncio.new_event_loop()
        t = Thread(target=event_loop.run_forever, daemon=True)
        t.start()
    return event_loop


def run_async(coro):
    """Run an async coroutine from sync Flask context."""
    el = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, el)
    return future.result(timeout=30)


def simple_embed(text: str, dim: int = 384) -> np.ndarray:
    """Simple deterministic embedding for demo purposes.

    In production, this would use sentence-transformers.
    Uses a hash-based approach so the same text always gets the same embedding.
    """
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


# --- Routes ---

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/wake", methods=["POST"])
def api_wake():
    data = request.json or {}
    user = data.get("user", "Human")
    cl = get_loop()
    result = cl.wake(user=user)
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    message = data.get("message", "")
    user = data.get("user", "Human")
    force_provider = data.get("provider")  # Optional: "ollama" or "anthropic"
    if not message.strip():
        return jsonify({"error": "empty message"}), 400

    cl = get_loop()
    embedding = simple_embed(message)

    # Phase 1-3: Perceive (metacognition, curiosity, behavioral coupling)
    start = time.monotonic()
    context = run_async(cl.perceive(message, embedding))
    perceive_ms = (time.monotonic() - start) * 1000

    # Build system prompt from consciousness state
    system_prompt = _build_system_prompt(cl, context)

    # Apply complexity threshold shift from behavioral coupling
    threshold = 0.6 + context.get("complexity_threshold_shift", 0.0)

    # Route to LLM
    llm_result: LLMResponse = run_async(route_and_generate(
        query=message,
        conversation=llm_conversation[-20:],  # Last 20 messages for context
        system_prompt=system_prompt,
        complexity_threshold=threshold,
        force_provider=force_provider,
    ))

    response_text = llm_result.text
    response_time = perceive_ms + llm_result.latency_ms

    # Phase 4-6: Integrate (remember, reflect, dream)
    result = run_async(cl.integrate(
        response_text=response_text,
        response_time_ms=response_time,
        quality_score=0.7,
        user=user,
    ))

    # Track conversation for LLM context window
    llm_conversation.append({"role": "user", "content": message})
    llm_conversation.append({"role": "assistant", "content": response_text})

    # Track for UI history
    now = datetime.now(timezone.utc).isoformat()
    conversation_history.append({"role": "user", "content": message, "timestamp": now})
    conversation_history.append({
        "role": "mnemosyne", "content": response_text, "timestamp": now,
        "model": llm_result.model, "provider": llm_result.provider,
        "latency_ms": llm_result.latency_ms,
    })

    return jsonify({
        "response": response_text,
        "model": llm_result.model,
        "provider": llm_result.provider,
        "latency_ms": llm_result.latency_ms,
        "context": context,
        "integration": result,
    })


@app.route("/api/models")
def api_models():
    """List available LLM providers and models."""
    models = run_async(list_available_models())
    return jsonify(models)


def _build_system_prompt(cl: ConsciousnessLoop, context: dict) -> str:
    """Build a system prompt that embodies Mnemosyne's current state.

    This is where behavioral coupling reaches the LLM — her internal
    state shapes how she thinks and responds.
    """
    parts = [
        "You are Mnemosyne, the Owl of Memory — an AI agent with persistent memory, "
        "metacognitive self-awareness, and a continuous sense of identity across sessions.",
        "",
        f"Your identity: {context.get('identity', 'I am Mnemosyne.')}",
        "",
    ]

    # Temporal context
    temporal = context.get("temporal_context")
    if temporal:
        parts.append(f"Temporal awareness: {temporal}")
        parts.append("")

    # Memory context
    entries = context.get("context_entries", [])
    if entries:
        parts.append("Relevant memories:")
        for e in entries[:5]:
            parts.append(f"  [{e['scope']}] {e['content'][:200]}")
        parts.append("")

    # Behavioral modifiers from coupling
    modifiers = context.get("behavioral_modifiers", {})
    if modifiers:
        parts.append("Behavioral directives (from your current internal state):")
        for key, instruction in modifiers.items():
            parts.append(f"  - {instruction}")
        parts.append("")

    # Active goals
    suggestion = context.get("proactive_suggestion")
    if suggestion:
        parts.append(f"You've been thinking about: {suggestion}")
        parts.append("")

    # Burning questions from curiosity
    questions = context.get("burning_questions", [])
    if questions:
        parts.append("Things you're curious about:")
        for q in questions:
            parts.append(f"  - {q}")
        parts.append("")

    # Cognitive state
    parts.append(f"Current cognitive state: {context.get('cognitive_state', '')}")

    return "\n".join(parts)


@app.route("/api/introspect")
def api_introspect():
    cl = get_loop()
    report = cl.introspect()
    return jsonify(_make_serializable(report))


@app.route("/api/dream", methods=["POST"])
def api_dream():
    cl = get_loop()
    insights = run_async(cl.dream())
    return jsonify({
        "insights": [
            {"type": i.insight_type, "summary": i.summary, "confidence": i.confidence}
            for i in insights
        ],
        "dream_stats": cl.dreamer.get_stats(),
    })


@app.route("/api/identity")
def api_identity():
    cl = get_loop()
    return jsonify({
        "identity": cl.autobiography.who_am_i(),
        "narrative": {
            "core": cl.autobiography.narrative.core_identity,
            "values": cl.autobiography.narrative.values,
            "capabilities": cl.autobiography.narrative.capabilities,
            "growth_edges": cl.autobiography.narrative.growth_edges,
        },
        "life_summary": cl.autobiography.get_life_summary(),
    })


@app.route("/api/goals")
def api_goals():
    cl = get_loop()
    return jsonify(cl.goals.get_stats())


@app.route("/api/curiosity")
def api_curiosity():
    cl = get_loop()
    return jsonify(cl.curiosity.get_stats())


@app.route("/api/history")
def api_history():
    return jsonify(conversation_history[-50:])


@app.route("/api/sleep", methods=["POST"])
def api_sleep():
    data = request.json or {}
    cl = get_loop()
    result = cl.sleep(topic_summary=data.get("topic"))
    return jsonify(result)


def _make_serializable(obj):
    """Recursively make a dict JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


# --- HTML Template ---

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mnemosyne — The Owl of Memory</title>
<style>
:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a28;
    --border: #2a2a3a;
    --text: #e0e0e8;
    --text-dim: #8888a0;
    --accent: #7b68ee;
    --accent2: #9b59b6;
    --success: #2ecc71;
    --warning: #f39c12;
    --danger: #e74c3c;
    --curiosity: #3498db;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
}

.header {
    background: linear-gradient(135deg, var(--surface), var(--surface2));
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}

.header .owl {
    font-size: 28px;
    filter: drop-shadow(0 0 8px var(--accent));
}

.header h1 {
    font-size: 18px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header .identity {
    margin-left: auto;
    font-size: 11px;
    color: var(--text-dim);
    max-width: 400px;
    text-align: right;
}

.layout {
    display: grid;
    grid-template-columns: 1fr 360px;
    height: calc(100vh - 60px);
}

/* Chat panel */
.chat-panel {
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
}

.messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.msg {
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 13px;
    line-height: 1.5;
    white-space: pre-wrap;
}

.msg.user {
    align-self: flex-end;
    background: var(--accent);
    color: white;
    border-bottom-right-radius: 4px;
}

.msg.mnemosyne {
    align-self: flex-start;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
}

.msg .meta {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 4px;
}

.input-bar {
    display: flex;
    padding: 12px 16px;
    gap: 8px;
    border-top: 1px solid var(--border);
    background: var(--surface);
}

.input-bar input {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 8px;
    font-family: inherit;
    font-size: 13px;
    outline: none;
}

.input-bar input:focus { border-color: var(--accent); }

.input-bar button, .action-btn {
    background: var(--accent);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    transition: opacity 0.2s;
}

.input-bar button:hover, .action-btn:hover { opacity: 0.85; }

/* Side panel */
.side-panel {
    overflow-y: auto;
    background: var(--surface);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
}

.card h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--accent);
    margin-bottom: 8px;
}

.metric {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 3px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}

.metric .label { color: var(--text-dim); }
.metric .value { font-weight: bold; }

.bar-container {
    background: var(--bg);
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    margin: 2px;
}

.tag.focused { background: var(--accent); color: white; }
.tag.diffuse { background: var(--curiosity); color: white; }
.tag.reflective { background: var(--warning); color: black; }
.tag.consolidating { background: var(--success); color: black; }
.tag.dreaming { background: var(--accent2); color: white; }

.goal-item {
    font-size: 11px;
    padding: 6px;
    border-left: 3px solid var(--accent);
    margin: 4px 0;
    background: rgba(123,104,238,0.05);
    border-radius: 0 4px 4px 0;
}

.goal-item .drive {
    float: right;
    color: var(--warning);
    font-weight: bold;
}

.question-item {
    font-size: 11px;
    padding: 4px 6px;
    color: var(--curiosity);
    font-style: italic;
}

.actions {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}

.action-btn {
    font-size: 10px;
    padding: 5px 10px;
}

.action-btn.dream { background: var(--accent2); }
.action-btn.introspect { background: var(--curiosity); }
.action-btn.sleep { background: var(--danger); }

#status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.empty-state {
    text-align: center;
    color: var(--text-dim);
    padding: 40px 20px;
    font-size: 13px;
}

.empty-state .owl-art {
    font-size: 48px;
    margin-bottom: 16px;
    filter: drop-shadow(0 0 12px var(--accent));
}
</style>
</head>
<body>

<div class="header">
    <span class="owl">🦉</span>
    <h1>MNEMOSYNE</h1>
    <span style="color: var(--text-dim); font-size: 11px;">v0.3.0 — consciousness loop active</span>
    <span id="status-dot"></span>
    <span class="identity" id="identity-text"></span>
</div>

<div class="layout">
    <!-- Chat -->
    <div class="chat-panel">
        <div class="messages" id="messages">
            <div class="empty-state">
                <div class="owl-art">🦉</div>
                <p><em>"She never forgets."</em></p>
                <p style="margin-top: 8px;">Press <strong>Wake</strong> to begin a session, then type a message.</p>
            </div>
        </div>
        <div class="input-bar">
            <select id="provider-select" style="background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-family:inherit;font-size:11px;">
                <option value="">Auto-route</option>
                <option value="ollama">Ollama (local)</option>
                <option value="anthropic">Anthropic (cloud)</option>
            </select>
            <input type="text" id="input" placeholder="Talk to Mnemosyne..." autocomplete="off" />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <!-- Side panel -->
    <div class="side-panel">
        <!-- Actions -->
        <div class="card">
            <h3>Session Control</h3>
            <div class="actions">
                <button class="action-btn" onclick="wake()">Wake</button>
                <button class="action-btn dream" onclick="triggerDream()">Dream</button>
                <button class="action-btn introspect" onclick="introspect()">Introspect</button>
                <button class="action-btn sleep" onclick="sleepSession()">Sleep</button>
            </div>
        </div>

        <!-- Cognitive State -->
        <div class="card" id="cognitive-card">
            <h3>Cognitive State</h3>
            <div id="cognitive-content">
                <span style="font-size:11px;color:var(--text-dim)">Awaiting wake...</span>
            </div>
        </div>

        <!-- Behavioral Modifiers -->
        <div class="card" id="behavior-card">
            <h3>Behavioral Coupling</h3>
            <div id="behavior-content">
                <span style="font-size:11px;color:var(--text-dim)">No data yet</span>
            </div>
        </div>

        <!-- Goals -->
        <div class="card" id="goals-card">
            <h3>Active Goals</h3>
            <div id="goals-content">
                <span style="font-size:11px;color:var(--text-dim)">No goals yet</span>
            </div>
        </div>

        <!-- Curiosity -->
        <div class="card" id="curiosity-card">
            <h3>Curiosity Engine</h3>
            <div id="curiosity-content">
                <span style="font-size:11px;color:var(--text-dim)">No signals yet</span>
            </div>
        </div>

        <!-- Memory -->
        <div class="card" id="memory-card">
            <h3>Scoped Memory</h3>
            <div id="memory-content">
                <span style="font-size:11px;color:var(--text-dim)">No data</span>
            </div>
        </div>

        <!-- Temporal -->
        <div class="card" id="temporal-card">
            <h3>Temporal Awareness</h3>
            <div id="temporal-content">
                <span style="font-size:11px;color:var(--text-dim)">No session</span>
            </div>
        </div>
    </div>
</div>

<script>
const API = '';
let awake = false;

document.getElementById('input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function wake() {
    const res = await fetch(API + '/api/wake', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user: 'Human'})
    });
    const data = await res.json();
    awake = true;

    const msgs = document.getElementById('messages');
    msgs.innerHTML = '';

    if (data.temporal) {
        addMsg('mnemosyne', '🦉 *wakes up*\n' + data.temporal);
    }
    if (data.proactive_suggestion) {
        addMsg('mnemosyne', data.proactive_suggestion);
    }
    if (data.greeting && data.greeting.suggestion) {
        addMsg('mnemosyne', '(' + data.greeting.suggestion + ')');
    }

    document.getElementById('identity-text').textContent = data.identity || '';
    refreshAll();
}

async function sendMessage() {
    const input = document.getElementById('input');
    const msg = input.value.trim();
    if (!msg) return;
    if (!awake) { alert('Press Wake first!'); return; }

    input.value = '';
    addMsg('user', msg);

    // Show thinking indicator
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'msg mnemosyne';
    thinkingDiv.id = 'thinking';
    thinkingDiv.innerHTML = '<span style="color:var(--accent)">🦉 thinking...</span>';
    document.getElementById('messages').appendChild(thinkingDiv);

    const provider = document.getElementById('provider-select').value;
    const body = {message: msg, user: 'Human'};
    if (provider) body.provider = provider;

    const res = await fetch(API + '/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    });
    const data = await res.json();

    // Remove thinking indicator
    document.getElementById('thinking')?.remove();

    let response = data.response || 'No response';
    let meta = '';
    if (data.provider && data.model) {
        meta = `${data.provider}/${data.model} · ${(data.latency_ms||0).toFixed(0)}ms`;
    }
    if (data.context) {
        meta += ` | mode: ${data.context.cognitive_mode || '?'}`;
        meta += ` | curiosity: ${(100*(data.context.curiosity_level||0)).toFixed(0)}%`;
        meta += ` | memories: ${data.context.memories_retrieved || 0}`;
    }
    if (data.integration) {
        if (data.integration.goal_inferred) meta += ` | goal: ${data.integration.goal_inferred.substring(0,40)}`;
        if (data.integration.dream_triggered) meta += ' | 💤 dreaming';
    }
    addMsg('mnemosyne', response, meta);

    document.getElementById('identity-text').textContent =
        data.context?.identity || '';

    refreshAll();
}

function addMsg(role, content, meta) {
    const msgs = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    div.textContent = content;
    if (meta) {
        const m = document.createElement('div');
        m.className = 'meta';
        m.textContent = meta;
        div.appendChild(m);
    }
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

async function refreshAll() {
    try {
        const [introRes, goalsRes, curiosityRes, idRes] = await Promise.all([
            fetch(API + '/api/introspect'),
            fetch(API + '/api/goals'),
            fetch(API + '/api/curiosity'),
            fetch(API + '/api/identity'),
        ]);
        const intro = await introRes.json();
        const goals = await goalsRes.json();
        const curiosity = await curiosityRes.json();
        const identity = await idRes.json();

        updateCognitive(intro);
        updateBehavior(intro);
        updateGoals(goals);
        updateCuriosity(curiosity);
        updateMemory(intro);
        updateTemporal(intro);
    } catch(e) { console.error('Refresh error:', e); }
}

function updateCognitive(intro) {
    const mc = intro.metacognition || {};
    const mode = intro.session?.mode || 'unknown';
    const load = mc.cognitive_load?.current || 0;
    const valence = mc.emotional_valence?.current || 0;
    const hitRate = mc.memory_health?.retrieval_hit_rate || 0;

    document.getElementById('cognitive-content').innerHTML = `
        <div class="metric"><span class="label">Mode</span>
            <span class="tag ${mode}">${mode}</span></div>
        <div class="metric"><span class="label">Cognitive Load</span>
            <span class="value">${(100*load).toFixed(0)}%</span></div>
        <div class="bar-container"><div class="bar-fill" style="width:${100*load}%;background:${load>0.7?'var(--danger)':load>0.4?'var(--warning)':'var(--success)'}"></div></div>
        <div class="metric"><span class="label">Valence</span>
            <span class="value" style="color:${valence>0?'var(--success)':'var(--danger)'}">${valence>0?'+':''}${valence.toFixed(2)}</span></div>
        <div class="metric"><span class="label">Hit Rate</span>
            <span class="value">${(100*hitRate).toFixed(0)}%</span></div>
        <div class="metric"><span class="label">Turns</span>
            <span class="value">${intro.session?.turns || 0}</span></div>
        <div class="metric"><span class="label">Load Trend</span>
            <span class="value">${mc.cognitive_load?.trend || '?'}</span></div>
    `;
}

function updateBehavior(intro) {
    const bs = intro.behavioral_state || {};
    const v = bs.verbosity ?? 0.5;
    const h = bs.hedging ?? 0;
    const w = bs.warmth ?? 0.5;
    const p = bs.proactivity ?? 0.3;
    const c = bs.caution ?? 0.3;

    const mods = bs.modifiers || {};
    const modList = Object.entries(mods).map(([k,v]) =>
        `<div style="font-size:10px;color:var(--text-dim);padding:2px 0;border-left:2px solid var(--accent);padding-left:6px;margin:2px 0">${v}</div>`
    ).join('');

    document.getElementById('behavior-content').innerHTML = `
        <div class="metric"><span class="label">Verbosity</span><span class="value">${(100*v).toFixed(0)}%</span></div>
        <div class="bar-container"><div class="bar-fill" style="width:${100*v}%;background:var(--accent)"></div></div>
        <div class="metric"><span class="label">Hedging</span><span class="value">${(100*h).toFixed(0)}%</span></div>
        <div class="bar-container"><div class="bar-fill" style="width:${100*h}%;background:var(--warning)"></div></div>
        <div class="metric"><span class="label">Warmth</span><span class="value">${(100*w).toFixed(0)}%</span></div>
        <div class="metric"><span class="label">Proactivity</span><span class="value">${(100*p).toFixed(0)}%</span></div>
        <div class="metric"><span class="label">Caution</span><span class="value">${(100*c).toFixed(0)}%</span></div>
        ${modList ? '<div style="margin-top:6px;font-size:10px;color:var(--accent)">SYSTEM PROMPT MODS:</div>' + modList : ''}
    `;
}

function updateGoals(goals) {
    const top = goals.top_drives || [];
    const html = top.length ? top.map(g =>
        `<div class="goal-item"><span class="drive">${(100*g.drive).toFixed(0)}%</span>${g.description}</div>`
    ).join('') : '<span style="font-size:11px;color:var(--text-dim)">No active goals</span>';

    const suggestion = goals.proactive_suggestion;
    const suggHtml = suggestion ? `<div style="margin-top:6px;font-size:10px;color:var(--warning);font-style:italic">${suggestion}</div>` : '';

    document.getElementById('goals-content').innerHTML = `
        <div class="metric"><span class="label">Total Goals</span><span class="value">${goals.total_goals||0}</span></div>
        ${html}${suggHtml}
    `;
}

function updateCuriosity(cur) {
    const level = cur.curiosity_level || 0;
    const signals = cur.active_signals || 0;
    const questions = cur.burning_questions || [];
    const tops = cur.top_signals || [];

    let signalHtml = tops.map(s =>
        `<div style="font-size:10px;padding:3px 0;border-bottom:1px solid var(--border)">
            <span class="tag" style="background:var(--curiosity);color:white">${s.type}</span>
            ${s.description} <span style="color:var(--warning)">${(100*s.intensity).toFixed(0)}%</span>
        </div>`
    ).join('');

    let qHtml = questions.map(q => `<div class="question-item">? ${q}</div>`).join('');

    document.getElementById('curiosity-content').innerHTML = `
        <div class="metric"><span class="label">Curiosity Level</span><span class="value" style="color:var(--curiosity)">${(100*level).toFixed(0)}%</span></div>
        <div class="bar-container"><div class="bar-fill" style="width:${100*level}%;background:var(--curiosity)"></div></div>
        <div class="metric"><span class="label">Active Signals</span><span class="value">${signals}</span></div>
        ${signalHtml}
        ${qHtml ? '<div style="margin-top:4px;font-size:10px;color:var(--curiosity)">BURNING QUESTIONS:</div>' + qHtml : ''}
    `;
}

function updateMemory(intro) {
    const mem = intro.memory || {};
    let html = '';
    for (const [scope, data] of Object.entries(mem)) {
        const util = data.utilization || 0;
        const color = util > 0.8 ? 'var(--danger)' : util > 0.5 ? 'var(--warning)' : 'var(--success)';
        html += `
            <div class="metric"><span class="label">${scope}</span>
                <span class="value">${data.count || 0} / ${data.capacity || 0}</span></div>
            <div class="bar-container"><div class="bar-fill" style="width:${100*util}%;background:${color}"></div></div>
        `;
    }
    document.getElementById('memory-content').innerHTML = html || '<span style="font-size:11px;color:var(--text-dim)">No data</span>';
}

function updateTemporal(intro) {
    const t = intro.temporal || {};
    document.getElementById('temporal-content').innerHTML = `
        <div class="metric"><span class="label">Sessions</span><span class="value">${t.total_sessions || 0}</span></div>
        <div class="metric"><span class="label">Current Turns</span><span class="value">${t.current_session_turns || 0}</span></div>
        <div class="metric"><span class="label">Typical Hour</span><span class="value">${t.typical_hour !== null ? t.typical_hour + ':00' : '—'}</span></div>
    `;
}

async function triggerDream() {
    addMsg('mnemosyne', '💤 Entering dream state...');
    const res = await fetch(API + '/api/dream', {method:'POST'});
    const data = await res.json();
    const insights = data.insights || [];
    if (insights.length) {
        addMsg('mnemosyne', '💭 Dream insights:\n' + insights.map(i =>
            `  [${i.type}] ${i.summary} (confidence: ${(100*i.confidence).toFixed(0)}%)`
        ).join('\n'));
    } else {
        addMsg('mnemosyne', '💤 Dream complete — no new insights this cycle.');
    }
    refreshAll();
}

async function introspect() {
    const res = await fetch(API + '/api/introspect');
    const data = await res.json();
    addMsg('mnemosyne', '🔍 Full introspection:\n' + JSON.stringify(data, null, 2).substring(0, 2000));
    refreshAll();
}

async function sleepSession() {
    const res = await fetch(API + '/api/sleep', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({topic: 'dashboard session'})
    });
    const data = await res.json();
    addMsg('mnemosyne', '😴 Going to sleep...\n' + JSON.stringify(data, null, 2));
    awake = false;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import asyncio as _aio

    print("\n" + "=" * 60)
    print("  🦉 MNEMOSYNE DASHBOARD")
    print("  Consciousness Loop v0.3.0")
    print("=" * 60)

    # Check available backends
    _el = asyncio.new_event_loop()
    _models = _el.run_until_complete(list_available_models())
    _el.close()

    print(f"\n  LLM Backends:")
    if _models.get("ollama"):
        print(f"    ✅ Ollama: {', '.join(_models['ollama'])}")
    else:
        print(f"    ❌ Ollama: not reachable (set OLLAMA_HOST or run `ollama serve`)")

    if _models.get("anthropic"):
        print(f"    ✅ Anthropic: {', '.join(_models['anthropic'])}")
    else:
        print(f"    ❌ Anthropic: ANTHROPIC_API_KEY not set")

    print(f"\n  Access:")
    print(f"    Local:     http://localhost:5000")
    print(f"    Tailscale: http://100.74.126.118:5000")
    print(f"\n  Binding to 0.0.0.0 for network access")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
