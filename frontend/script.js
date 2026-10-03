/**
 * ============================================
 * SentimentAI – Frontend Logic
 * ============================================
 * Handles chat interaction, API calls, statistics tracking,
 * and dynamic UI updates.
 */

const API_BASE = window.location.origin;

// ============================================
// State Management
// ============================================
const state = {
    messages: [],
    stats: { total: 0, positive: 0, neutral: 0, negative: 0 },
    isLoading: false,
    modelsLoaded: false,
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    chatMessages: document.getElementById("chatMessages"),
    messageInput: document.getElementById("messageInput"),
    btnSend: document.getElementById("btnSend"),
    btnClear: document.getElementById("btnClear"),
    btnToggleSidebar: document.getElementById("btnToggleSidebar"),
    sidebar: document.getElementById("sidebar"),
    modelSelect: document.getElementById("modelSelect"),
    welcomeScreen: document.getElementById("welcomeScreen"),
    headerStatus: document.getElementById("headerStatus"),
    // Stats
    statTotal: document.getElementById("statTotal"),
    statPositive: document.getElementById("statPositive"),
    statNeutral: document.getElementById("statNeutral"),
    statNegative: document.getElementById("statNegative"),
    // Bars
    barPositive: document.getElementById("barPositive"),
    barNeutral: document.getElementById("barNeutral"),
    barNegative: document.getElementById("barNegative"),
    pctPositive: document.getElementById("pctPositive"),
    pctNeutral: document.getElementById("pctNeutral"),
    pctNegative: document.getElementById("pctNegative"),
};

// ============================================
// Initialize
// ============================================
async function init() {
    setupEventListeners();
    await checkAPIHealth();
    await loadModels();
}

function setupEventListeners() {
    elements.btnSend.addEventListener("click", sendMessage);
    elements.btnClear.addEventListener("click", clearChat);
    elements.btnToggleSidebar.addEventListener("click", toggleSidebar);

    elements.messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    elements.messageInput.addEventListener("input", () => {
        const textarea = elements.messageInput;
        textarea.style.height = "auto";
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
    });
}

// ============================================
// API Communication
// ============================================
async function checkAPIHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();

        if (data.status === "healthy") {
            setStatus("online", `${data.models_loaded} model(s) ready`);
            state.modelsLoaded = data.models_loaded > 0;
        } else {
            setStatus("offline", "API not ready");
        }
    } catch (err) {
        setStatus("offline", "Cannot connect to API");
        console.error("Health check failed:", err);
    }
}

async function loadModels() {
    try {
        const res = await fetch(`${API_BASE}/api/models`);
        const data = await res.json();

        if (data.models && data.models.length > 0) {
            elements.modelSelect.innerHTML = '<option value="">Auto (Best)</option>';
            data.models.forEach((model) => {
                const opt = document.createElement("option");
                opt.value = model;
                opt.textContent = formatModelName(model);
                elements.modelSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Failed to load models:", err);
    }
}

async function predictSentiment(text) {
    const model = elements.modelSelect.value || undefined;
    const payload = { text };
    if (model) payload.model = model;

    const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
    }

    return await res.json();
}

// ============================================
// Chat Logic
// ============================================
async function sendMessage() {
    const text = elements.messageInput.value.trim();
    if (!text || state.isLoading) return;

    // Hide welcome screen
    if (elements.welcomeScreen) {
        elements.welcomeScreen.style.display = "none";
    }

    // Add user message
    addMessage("user", text);
    elements.messageInput.value = "";
    elements.messageInput.style.height = "auto";

    // Show typing indicator
    state.isLoading = true;
    elements.btnSend.disabled = true;
    const typingEl = showTypingIndicator();

    try {
        const response = await predictSentiment(text);
        removeTypingIndicator(typingEl);
        addBotResponse(response);
        updateStats(response.analysis.sentiment);
    } catch (err) {
        removeTypingIndicator(typingEl);
        addMessage("bot", "Sorry, I couldn't analyze that message. Please make sure the API is running and models are trained.", "error");
        console.error("Prediction error:", err);
    } finally {
        state.isLoading = false;
        elements.btnSend.disabled = false;
        elements.messageInput.focus();
    }
}

function sendQuick(text) {
    elements.messageInput.value = text;
    sendMessage();
}

// Make sendQuick accessible globally for inline onclick handlers
window.sendQuick = sendQuick;

// ============================================
// Message Rendering
// ============================================
function addMessage(type, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${type}`;

    const avatar = type === "user" ? "👤" : "🤖";
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    msgDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">${escapeHtml(text)}</div>
            <div class="message-time">${time}</div>
        </div>
    `;

    elements.chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function addBotResponse(response) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message bot";

    const sentiment = response.analysis.sentiment;
    const confidence = response.analysis.confidence;
    const sarcasm = response.analysis.sarcasm_detected;
    const botMessage = response.response.message;
    const followUp = response.response.follow_up;
    const tone = response.response.tone;
    const processingTime = response.metadata.processing_time_ms;
    const modelUsed = response.metadata.model_used;
    const pipeline = response.nlp_pipeline || {};
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const sentimentEmoji = {
        positive: "😊",
        neutral: "😐",
        negative: "😔",
    };

    const pipelineId = "pipeline_" + Date.now();

    let html = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-bubble">
                ${escapeHtml(botMessage)}
                <br><br>
                <em style="color: var(--text-muted); font-size: 0.85em;">${escapeHtml(followUp)}</em>
            </div>
            <div class="sentiment-badge ${sentiment}">
                ${sentimentEmoji[sentiment] || "🔍"} ${sentiment}
                <span style="opacity:0.7">• ${tone} tone</span>
            </div>
            <div class="confidence-bar">
                <span class="confidence-label">Confidence</span>
                <div class="confidence-track">
                    <div class="confidence-fill ${sentiment}" style="width: ${confidence * 100}%"></div>
                </div>
                <span class="confidence-value">${(confidence * 100).toFixed(1)}%</span>
            </div>`;

    if (sarcasm) {
        html += `
            <div class="sarcasm-warning">
                🙄 Sarcasm detected — sentiment adjusted accordingly
            </div>`;
    }

    // NLP Pipeline Section
    if (pipeline.steps && pipeline.steps.length > 0) {
        html += `
            <div class="nlp-pipeline-toggle" onclick="togglePipeline('${pipelineId}')">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" class="pipeline-chevron" id="chevron_${pipelineId}">
                    <path d="M4 5.5L7 8.5L10 5.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                🔬 NLP Pipeline (${pipeline.total_steps} steps applied)
            </div>
            <div class="nlp-pipeline-steps" id="${pipelineId}" style="display:none;">`;

        for (const step of pipeline.steps) {
            const iconMap = {
                "Original Input": "📝",
                "HTML Removal": "🧹",
                "URL Removal": "🔗",
                "Mention Removal": "👤",
                "Hashtag Processing": "#️⃣",
                "Emoji → Text": "😀",
                "Lowercasing": "🔡",
                "Contraction Expansion": "📖",
                "Slang Expansion": "💬",
                "Punctuation Removal": "✂️",
                "Tokenization": "🔪",
                "Stopword Removal": "🚫",
                "Lemmatization": "🌱",
                "Short Token Filter": "📏",
                "TF-IDF Vectorization": "📊",
                "Model Classification": "🤖",
            };
            const icon = iconMap[step.name] || "⚙️";

            html += `
                <div class="pipeline-step">
                    <div class="pipeline-step-header">
                        <span class="pipeline-step-num">${step.step}</span>
                        <span class="pipeline-step-icon">${icon}</span>
                        <div class="pipeline-step-info">
                            <strong>${escapeHtml(step.name)}</strong>
                            <span class="pipeline-technique">${escapeHtml(step.technique)}</span>
                        </div>
                    </div>
                    <div class="pipeline-step-desc">${escapeHtml(step.description)}</div>
                    <div class="pipeline-step-output">${escapeHtml(step.output)}</div>
                </div>`;
        }

        // Stemming vs Lemmatization comparison
        if (pipeline.stemming_vs_lemmatization && pipeline.stemming_vs_lemmatization.changes.length > 0) {
            html += `
                <div class="pipeline-comparison">
                    <div class="pipeline-comparison-title">📊 Stemming vs Lemmatization Comparison</div>
                    <div class="pipeline-comparison-table">
                        <div class="comp-row comp-header">
                            <span>Original</span><span>Lemmatized</span><span>Stemmed</span>
                        </div>`;
            const lem = pipeline.stemming_vs_lemmatization.lemmatized;
            const stem = pipeline.stemming_vs_lemmatization.stemmed;
            const maxShow = Math.min(lem.length, 8);
            for (let i = 0; i < maxShow; i++) {
                const changed = lem[i] !== stem[i];
                html += `
                        <div class="comp-row ${changed ? 'comp-diff' : ''}">
                            <span>${escapeHtml(lem[i])}</span>
                            <span class="comp-lem">${escapeHtml(lem[i])}</span>
                            <span class="comp-stem">${escapeHtml(stem[i])}</span>
                        </div>`;
            }
            html += `</div></div>`;
        }

        html += `</div>`;
    }

    html += `
            <div class="message-time">${time} • ${modelUsed} • ${processingTime}ms</div>
        </div>
    `;

    msgDiv.innerHTML = html;
    elements.chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function togglePipeline(id) {
    const el = document.getElementById(id);
    const chevron = document.getElementById("chevron_" + id);
    if (el.style.display === "none") {
        el.style.display = "block";
        if (chevron) chevron.style.transform = "rotate(180deg)";
    } else {
        el.style.display = "none";
        if (chevron) chevron.style.transform = "rotate(0deg)";
    }
    scrollToBottom();
}

function showTypingIndicator() {
    const div = document.createElement("div");
    div.className = "message bot";
    div.id = "typingIndicator";
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    elements.chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function removeTypingIndicator(el) {
    if (el && el.parentNode) {
        el.parentNode.removeChild(el);
    }
}

// ============================================
// Statistics
// ============================================
function updateStats(sentiment) {
    state.stats.total++;
    if (sentiment === "positive") state.stats.positive++;
    else if (sentiment === "neutral") state.stats.neutral++;
    else if (sentiment === "negative") state.stats.negative++;

    // Update stat cards
    elements.statTotal.textContent = state.stats.total;
    elements.statPositive.textContent = state.stats.positive;
    elements.statNeutral.textContent = state.stats.neutral;
    elements.statNegative.textContent = state.stats.negative;

    // Update bars
    const total = state.stats.total || 1;
    const pPos = ((state.stats.positive / total) * 100).toFixed(0);
    const pNeu = ((state.stats.neutral / total) * 100).toFixed(0);
    const pNeg = ((state.stats.negative / total) * 100).toFixed(0);

    elements.barPositive.style.width = pPos + "%";
    elements.barNeutral.style.width = pNeu + "%";
    elements.barNegative.style.width = pNeg + "%";

    elements.pctPositive.textContent = pPos + "%";
    elements.pctNeutral.textContent = pNeu + "%";
    elements.pctNegative.textContent = pNeg + "%";
}

// ============================================
// UI Helpers
// ============================================
function setStatus(status, text) {
    const statusDot = elements.headerStatus.querySelector(".status-dot");
    if (status === "online") {
        statusDot.classList.add("online");
    } else {
        statusDot.classList.remove("online");
    }
    elements.headerStatus.innerHTML = `<span class="status-dot ${status === 'online' ? 'online' : ''}"></span>${text}`;
}

function toggleSidebar() {
    elements.sidebar.classList.toggle("collapsed");
    elements.sidebar.classList.toggle("show");
}

function clearChat() {
    // Remove all messages except welcome
    const messages = elements.chatMessages.querySelectorAll(".message");
    messages.forEach((m) => m.remove());

    // Reset stats
    state.stats = { total: 0, positive: 0, neutral: 0, negative: 0 };
    updateStats("none");
    state.stats.total = 0; // Fix the increment from updateStats

    // Show welcome screen
    if (elements.welcomeScreen) {
        elements.welcomeScreen.style.display = "flex";
    }
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    });
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatModelName(name) {
    return name
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ============================================
// Start
// ============================================
document.addEventListener("DOMContentLoaded", init);
