(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const STORAGE_API_KEY = "echolang-api-base";
  const defaultApiBase = window.location.origin && window.location.origin !== "null"
    ? window.location.origin
    : "http://127.0.0.1:8000";
  let apiBase = localStorage.getItem(STORAGE_API_KEY) || defaultApiBase;
  $("#api-base").value = apiBase;

  // ---------- state ----------
  let mode = "text";               // "text" | "record" | "upload"
  let audioBlob = null;            // Blob for record/upload modes
  let audioFilename = "recording.webm";
  let mediaRecorder = null;
  let recordedChunks = [];
  let recTimer = null;
  let recSeconds = 0;
  let log = [];

  // ---------- health check ----------
  async function checkHealth() {
    const dot = $("#api-dot");
    const text = $("#api-status-text");
    try {
      const res = await fetch(`${apiBase}/health`);
      if (!res.ok) throw new Error("bad status");
      const data = await res.json();
      dot.className = "api-status ok";
      text.textContent = "Live";
      $("#stt-model").textContent = data.stt_model || "—";
      $("#llm-model").textContent = data.llm_model || "—";
    } catch (e) {
      dot.className = "api-status bad";
      text.textContent = "Unreachable";
      $("#stt-model").textContent = "—";
      $("#llm-model").textContent = "—";
    }
  }

  $("#api-save").addEventListener("click", () => {
    apiBase = $("#api-base").value.trim().replace(/\/$/, "");
    localStorage.setItem(STORAGE_API_KEY, apiBase);
    checkHealth();
  });

  checkHealth();

  // ---------- mode tabs ----------
  $$(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".mode-tab").forEach((t) => t.classList.remove("active"));
      $$(".mode-pane").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      mode = tab.dataset.mode;
      $(`.mode-pane[data-pane="${mode}"]`).classList.add("active");
      $("#mode-meta").textContent = mode;
    });
  });

  // ---------- example chips ----------
  $$(".example-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#text-input").value = chip.dataset.ex;
    });
  });

  // ---------- upload ----------
  const uploadZone = $("#upload-zone");
  const fileInput = $("#file-input");

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleUploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleUploadFile(fileInput.files[0]);
  });

  function handleUploadFile(file) {
    audioBlob = file;
    audioFilename = file.name;
    $("#upload-filename").textContent = file.name;
  }

  // ---------- recording ----------
  const recBtn = $("#rec-btn");
  const recDot = $("#rec-dot");
  const recTimeEl = $("#rec-time");
  const recZone = $("#rec-zone");

  recZone.addEventListener("click", () => recBtn.click());

  recBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (evt) => {
        if (evt.data.size > 0) recordedChunks.push(evt.data);
      };
      mediaRecorder.onstop = () => {
        audioBlob = new Blob(recordedChunks, { type: "audio/webm" });
        audioFilename = "recording.webm";
        $("#rec-filename").textContent = `Captured ${recSeconds}s clip`;
        stream.getTracks().forEach((t) => t.stop());
      };
      mediaRecorder.start();
      recBtn.textContent = "Stop";
      recDot.classList.add("live");
      recSeconds = 0;
      recTimeEl.textContent = "00:00";
      recTimer = setInterval(() => {
        recSeconds += 1;
        const m = String(Math.floor(recSeconds / 60)).padStart(2, "0");
        const s = String(recSeconds % 60).padStart(2, "0");
        recTimeEl.textContent = `${m}:${s}`;
      }, 1000);
    } catch (err) {
      $("#rec-filename").textContent = "Mic access denied";
    }
  });

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
    clearInterval(recTimer);
    recBtn.textContent = "Start";
    recDot.classList.remove("live");
  }

  // ---------- clear ----------
  $("#clear-btn").addEventListener("click", () => {
    $("#text-input").value = "";
    audioBlob = null;
    $("#upload-filename").textContent = "No file selected";
    $("#rec-filename").textContent = "No recording yet";
    fileInput.value = "";
    showEmpty();
  });

  // ---------- analyze ----------
  $("#analyze-btn").addEventListener("click", runAnalysis);

  async function runAnalysis() {
    setLoading(true, mode === "text" ? "Contacting backend" : "Transcribing audio");
    try {
      let result;
      if (mode === "text") {
        const text = $("#text-input").value.trim();
        if (!text) {
          setLoading(false);
          return;
        }
        result = await analyzeText(text);
        renderResult(result, text);
        pushLog(result, text);
      } else {
        if (!audioBlob) {
          setLoading(false);
          return;
        }
        const processed = await processAudio(audioBlob, audioFilename);
        setLoading(true, "Classifying intent");
        renderResult(processed.analysis, processed.transcript);
        pushLog(processed.analysis, processed.transcript);
      }
    } catch (err) {
      showError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function analyzeText(text) {
    const res = await fetch(`${apiBase}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    return res.json();
  }

  async function processAudio(blob, filename) {
    const form = new FormData();
    form.append("audio", blob, filename);
    const res = await fetch(`${apiBase}/process`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    return res.json();
  }

  // ---------- render ----------
  function setLoading(active, text) {
    $("#loading-state").classList.toggle("active", active);
    $("#empty-state").classList.toggle("active", false);
    if (!active) return;
    $("#result-body").classList.remove("active");
    $("#loading-text").textContent = text || "Working";
    $("#result-meta").textContent = "running";
  }

  function showEmpty() {
    $("#result-body").classList.remove("active");
    $("#loading-state").classList.remove("active");
    $("#empty-state").classList.add("active");
    $("#empty-state").style.display = "flex";
    $("#result-meta").textContent = "idle";
  }

  function showError(msg) {
    $("#result-body").classList.remove("active");
    $("#empty-state").classList.add("active");
    $("#empty-state").style.display = "flex";
    $("#empty-state").innerHTML = `<div style="color:var(--red)">// error</div><div>${escapeHtml(msg)}</div>`;
    $("#result-meta").textContent = "error";
  }

  function renderResult(analysis, originalText) {
    $("#empty-state").classList.remove("active");
    $("#empty-state").style.display = "none";
    $("#result-body").classList.add("active");
    $("#result-meta").textContent = "done";

    $("#lang-label").textContent = analysis.detected_language || "unknown";
    $("#orig-text").textContent = originalText;
    $("#translated-text").textContent = analysis.translated_text || "";

    $("#category-badge").textContent = analysis.intent_category || "—";

    const urgency = (analysis.urgency || "low").toLowerCase();
    const uBadge = $("#urgency-badge");
    uBadge.textContent = urgency.toUpperCase();
    uBadge.className = `badge urgency-${urgency}`;

    const conf = Math.round((analysis.confidence || 0) * 100);
    $("#conf-fill").style.width = `${conf}%`;
    $("#conf-num").textContent = `${conf}%`;

    $("#urgency-reason").textContent = analysis.urgency_reason || "—";
    $("#reasoning-text").textContent = analysis.reasoning || "—";
  }

  function pushLog(analysis, originalText) {
    log.unshift({ analysis, originalText, time: new Date() });
    if (log.length > 25) log.pop();
    renderLog();
  }

  function renderLog() {
    const list = $("#log-list");
    $("#log-count").textContent = `${log.length} request${log.length === 1 ? "" : "s"}`;
    if (!log.length) {
      list.innerHTML = '<div class="log-empty">// requests you analyze this session<br>will appear here</div>';
      return;
    }
    list.innerHTML = log
      .map((entry) => {
        const t = entry.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        return `<li class="log-item">
          <div class="li-top">
            <span class="li-cat">${escapeHtml(entry.analysis.intent_category || "—")}</span>
            <span class="li-time">${t}</span>
          </div>
          <div class="li-text">${escapeHtml(entry.originalText)}</div>
        </li>`;
      })
      .join("");

    $$(".log-item").forEach((el, i) => {
      el.addEventListener("click", () => {
        renderResult(log[i].analysis, log[i].originalText);
      });
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  showEmpty();
})();
