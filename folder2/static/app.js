let appState = {
  session: null,
  current_node: null,
  nodes: [],
  messages: [],
  attachments: [],
};

let currentPhase = "discovery";
let learningMode = false;
let network = null;
let selectedNodeId = null;
let lastSyncedNodeId = null;
let autoRunTimer = null;
let lastRunCode = "";

const discoverySuggestions = [
  {
    text: "I have already a good sense of emotions I want to express, help me implement.",
    phase: "implementation",
  },
  { text: "The setting was...", phase: null },
  { text: "2-3 emojis that fit: ...", phase: null },
  { text: "A song that matches this feeling is...", phase: null },
];

const implementationSuggestions = [
  { text: "I want to refine/explore my emotions more.", phase: "discovery" },
  { text: "Make the motion slower and more floating.", phase: null },
  { text: "Use a softer color palette with more negative space.", phase: null },
  { text: "Can we try a different visual metaphor?", phase: null },
];

function $(id) {
  return document.getElementById(id);
}

async function fetchState() {
  const res = await fetch("/api/state");
  appState = await res.json();
  currentPhase = appState.session?.phase || "discovery";
  renderPhase();
  renderMessages();
  renderProfiles();
  renderGraph();
}

function renderPhase() {
  $("phaseBadge").textContent = `Phase: ${capitalize(currentPhase)}`;
  renderSuggestions();
}

function renderSuggestions() {
  const container = $("suggestions");
  container.innerHTML = "";
  const list = currentPhase === "discovery" ? discoverySuggestions : implementationSuggestions;
  list.forEach((item) => {
    const btn = document.createElement("button");
    btn.className = "suggestion";
    btn.textContent = item.text;
    btn.onclick = async () => {
      if (item.phase) {
        await setPhase(item.phase);
      }
      $("chatInput").value = item.text;
      $("chatInput").focus();
    };
    container.appendChild(btn);
  });
}

function renderMessages() {
  const container = $("chatMessages");
  container.innerHTML = "";
  appState.messages.forEach((msg) => {
    const bubble = document.createElement("div");
    bubble.className = `message ${msg.role}`;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = msg.role === "user" ? "You" : "Assistant";
    const content = document.createElement("div");
    const rawText = msg.content || "";
    const displayText =
      msg.role === "assistant" ? cleanAssistantDisplay(rawText) : rawText;
    content.textContent = displayText;
    bubble.appendChild(meta);
    bubble.appendChild(content);

    const attachments = parseAttachments(msg.attachments_json);
    if (attachments.length) {
      const wrap = document.createElement("div");
      wrap.style.marginTop = "8px";
      attachments.forEach((att) => {
        const item = document.createElement("div");
        item.style.marginTop = "6px";
        if (att.kind === "image") {
          const img = document.createElement("img");
          img.src = att.web_path;
          img.style.maxWidth = "180px";
          img.style.borderRadius = "10px";
          item.appendChild(img);
        } else if (att.kind === "audio") {
          const audio = document.createElement("audio");
          audio.src = att.web_path;
          audio.controls = true;
          item.appendChild(audio);
        } else {
          const link = document.createElement("a");
          link.href = att.web_path;
          link.textContent = att.filename;
          link.target = "_blank";
          item.appendChild(link);
        }
        wrap.appendChild(item);
      });
      bubble.appendChild(wrap);
    }

    container.appendChild(bubble);
  });
  container.scrollTop = container.scrollHeight;
}

function cleanAssistantDisplay(text) {
  if (!text) return "";
  let cleaned = text;
  // Strip code blocks
  cleaned = cleaned.replace(/```[\s\S]*?```/g, "").trim();
  // Strip structured sections (profiles/code) from display
  const sectionIndex = cleaned.search(
    /===\s*(EMOTION PROFILE|ARTISTIC PROFILE|P5JS CODE)\s*===/i
  );
  if (sectionIndex >= 0) {
    cleaned = cleaned.slice(0, sectionIndex).trim();
  }
  // Remove trailing confidence lines if they remain without header
  cleaned = cleaned.replace(/confidence level\s*:\s*.+$/i, "").trim();
  return cleaned || "Updated state in the side panels.";
}

function parseAttachments(raw) {
  if (!raw) return [];
  let ids = [];
  try {
    ids = JSON.parse(raw);
  } catch (e) {
    return [];
  }
  return ids
    .map((id) => {
      const att = appState.attachments.find((a) => a.id === id);
      if (!att) return null;
      return {
        id: att.id,
        filename: att.filename,
        kind: att.kind,
        web_path: att.path ? `/uploads/${att.path.split("/").pop()}` : "",
      };
    })
    .filter(Boolean);
}

function renderProfiles() {
  const node = appState.current_node || {};
  $("emotionProfile").value = node.emotion_profile || "";
  $("artisticProfile").value = node.artistic_profile || "";
  const code = node.code || "";
  $("codeEditor").value = code;
  lastSyncedNodeId = node.id || null;
  scheduleAutoRun("state-sync");
}

function renderGraph() {
  const graphContainer = $("historyGraph");
  if (!graphContainer) return;

  const nodes = new vis.DataSet(
    appState.nodes.map((node) => {
      const isCurrent = appState.current_node && node.id === appState.current_node.id;
      const hasConflict = node.conflict_json && node.conflict_json !== "null";
      const title = buildNodeTitle(node);
      return {
        id: node.id,
        label: getNodeLabel(node),
        title,
        color: {
          background: isCurrent ? "#ff6b35" : hasConflict ? "#ffe0a3" : "#fdf6ef",
          border: isCurrent ? "#c4552a" : hasConflict ? "#c9931a" : "#c9b9a8",
          highlight: {
            background: "#f5b82e",
            border: "#c9931a",
          },
        },
        font: {
          color: isCurrent ? "#fff" : "#1e1b16",
          size: 12,
        },
        shape: "box",
        margin: 10,
      };
    })
  );

  const edges = new vis.DataSet(
    (appState.nodes.length ? appState.nodes : []).flatMap((node) => {
      const list = [];
      if (node.parent_id) {
        list.push({ from: node.parent_id, to: node.id, arrows: "to" });
      }
      if (node.merge_parent_id) {
        list.push({ from: node.merge_parent_id, to: node.id, arrows: "to", dashes: true });
      }
      return list;
    })
  );

  const options = {
    layout: { hierarchical: { direction: "LR", sortMethod: "directed", nodeSpacing: 140 } },
    physics: false,
    interaction: { hover: true },
  };

  if (!network) {
    network = new vis.Network(graphContainer, { nodes, edges }, options);
    network.on("selectNode", (params) => {
      selectedNodeId = params.nodes[0];
      const node = appState.nodes.find((n) => n.id === selectedNodeId);
      showNodeDetails(node);
    });
  } else {
    network.setData({ nodes, edges });
  }
}

function buildNodeTitle(node) {
  const summary = (node.summary || node.label || "").slice(0, 160);
  const emotion = (node.emotion_profile || "").slice(0, 120);
  const artistic = (node.artistic_profile || "").slice(0, 120);
  const code = (node.code || "").split("\n")[0] || "";
  const conflict = node.conflict_json && node.conflict_json !== "null" ? "Conflict flagged" : "No conflict";
  return `Summary: ${summary || "(empty)"}\nEmotion: ${emotion || "(empty)"}\nArtistic: ${artistic || "(empty)"}\nCode: ${code || "(empty)"}\n${conflict}`;
}

function showNodeDetails(node) {
  const box = $("nodeDetails");
  if (!node) {
    box.textContent = "Select a version to see details.";
    $("revertNode").disabled = true;
    $("setCurrent").disabled = true;
    return;
  }
  const validCode = isLikelyCompleteCode(node.code || "");
  const validityNote = validCode ? "Code: valid" : "Code: incomplete";
  box.textContent =
    `Summary: ${(node.summary || node.label || node.id).slice(0, 200)}\n` +
    `Emotion: ${(node.emotion_profile || "").slice(0, 180)}\n` +
    `Artistic: ${(node.artistic_profile || "").slice(0, 180)}\n` +
    `Code: ${(node.code || "").split("\n")[0] || ""}\n` +
    `${validityNote}\n` +
    `Conflict: ${node.conflict_json && node.conflict_json !== "null" ? "Flagged" : "None"}`;
  $("revertNode").disabled = false;
  $("setCurrent").disabled = false;
}

function isGenericLabel(label) {
  const clean = (label || "").trim().toLowerCase();
  if (!clean) return true;
  if (clean.startsWith("branch from")) return true;
  return ["assistant update", "snapshot", "manual snapshot", "profile update", "root"].includes(clean);
}

function getNodeLabel(node) {
  const label = (node.label || "").trim();
  const summary = (node.summary || "").trim();
  let chosen = label;
  if (!label || isGenericLabel(label)) {
    chosen = summary || label || "";
  }
  if (!chosen) {
    return node.id ? node.id.slice(0, 6) : "node";
  }
  return chosen.length > 42 ? `${chosen.slice(0, 39)}...` : chosen;
}

async function sendMessage() {
  const input = $("chatInput");
  const text = input.value.trim();
  if (!text) return;

  const files = $("fileInput").files;
  const attachmentIds = [];
  if (files && files.length) {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/attachments", { method: "POST", body: form });
      const data = await res.json();
      if (data.id) attachmentIds.push(data.id);
    }
  }

  input.value = "";
  $("fileInput").value = "";

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: text,
      phase: currentPhase,
      learning_mode: learningMode,
      attachment_ids: attachmentIds,
    }),
  });

  const data = await res.json();
  if (data.reply) {
    await fetchState();
  }
}

async function setPhase(phase) {
  currentPhase = phase;
  await fetch("/api/phase", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phase }),
  });
  renderPhase();
}

async function saveProfiles() {
  const payload = {
    emotion_profile: $("emotionProfile").value,
    artistic_profile: $("artisticProfile").value,
    code: $("codeEditor").value,
    label: "profile update",
    note: "User updated profiles",
    source: "user",
  };
  await fetch("/api/node", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await fetchState();
}

async function saveSnapshot() {
  const payload = {
    emotion_profile: $("emotionProfile").value,
    artistic_profile: $("artisticProfile").value,
    code: $("codeEditor").value,
    label: "snapshot",
    note: "Manual snapshot",
    source: "user",
  };
  await fetch("/api/node", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await fetchState();
}

async function branchFromCurrent() {
  await saveSnapshot();
}

async function revertToSelected() {
  if (!selectedNodeId) return;
  await fetch("/api/revert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_node_id: selectedNodeId }),
  });
  await fetchState();
}

async function setCurrentSelected() {
  if (!selectedNodeId) return;
  await fetch("/api/set-current", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: selectedNodeId }),
  });
  await fetchState();
  scheduleAutoRun("state-sync");
}

async function newSession() {
  await fetch("/api/new-session", { method: "POST" });
  selectedNodeId = null;
  await fetchState();
}

function setNodeDetailsMessage(message) {
  const box = $("nodeDetails");
  if (box) {
    box.textContent = message;
  }
}

function runSketch() {
  // Ensure editor is synced to the current node before running
  if (appState.current_node && lastSyncedNodeId !== appState.current_node.id) {
    $("codeEditor").value = appState.current_node.code || "";
    lastSyncedNodeId = appState.current_node.id;
    setNodeDetailsMessage("Synced code to the selected node before running.");
  }

  const code = $("codeEditor").value || "";
  if (!code.trim()) {
    setNodeDetailsMessage("No code to run yet. Ask the assistant for a sketch or paste code.");
    return;
  }

  const normalized = normalizeCode(code);
  try {
    new Function(normalized);
  } catch (err) {
    setNodeDetailsMessage(`Syntax error: ${err.message}`);
    return;
  }

  lastRunCode = code;
  const html = buildSketchHtml(normalized);
  replacePreview(html);
}

function buildSketchHtml(code) {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
  <style>body{margin:0; background:#111;} canvas{display:block;}</style>
  <script>
    window.addEventListener('error', function(e) {
      parent.postMessage({ type: 'p5-error', message: e.message || 'Unknown error', stack: e.error && e.error.stack }, '*');
    });
    window.addEventListener('unhandledrejection', function(e) {
      parent.postMessage({ type: 'p5-error', message: e.reason ? e.reason.toString() : 'Unhandled promise rejection' }, '*');
    });
    window.addEventListener('load', function() {
      setTimeout(function() {
        if (!window.p5) {
          parent.postMessage({ type: 'p5-error', message: 'p5.js failed to load (CDN blocked or offline).' }, '*');
        }
      }, 500);
    });
  </script>
</head>
<body>
  <script>
  ${code}
  <\/script>
</body>
</html>`;
}

function replacePreview(html) {
  const oldFrame = $("preview");
  if (!oldFrame || !oldFrame.parentElement) return;
  const frame = document.createElement("iframe");
  frame.id = "preview";
  frame.className = oldFrame.className || "preview-frame";
  frame.title = oldFrame.title || "p5 preview";
  frame.srcdoc = html;
  oldFrame.parentElement.replaceChild(frame, oldFrame);
}

function normalizeCode(code) {
  // Convert let/const to var to avoid redeclare errors from LLM output
  return code.replace(/^\s*(let|const)\s+/gm, "var ");
}

function scheduleAutoRun(reason) {
  const editor = $("codeEditor");
  if (!editor) return;
  const code = editor.value || "";
  if (!code.trim()) return;
  if (code === lastRunCode && reason !== "force") return;
  if (autoRunTimer) clearTimeout(autoRunTimer);
  autoRunTimer = setTimeout(() => {
    runSketch();
  }, 400);
}

function isLikelyCompleteCode(code) {
  if (!code || !code.trim()) return false;
  return /function\s+setup|function\s+draw|new\s+p5/i.test(code);
}

function findLastValidCode() {
  if (!appState.nodes) return "";
  for (let i = appState.nodes.length - 1; i >= 0; i--) {
    const code = appState.nodes[i].code || "";
    if (code.trim()) return code;
  }
  return "";
}

function useLatestCode() {
  const current = appState.current_node && appState.current_node.code;
  const fallback = findLastValidCode();
  const code = current && current.trim() ? current : fallback;
  if (code) {
    $("codeEditor").value = code;
    runSketch();
  }
}

function toggleLearning() {
  learningMode = !learningMode;
  $("toggleLearning").textContent = `Learning Mode: ${learningMode ? "On" : "Off"}`;
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// Event bindings
window.addEventListener("DOMContentLoaded", () => {
  fetchState();
  $("sendMessage").onclick = sendMessage;
  $("togglePhase").onclick = async () => {
    const next = currentPhase === "discovery" ? "implementation" : "discovery";
    await setPhase(next);
  };
  $("toggleLearning").onclick = toggleLearning;
  $("saveProfiles").onclick = saveProfiles;
  $("saveSnapshot").onclick = saveSnapshot;
  $("branchFromCurrent").onclick = branchFromCurrent;
  $("runSketch").onclick = runSketch;
  $("useLatestCode").onclick = useLatestCode;
  $("refreshState").onclick = fetchState;
  $("revertNode").onclick = revertToSelected;
  $("setCurrent").onclick = setCurrentSelected;
  $("newSession").onclick = newSession;
  $("codeEditor").addEventListener("input", () => scheduleAutoRun("typing"));
});

window.addEventListener("message", (event) => {
  if (!event.data || event.data.type !== "p5-error") return;
  const box = $("nodeDetails");
  if (!box) return;
  const details = event.data.stack ? `\n${event.data.stack}` : "";
  box.textContent = `Sketch error: ${event.data.message}${details}`;
});
