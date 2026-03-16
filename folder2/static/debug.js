async function renderDebugState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  const pre = document.getElementById("stateJson");
  if (pre) {
    pre.textContent = JSON.stringify(data, null, 2);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const originalFetchState = window.fetchState;
  if (originalFetchState) {
    window.fetchState = async () => {
      await originalFetchState();
      await renderDebugState();
    };
  }
  renderDebugState();
});
