const pretty = (data) => JSON.stringify(data, null, 2);

function formatCandidates(payload) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  if (!candidates.length) {
    return pretty(payload);
  }

  const lines = [
    `Source: ${payload.source}`,
    `Risk profile: ${payload.risk_profile}`,
    "",
    "Candidates:",
  ];

  candidates.forEach((candidate, index) => {
    lines.push(
      `${index + 1}. ${candidate.symbol} | zone: ${candidate.zone || "n/a"} | ` +
        `score: ${candidate.score} | combined: ${candidate.combined_score}`
    );
  });

  lines.push("", "Raw payload:", pretty(payload));
  return lines.join("\n");
}

async function fetchAndRender(url, outputEl, formatter = pretty) {
  outputEl.classList.remove("error");
  outputEl.textContent = "Loading...";
  try {
    const response = await fetch(url);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    outputEl.textContent = formatter(payload);
  } catch (error) {
    outputEl.classList.add("error");
    outputEl.textContent = error.message;
  }
}

const suggestionForm = document.getElementById("suggestion-form");
const suggestionOutput = document.getElementById("suggestion-output");
suggestionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const symbol = document.getElementById("symbol").value;
  const risk = document.getElementById("risk").value;
  fetchAndRender(`/api/suggestion/?symbol=${encodeURIComponent(symbol)}&risk=${encodeURIComponent(risk)}`, suggestionOutput);
});

const candidatesForm = document.getElementById("candidates-form");
const candidatesOutput = document.getElementById("candidates-output");
candidatesForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const limit = document.getElementById("limit").value;
  const risk = document.getElementById("risk-candidates").value;
  fetchAndRender(
    `/api/candidates/?limit=${encodeURIComponent(limit)}&risk=${encodeURIComponent(risk)}`,
    candidatesOutput,
    formatCandidates,
  );
});

const monitorForm = document.getElementById("monitor-form");
const monitorOutput = document.getElementById("monitor-output");
monitorForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const limit = document.getElementById("news-limit").value;
  fetchAndRender(`/api/market-monitor/?limit=${encodeURIComponent(limit)}`, monitorOutput);
});
