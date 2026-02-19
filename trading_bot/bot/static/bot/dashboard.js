const pretty = (data) => JSON.stringify(data, null, 2);

async function fetchAndRender(url, outputEl) {
  outputEl.classList.remove("error");
  outputEl.textContent = "Loading...";
  try {
    const response = await fetch(url);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }
    outputEl.textContent = pretty(payload);
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
  fetchAndRender(`/api/candidates/?limit=${encodeURIComponent(limit)}&risk=${encodeURIComponent(risk)}`, candidatesOutput);
});

const monitorForm = document.getElementById("monitor-form");
const monitorOutput = document.getElementById("monitor-output");
monitorForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const limit = document.getElementById("news-limit").value;
  fetchAndRender(`/api/market-monitor/?limit=${encodeURIComponent(limit)}`, monitorOutput);
});

suggestionForm.requestSubmit();
candidatesForm.requestSubmit();
monitorForm.requestSubmit();
