const sections = [...document.querySelectorAll('.section')];
const navItems = [...document.querySelectorAll('.nav-item')];
const suggestionForm = document.getElementById('suggestion-form');
const suggestionOutput = document.getElementById('suggestion-output');
const playgroundForm = document.getElementById('playground-form');
const appShell = document.querySelector('.app-shell');
const manualSymbolInputEnabled = appShell?.dataset.manualSymbolInputEnabled === 'true';
const feedbackModal = document.querySelector('[data-feedback-modal]');

function closeFeedbackModal() {
  if (!feedbackModal) {
    return;
  }

  if (typeof feedbackModal.close === 'function') {
    feedbackModal.close();
    return;
  }

  feedbackModal.classList.remove('is-open');
  feedbackModal.setAttribute('aria-hidden', 'true');
}

if (feedbackModal) {
  feedbackModal.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const feedbackButton = target.closest('[data-feedback-value]');
    if (feedbackButton) {
      closeFeedbackModal();
      return;
    }

    if (target.closest('[data-modal-close]')) {
      closeFeedbackModal();
    }
  });
}

function showSection(target) {
  sections.forEach((section) => {
    section.classList.toggle('is-active', section.dataset.section === target);
  });
  navItems.forEach((item) => {
    item.classList.toggle('is-active', item.dataset.target === target);
  });
}

navItems.forEach((item) => {
  item.addEventListener('click', () => showSection(item.dataset.target));
});

function renderPayload(output, data) {
  const readableSummary = data?.result?.readable_summary || data?.readable_summary;
  if (!readableSummary) {
    output.textContent = JSON.stringify(data, null, 2);
    return;
  }

  output.innerHTML = '';
  const summaryBlock = document.createElement('div');
  summaryBlock.textContent = readableSummary;

  const details = document.createElement('details');
  details.open = false;
  const summary = document.createElement('summary');
  summary.textContent = 'Raw JSON';
  const jsonBlock = document.createElement('pre');
  jsonBlock.textContent = JSON.stringify(data, null, 2);

  details.appendChild(summary);
  details.appendChild(jsonBlock);
  output.appendChild(summaryBlock);
  output.appendChild(details);
}

async function callApi(url, outputId) {
  const output = document.getElementById(outputId);
  output.textContent = 'Loading...';
  try {
    const response = await fetch(url);
    const data = await response.json();
    renderPayload(output, data);
  } catch (error) {
    output.textContent = `Error: ${error.message}`;
  }
}

async function startSuggestionResearch(riskValue, symbolsValue = '') {
  const params = new URLSearchParams({ risk: riskValue, async: 'true' });
  if (symbolsValue.trim()) {
    params.set('symbols', symbolsValue);
  }
  let payload;

  try {
    const response = await fetch(`/api/suggestion/?${params.toString()}`);
    payload = await response.json();
  } catch (error) {
    suggestionOutput.textContent = `Error: ${error.message}`;
    return;
  }

  if (!payload.session_id) {
    renderPayload(suggestionOutput, payload);
    return;
  }

  let isDone = false;
  while (!isDone) {
    try {
      const statusParams = new URLSearchParams({
        async: 'true',
        status: 'true',
        session_id: payload.session_id,
      });
      const response = await fetch(`/api/suggestion/?${statusParams.toString()}`);
      const statusPayload = await response.json();

      renderPayload(suggestionOutput, statusPayload);
      isDone = ['completed', 'failed'].includes(statusPayload.status);
    } catch (error) {
      suggestionOutput.textContent = `Error: ${error.message}`;
      isDone = true;
    }

    if (!isDone) {
      await new Promise((resolve) => {
        window.setTimeout(resolve, 1200);
      });
    }
  }
}

function renderProjectionChart(payload) {
  const chart = document.getElementById('projections-chart');
  const projectionPayload = payload?.result || payload || {};
  const historical = (projectionPayload.historical_granularity_closes || [])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  const predicted = (projectionPayload.predicted_granularity_closes || [])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  const allValues = [...historical, ...predicted];

  if (!chart || allValues.length < 2) {
    if (chart) {
      chart.innerHTML = '<text x="20" y="40">Not enough data for chart.</text>';
    }
    return;
  }

  const width = 1000;
  const height = 320;
  const paddingX = 40;
  const paddingY = 30;
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = Math.max(maxValue - minValue, 0.0001);

  const totalPoints = allValues.length;
  const xFor = (index) => {
    if (totalPoints <= 1) {
      return width / 2;
    }
    return paddingX + ((width - paddingX * 2) * index) / (totalPoints - 1);
  };
  const yFor = (value) => {
    const normalized = (value - minValue) / range;
    return height - paddingY - normalized * (height - paddingY * 2);
  };

  const buildPath = (points) => points.map((point, index) => `${index === 0 ? 'M' : 'L'}${point[0].toFixed(2)} ${point[1].toFixed(2)}`).join(' ');

  const historicalPoints = historical.map((value, index) => [xFor(index), yFor(value)]);
  const predictedPoints = predicted.map((value, index) => [xFor(historical.length - 1 + index), yFor(value)]);

  const historicalPath = buildPath(historicalPoints);
  const predictedPath = buildPath(predictedPoints);
  const separatorX = xFor(Math.max(historical.length - 1, 0));

  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
  chart.innerHTML = `
    <line class="axis" x1="${paddingX}" y1="${height - paddingY}" x2="${width - paddingX}" y2="${height - paddingY}" />
    <line class="axis" x1="${paddingX}" y1="${paddingY}" x2="${paddingX}" y2="${height - paddingY}" />
    <path class="historical-line" d="${historicalPath}" />
    <path class="predicted-line" d="${predictedPath}" />
    <line class="separator" x1="${separatorX}" y1="${paddingY}" x2="${separatorX}" y2="${height - paddingY}" />
    <text x="${paddingX + 4}" y="${paddingY + 16}">Max ${maxValue.toFixed(2)}</text>
    <text x="${paddingX + 4}" y="${height - paddingY - 8}">Min ${minValue.toFixed(2)}</text>
    <text x="${Math.max(separatorX - 120, paddingX + 4)}" y="${paddingY + 36}">Next candle projection</text>
  `;
}

async function loadProjection(params) {
  const output = document.getElementById('projections-output');
  const candidateLabel = document.getElementById('projection-candidate-name');

  output.textContent = 'Loading...';
  if (candidateLabel) {
    candidateLabel.textContent = 'Best candidate: Loading...';
  }

  try {
    const response = await fetch(`/api/projections/?${params.toString()}`);
    const data = await response.json();
    const projectionPayload = data?.result || data;
    renderPayload(output, data);

    if (candidateLabel) {
      candidateLabel.textContent = `Best candidate: ${projectionPayload.candidate_name || '-'}`;
    }

    renderProjectionChart(projectionPayload);
  } catch (error) {
    output.textContent = `Error: ${error.message}`;
    if (candidateLabel) {
      candidateLabel.textContent = 'Best candidate: -';
    }
  }
}

suggestionForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const formData = new FormData(event.target);
  const risk = String(formData.get('risk') || 'medium');
  const symbols = String(formData.get('symbols') || '').trim();

  if (manualSymbolInputEnabled && !symbols) {
    suggestionOutput.textContent = 'Please provide at least one symbol (comma separated).';
    return;
  }

  suggestionOutput.textContent = 'Research started. Stream log will update automatically...';
  startSuggestionResearch(risk, symbols);
});

document.getElementById('candidates-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const params = new URLSearchParams(new FormData(event.target));
  callApi(`/api/candidates/?${params.toString()}`, 'candidates-output');
});

document.getElementById('monitor-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const params = new URLSearchParams(new FormData(event.target));
  callApi(`/api/market-monitor/?${params.toString()}`, 'monitor-output');
});

document.getElementById('projections-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const params = new URLSearchParams(new FormData(event.target));
  loadProjection(params);
});

playgroundForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const params = new URLSearchParams(new FormData(event.target));
  callApi(`/api/playground/?${params.toString()}`, 'playground-output');
});

const playgroundDateInput = playgroundForm.querySelector('input[name="date"]');
if (playgroundDateInput && !playgroundDateInput.value) {
  const today = new Date();
  today.setDate(today.getDate() - 14);
  playgroundDateInput.value = today.toISOString().slice(0, 10);
}

startSuggestionResearch('medium');
document.getElementById('projections-form').requestSubmit();
playgroundForm.requestSubmit();
