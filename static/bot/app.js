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
  const output = document.getElementById('playground-output');
  output.textContent = 'Loading...';

  fetch(`/api/playground/?${params.toString()}`)
    .then(async (response) => {
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || `Request failed with status ${response.status}`);
      }
      output.textContent = (data?.sentiment || '').toString().toUpperCase();
    })
    .catch((error) => {
      output.textContent = `Error: ${error.message}`;
    });
});

const playgroundDateInput = playgroundForm.querySelector('input[name="date"]');
if (playgroundDateInput && !playgroundDateInput.value) {
  const today = new Date();
  today.setDate(today.getDate() - 14);
  playgroundDateInput.value = today.toISOString().slice(0, 10);
}

const agentSessionOutput = document.getElementById('agent-session-output');
const agentCapabilitiesOutput = document.getElementById('agent-capabilities-output');
const agentProviderOutput = document.getElementById('agent-provider-output');
const agentLogsOutput = document.getElementById('agent-logs-output');
const agentProposalsContainer = document.getElementById('agent-proposals');
const agentStartButton = document.getElementById('agent-start');
const agentStopButton = document.getElementById('agent-stop');
let agentRefreshIntervalId = null;

function getCookie(name) {
  const cookieString = document.cookie || '';
  const cookies = cookieString.split(';');

  for (const cookie of cookies) {
    const trimmed = cookie.trim();
    if (trimmed.startsWith(`${name}=`)) {
      return decodeURIComponent(trimmed.slice(name.length + 1));
    }
  }

  return '';
}

async function postAgent(url) {
  const csrfToken = getCookie('csrftoken');
  const response = await fetch(url, {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'X-CSRFToken': csrfToken,
      Accept: 'application/json',
    },
  });
  const payload = await response.json();
  if (!response.ok) {
    const details = payload?.reason ? ` (${payload.reason})` : '';
    throw new Error(payload.error || `Agent request failed with status ${response.status}${details}`);
  }
  return payload;
}

function renderAgentProposals(proposals) {
  if (!agentProposalsContainer) return;
  agentProposalsContainer.innerHTML = '';
  if (!proposals.length) {
    agentProposalsContainer.innerHTML = '<div class="card">No pending decisions.</div>';
    return;
  }

  proposals.forEach((proposal) => {
    const card = document.createElement('article');
    card.className = 'card';
    const isMicro = proposal.proposalOrigin === 'micro';
    const microMetrics = proposal.microMetrics || {};
    card.innerHTML = `
      <h3>${proposal.symbol} · ${proposal.action} ${isMicro ? '<span class="badge">MICRO</span>' : ''}</h3>
      <p>Position impact: ${proposal.size}</p>
      <p>Confidence: ${proposal.confidence}</p>
      <p>Technical score: ${proposal.providerScores?.technical?.score ?? '-'}</p>
      <p>Risk score: ${proposal.providerScores?.portfolio_risk?.score ?? '-'}</p>
      <p>Sentiment score: ${proposal.providerScores?.social_sentiment?.score ?? '-'}</p>
      <p>Attention score: ${proposal.attentionSummary?.attentionScore ?? '-'}</p>
      <p>Crowding score: ${proposal.attentionSummary?.crowdingScore ?? '-'}</p>
      <p>Narrative velocity: ${proposal.attentionSummary?.narrativeVelocity ?? '-'}</p>
      <p>Watchlist/attention score: ${proposal.providerScores?.watchlist_interest?.score ?? '-'}</p>
      <p>Conflict flags: ${(proposal.conflictFlags || []).join(', ') || 'none'}</p>
      <p>Tag: ${isMicro ? 'micro' : 'macro'}</p>
      <p>TTL countdown: ${proposal.ttlRemaining ?? '-'}s</p>
      ${isMicro ? `<p>localLow: ${microMetrics.localLow ?? '-'}</p>` : ''}
      ${isMicro ? `<p>localHigh: ${microMetrics.localHigh ?? '-'}</p>` : ''}
      ${isMicro ? `<p>microRange: ${microMetrics.microRange ?? '-'}</p>` : ''}
      ${isMicro ? `<p>expectedEdge: ${microMetrics.expectedEdge ?? '-'}</p>` : ''}
      ${isMicro ? `<p>ttlRemaining: ${proposal.ttlRemaining ?? '-'}s</p>` : ''}
      <p>Why now: ${proposal.whyNow}</p>
      <p>${proposal.explanation}</p>
      <div class="inline-actions">
        <button class="approve" data-id="${proposal.proposalId}">Yes</button>
        <button class="reject secondary" data-id="${proposal.proposalId}">No</button>
      </div>
    `;
    agentProposalsContainer.appendChild(card);
  });
}

async function refreshAgentSection() {
  if (!agentSessionOutput) return;
  try {
    const [sessionResp, proposalResp, logsResp] = await Promise.all([
      fetch('/api/agent/session'),
      fetch('/api/agent/proposals'),
      fetch('/api/agent/logs'),
    ]);
    const sessionPayload = await sessionResp.json();
    const proposalPayload = await proposalResp.json();
    const logsPayload = await logsResp.json();
    agentSessionOutput.textContent = JSON.stringify(sessionPayload, null, 2);
    agentCapabilitiesOutput.textContent = JSON.stringify(sessionPayload.capabilities || {}, null, 2);
    renderAgentProposals(proposalPayload.proposals || []);
    agentProviderOutput.textContent = JSON.stringify(
      (proposalPayload.proposals || []).map((item) => ({
        symbol: item.symbol,
        providerScores: item.providerScores,
        signalFreshness: item.signalFreshness,
        sentimentSummary: item.sentimentSummary,
        invalidationReason: item.invalidationReason,
      })),
      null,
      2,
    );
    agentLogsOutput.textContent = JSON.stringify(logsPayload.recentLogs || [], null, 2);
  } catch (error) {
    agentLogsOutput.textContent = `Error: ${error.message}`;
  }
}

function stopAgentRefreshLoop() {
  if (agentRefreshIntervalId !== null) {
    window.clearInterval(agentRefreshIntervalId);
    agentRefreshIntervalId = null;
  }
}

function startAgentRefreshLoop() {
  stopAgentRefreshLoop();
  agentRefreshIntervalId = window.setInterval(refreshAgentSection, 4000);
}

if (agentStartButton) {
  agentStartButton.addEventListener('click', async () => {
    try {
      await postAgent('/api/agent/start');
      await refreshAgentSection();
      startAgentRefreshLoop();
    } catch (error) {
      agentLogsOutput.textContent = `Error: ${error.message}`;
    }
  });
}

if (agentStopButton) {
  agentStopButton.addEventListener('click', async () => {
    try {
      await postAgent('/api/agent/stop');
      stopAgentRefreshLoop();
      await refreshAgentSection();
    } catch (error) {
      agentLogsOutput.textContent = `Error: ${error.message}`;
    }
  });
}

if (agentProposalsContainer) {
  agentProposalsContainer.addEventListener('click', async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const proposalId = target.dataset.id;
    if (!proposalId) return;
    const isApprove = target.classList.contains('approve');
    const endpoint = isApprove
      ? `/api/agent/proposals/${proposalId}/approve`
      : `/api/agent/proposals/${proposalId}/reject`;
    try {
      await postAgent(endpoint);
      await refreshAgentSection();
    } catch (error) {
      agentLogsOutput.textContent = `Error: ${error.message}`;
    }
  });
}

const cryptoAgentSessionOutput = document.getElementById('crypto-agent-session-output');
const cryptoAgentLogsOutput = document.getElementById('crypto-agent-logs-output');
const cryptoAgentProposalsContainer = document.getElementById('crypto-agent-proposals');
const cryptoAgentStartButton = document.getElementById('crypto-agent-start');
const cryptoAgentStopButton = document.getElementById('crypto-agent-stop');
let cryptoAgentRefreshIntervalId = null;

function renderCryptoProposals(proposals) {
  if (!cryptoAgentProposalsContainer) return;
  cryptoAgentProposalsContainer.innerHTML = '';
  if (!proposals.length) {
    cryptoAgentProposalsContainer.innerHTML = '<div class="card">No pending decisions.</div>';
    return;
  }
  proposals.forEach((proposal) => {
    const card = document.createElement('article');
    card.className = 'card';
    card.innerHTML = `
      <h3>${proposal.symbol} · ${proposal.action}</h3>
      <p>Signal breakdown: technical=${proposal.providerScores?.technical?.score ?? '-'} / risk=${proposal.providerScores?.portfolio_risk?.score ?? '-'} / sentiment=${proposal.providerScores?.social_sentiment?.score ?? '-'}</p>
      <p>Microstructure: ${proposal.providerScores?.microstructure?.score ?? '-'}</p>
      <p>Conflict flags: ${(proposal.conflictFlags || []).join(', ') || 'none'}</p>
      <p>TTL countdown: ${proposal.ttlRemaining ?? '-'}s</p>
      <p>${proposal.explanation}</p>
      <div class="inline-actions">
        <button class="approve" data-id="${proposal.proposalId}">Yes</button>
        <button class="reject secondary" data-id="${proposal.proposalId}">No</button>
      </div>
    `;
    cryptoAgentProposalsContainer.appendChild(card);
  });
}

async function refreshCryptoAgentSection() {
  if (!cryptoAgentSessionOutput) return;
  try {
    const [sessionResp, proposalResp, logsResp] = await Promise.all([
      fetch('/api/agent-crypto/session'),
      fetch('/api/agent-crypto/proposals'),
      fetch('/api/agent-crypto/logs'),
    ]);
    const sessionPayload = await sessionResp.json();
    const proposalPayload = await proposalResp.json();
    const logsPayload = await logsResp.json();
    cryptoAgentSessionOutput.textContent = JSON.stringify(sessionPayload, null, 2);
    renderCryptoProposals(proposalPayload.proposals || []);
    cryptoAgentLogsOutput.textContent = JSON.stringify(logsPayload.recentLogs || [], null, 2);
  } catch (error) {
    cryptoAgentLogsOutput.textContent = `Error: ${error.message}`;
  }
}

function stopCryptoAgentRefreshLoop() {
  if (cryptoAgentRefreshIntervalId !== null) {
    window.clearInterval(cryptoAgentRefreshIntervalId);
    cryptoAgentRefreshIntervalId = null;
  }
}

function startCryptoAgentRefreshLoop() {
  stopCryptoAgentRefreshLoop();
  cryptoAgentRefreshIntervalId = window.setInterval(refreshCryptoAgentSection, 4000);
}

if (cryptoAgentStartButton) {
  cryptoAgentStartButton.addEventListener('click', async () => {
    await postAgent('/api/agent-crypto/start');
    await refreshCryptoAgentSection();
    startCryptoAgentRefreshLoop();
  });
}
if (cryptoAgentStopButton) {
  cryptoAgentStopButton.addEventListener('click', async () => {
    await postAgent('/api/agent-crypto/stop');
    stopCryptoAgentRefreshLoop();
    await refreshCryptoAgentSection();
  });
}
if (cryptoAgentProposalsContainer) {
  cryptoAgentProposalsContainer.addEventListener('click', async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const proposalId = target.dataset.id;
    if (!proposalId) return;
    const endpoint = target.classList.contains('approve')
      ? `/api/agent-crypto/proposals/${proposalId}/approve`
      : `/api/agent-crypto/proposals/${proposalId}/reject`;
    await postAgent(endpoint);
    await refreshCryptoAgentSection();
  });
}
