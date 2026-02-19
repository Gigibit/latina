const sections = [...document.querySelectorAll('.section')];
const navItems = [...document.querySelectorAll('.nav-item')];
const suggestionForm = document.getElementById('suggestion-form');
const suggestionOutput = document.getElementById('suggestion-output');
const appShell = document.querySelector('.app-shell');
const manualSymbolInputEnabled = appShell?.dataset.manualSymbolInputEnabled === 'true';

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

startSuggestionResearch('medium');
