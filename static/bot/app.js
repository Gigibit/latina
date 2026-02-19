const sections = [...document.querySelectorAll('.section')];
const navItems = [...document.querySelectorAll('.nav-item')];
const suggestionForm = document.getElementById('suggestion-form');
const suggestionOutput = document.getElementById('suggestion-output');

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

async function callApi(url, outputId) {
  const output = document.getElementById(outputId);
  output.textContent = 'Loading...';
  try {
    const response = await fetch(url);
    const data = await response.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    output.textContent = `Error: ${error.message}`;
  }
}

async function startSuggestionResearch(riskValue) {
  const params = new URLSearchParams({ risk: riskValue, async: 'true' });
  let payload;

  try {
    const response = await fetch(`/api/suggestion/?${params.toString()}`);
    payload = await response.json();
  } catch (error) {
    suggestionOutput.textContent = `Error: ${error.message}`;
    return;
  }

  if (!payload.session_id) {
    suggestionOutput.textContent = JSON.stringify(payload, null, 2);
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

      suggestionOutput.textContent = JSON.stringify(statusPayload, null, 2);
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
  const risk = formData.get('risk') || 'medium';
  suggestionOutput.textContent = 'Research started. Stream log will update automatically...';
  startSuggestionResearch(String(risk));
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
