const sections = [...document.querySelectorAll('.section')];
const navItems = [...document.querySelectorAll('.nav-item')];

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

document.getElementById('suggestion-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const params = new URLSearchParams(new FormData(event.target));
  callApi(`/api/suggestion/?${params.toString()}`, 'suggestion-output');
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

callApi('/api/suggestion/?symbol=AAPL&risk=medium', 'suggestion-output');
