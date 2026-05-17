const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchDataSummary() {
  const res = await fetch(`${API_BASE}/data/summary`);
  return res.json();
}

export async function fetchCategories() {
  const res = await fetch(`${API_BASE}/data/categories`);
  return res.json();
}

export async function fetchHourlyTraffic() {
  const res = await fetch(`${API_BASE}/data/hourly-traffic`);
  return res.json();
}

export async function fetchDimmingSchedule() {
  const res = await fetch(`${API_BASE}/data/dimming-schedule`);
  return res.json();
}

export async function fetchWeekdayWeekend() {
  const res = await fetch(`${API_BASE}/data/weekday-weekend`);
  return res.json();
}

export async function fetchSensors() {
  const res = await fetch(`${API_BASE}/data/sensors`);
  return res.json();
}

export async function createDesign(query, useLlm = true, conversationId = null) {
  const res = await fetch(`${API_BASE}/design`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      use_llm: useLlm,
      conversation_id: conversationId,
    }),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

// Conversation API

export async function fetchConversations() {
  const res = await fetch(`${API_BASE}/conversations`);
  return res.json();
}

export async function fetchConversation(id) {
  const res = await fetch(`${API_BASE}/conversations/${id}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function deleteConversation(id) {
  const res = await fetch(`${API_BASE}/conversations/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
