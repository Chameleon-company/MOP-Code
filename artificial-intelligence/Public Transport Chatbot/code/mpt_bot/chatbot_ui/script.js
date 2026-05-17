// ====== Config ======
const RASA_URL = "http://localhost:5005/webhooks/rest/webhook";
const sessionId = localStorage.getItem("sessionId") || crypto.randomUUID();
localStorage.setItem("sessionId", sessionId);

// ====== Map (Leaflet + OpenStreetMap) ======
let map;
let markers = [];
let polylines = [];

function initMap() {
  map = L.map('map').setView([-37.8136, 144.9631], 13);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
}

// Predefined icons
const stationIcon = L.icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const parkingIcon = L.icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

function clearMap() {
  markers.forEach(m => map.removeLayer(m));
  polylines.forEach(p => map.removeLayer(p));
  markers = [];
  polylines = [];
}

// Choose icon by label: S = station, numbers = parking
function addMarker({ lat, lng, title, label }) {
  const icon = (label === "S") ? stationIcon : parkingIcon;
  const marker = L.marker([lat, lng], { icon }).addTo(map);
  if (title) marker.bindPopup(title);
  marker.on('click', () => sendMessage(`Plan trip from ${title || 'this location'}`));
  markers.push(marker);
  return marker;
}

function fitToMarkers() {
  if (!markers.length) return;
  const group = L.featureGroup(markers);
  map.fitBounds(group.getBounds().pad(0.2));
}

function addPolyline(points) {
  const line = L.polyline(points.map(p => [p.lat, p.lng]), {
    weight: 4, color: "red"
  }).addTo(map);
  polylines.push(line);
  return line;
}

// ====== Chat UI ======
const chatPanel = document.getElementById("chatPanel");
const chatFab   = document.getElementById("chatFab");
const closeChat = document.getElementById("closeChat");
const chatbox   = document.getElementById("chatbox");
const inputEl   = document.getElementById("userInput");
const sendBtn   = document.getElementById("sendBtn");

chatFab.onclick = () => chatPanel.classList.add("open");
closeChat.onclick = () => chatPanel.classList.remove("open");

function escapeHTML(s){
  return s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}
function render(text){
  let t = escapeHTML(text);
  t = t.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  t = t.replace(/(https?:\/\/[^\s)]+)(?=\s|$)/g,'<a href="$1" target="_blank" rel="noopener">$1</a>');
  return t.replace(/\n/g,"<br>");
}
function addMsg(role, html){
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const b = document.createElement("div");
  b.className = "bubble";
  b.innerHTML = html;
  row.appendChild(b);
  chatbox.appendChild(row);
  chatbox.scrollTop = chatbox.scrollHeight;
}

// Handle Rasa custom payloads for map
function handleCustom(custom){
  if (!custom || !custom.map) return;
  const { clear, markers: ms = [], polyline = [] } = custom.map;
  if (clear) clearMap();
  ms.forEach(m => addMarker(m));
  if (polyline.length) addPolyline(polyline);
  fitToMarkers();
}

// ====== Make "View on Map" links focus the map ======
function focusMapTo(lat, lng, title) {
  const marker = addMarker({ lat, lng, title, label: "X" });
  map.setView([lat, lng], 16);
  if (marker && marker.bindPopup && title) {
    marker.bindPopup(title).openPopup();
  }
}

chatbox.addEventListener("click", (e) => {
  const a = e.target.closest("a");
  if (!a) return;
  if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;

  const href = a.href;
  const m =
    href.match(/[?&]query=([-0-9.]+),([-0-9.]+)/) ||
    href.match(/[?&]q=([-0-9.]+),([-0-9.]+)/);

  if (m) {
    e.preventDefault();
    const lat = parseFloat(m[1]);
    const lng = parseFloat(m[2]);
    const title = a.textContent.trim() || `${lat}, ${lng}`;
    focusMapTo(lat, lng, title);
  }
});

// ====== Send flow ======
async function sendMessage(textFromUI){
  const message = (textFromUI ?? inputEl.value).trim();
  if (!message) return;

  addMsg("you", render(message));
  inputEl.value = ""; inputEl.focus();

  try {
    sendBtn.disabled = true;
    const res = await fetch(RASA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sender: sessionId, message })
    });
    const data = await res.json();

    if (Array.isArray(data) && data.length){
      data.forEach(d => {
        if (d.text) addMsg("bot", render(d.text));
        if (d.custom) handleCustom(d.custom);
      });
    } else {
      addMsg("bot", "🤔 No response.");
    }
  } catch (e) {
    console.error(e);
    addMsg("bot", "❌ Could not reach the server.");
  } finally {
    sendBtn.disabled = false;
  }
}
sendBtn.onclick = () => sendMessage();
inputEl.addEventListener("keydown", (e)=>{
  if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); sendMessage(); }
});

// Init map
window.addEventListener('load', initMap);
