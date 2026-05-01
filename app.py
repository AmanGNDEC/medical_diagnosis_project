"""
app.py  ── STEP 4 & 5 : Browser-Based Desktop UI
==================================================
B.Tech Mini-Project: AI-Powered Medical Diagnosis Assistant

WHY NO TKINTER:
  The CommandLineTools Python 3.9 ships with system Tk 8.5 which crashes
  on modern macOS (TkpInit → Tcl_Panic → abort).  Instead we use Python's
  built-in http.server — zero extra deps, works on every OS, and the UI
  can be far richer using HTML/CSS/JS.

HOW IT WORKS:
  1. Starts a local HTTP server on port 5050.
  2. Opens http://localhost:5050 in your default browser.
  3. The HTML page sends symptom selections to /diagnose (POST → JSON).
  4. The server runs Naive Bayes + XAI, returns results as JSON.
  5. Matplotlib charts are sent as base64 PNG strings embedded in JSON.
"""

import io
import os
import json
import base64
import threading
import webbrowser
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

# ── Project modules ──────────────────────────────────────────────────────────
from model    import get_model_and_symptoms, predict_top_diseases
from xai      import get_top_contributions
from data_prep import get_prepared_data

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — no Tk needed at all
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ════════════════════════════════════════════════════════════════════════════
#  GLOBALS — loaded once at startup
# ════════════════════════════════════════════════════════════════════════════
MODEL, SYMPTOM_COLS = get_model_and_symptoms()

def _get_accuracy():
    X_train, X_test, y_train, y_test, _, _ = get_prepared_data()
    return float(np.mean(MODEL.predict(X_test) == y_test))

TEST_ACCURACY = _get_accuracy()
PORT = 5050


# ════════════════════════════════════════════════════════════════════════════
#  CHART GENERATION (returns base64 PNG string)
# ════════════════════════════════════════════════════════════════════════════
def make_xai_chart(contributions: list, disease: str) -> str:
    """Renders the XAI bar chart and returns it as a base64 PNG."""
    names   = [c[0].replace("_", " ").title() for c in contributions]
    values  = [c[1] for c in contributions]
    present = [c[2] for c in contributions]
    colors  = ["#2ecc71" if p else "#e74c3c" for p in present]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.45)),
                           facecolor="#1c2128")
    ax.set_facecolor("#1c2128")

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1],
                   edgecolor="#30363d", height=0.65)

    for bar, val in zip(bars, values[::-1]):
        x_pos = bar.get_width() + (0.03 if val >= 0 else -0.03)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color="#e6edf3")

    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Log-likelihood contribution", fontsize=9, color="#8b949e")
    ax.set_title(f"Why → {disease}", fontsize=11, fontweight="bold",
                 color="#e6edf3", pad=10)

    patch_pos = mpatches.Patch(color="#2ecc71", label="Present  → supports")
    patch_neg = mpatches.Patch(color="#e74c3c", label="Absent   → against")
    ax.legend(handles=[patch_pos, patch_neg], fontsize=8,
              facecolor="#21262d", labelcolor="#e6edf3", edgecolor="#30363d")

    ax.tick_params(colors="#e6edf3", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ════════════════════════════════════════════════════════════════════════════
#  HTML PAGE
# ════════════════════════════════════════════════════════════════════════════
def build_html() -> str:
    symptom_checkboxes = ""
    for sym in SYMPTOM_COLS:
        label = sym.replace("_", " ").title()
        symptom_checkboxes += (
            f'<label class="cb-item">'
            f'<input type="checkbox" name="symptom" value="{sym}" '
            f'onchange="liveUpdate()"> {label}</label>\n'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Medical Diagnosis Assistant — B.Tech Mini-Project</title>
<style>
  :root {{
    --bg:      #0d1117; --panel:  #161b22; --card:   #1c2128;
    --input:   #21262d; --border: #30363d; --accent: #58a6ff;
    --green:   #3fb950; --red:    #f85149; --gold:   #d29922;
    --text:    #e6edf3; --muted:  #8b949e;
    --font: 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font);
          min-height: 100vh; display: flex; flex-direction: column; }}

  /* HEADER */
  header {{ background: var(--panel); border-bottom: 1px solid var(--border);
            padding: 14px 24px; display: flex; justify-content: space-between;
            align-items: center; }}
  header h1 {{ font-size: 1.3rem; }}
  .badge {{ background: var(--gold); color: #000; padding: 5px 12px;
            border-radius: 6px; font-size: .85rem; font-weight: 600; }}

  /* BODY */
  .container {{ display: flex; flex: 1; gap: 0; overflow: hidden; }}

  /* LEFT */
  .left {{ width: 340px; min-width: 280px; background: var(--panel);
           border-right: 1px solid var(--border); display: flex;
           flex-direction: column; padding: 16px; gap: 12px;
           height: calc(100vh - 57px); overflow: hidden; }}
  .section-title {{ color: var(--accent); font-weight: 600; font-size: .9rem; }}

  .info-row {{ display: flex; gap: 14px; align-items: center; }}
  .info-row label {{ color: var(--muted); font-size: .85rem; }}
  .info-row input, .info-row select {{
    background: var(--input); border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 4px 8px;
    font-size: .85rem; width: 70px; }}
  .info-row select {{ width: 90px; }}

  #search {{ width: 100%; background: var(--input); border: 1px solid var(--border);
             color: var(--text); border-radius: 6px; padding: 7px 10px;
             font-size: .88rem; }}
  #search:focus {{ outline: none; border-color: var(--accent); }}

  .cb-list {{ flex: 1; overflow-y: auto; display: flex; flex-direction: column;
              gap: 2px; padding-right: 4px; }}
  .cb-list::-webkit-scrollbar {{ width: 4px; }}
  .cb-list::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
  .cb-item {{ display: flex; align-items: center; gap: 8px; padding: 5px 8px;
              border-radius: 6px; cursor: pointer; font-size: .88rem; }}
  .cb-item:hover {{ background: var(--card); }}
  .cb-item input {{ accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }}

  .btn-row {{ display: flex; gap: 10px; }}
  .btn {{ padding: 10px 20px; border: none; border-radius: 8px; font-size: .9rem;
          font-weight: 600; cursor: pointer; transition: opacity .15s; }}
  .btn:hover {{ opacity: .85; }}
  .btn-primary {{ background: var(--accent); color: #000; }}
  .btn-secondary {{ background: var(--input); color: var(--muted); }}

  .live-toggle {{ display: flex; align-items: center; gap: 8px;
                  font-size: .82rem; color: var(--muted); }}
  .live-toggle input {{ accent-color: var(--accent); }}

  /* RIGHT */
  .right {{ flex: 1; overflow-y: auto; padding: 20px 24px; }}
  .right::-webkit-scrollbar {{ width: 6px; }}
  .right::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}

  .placeholder {{ display: flex; flex-direction: column; align-items: center;
                  justify-content: center; height: 60vh; gap: 16px;
                  color: var(--muted); text-align: center; }}
  .placeholder .icon {{ font-size: 5rem; opacity: .3; }}

  .sec-label {{ color: var(--accent); font-weight: 600; font-size: .95rem;
                margin: 18px 0 8px; border-left: 3px solid var(--accent);
                padding-left: 10px; }}

  /* DISEASE CARDS */
  .disease-card {{ background: var(--card); border: 1px solid var(--border);
                   border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; }}
  .card-top {{ display: flex; align-items: center; gap: 14px; }}
  .rank {{ padding: 4px 10px; border-radius: 6px; color: #fff;
           font-weight: 700; font-size: .95rem; }}
  .rank-1 {{ background: var(--red); }}
  .rank-2 {{ background: var(--gold); }}
  .rank-3 {{ background: var(--green); }}
  .disease-name {{ font-weight: 600; font-size: 1rem; }}
  .conf-text {{ font-size: .88rem; margin-top: 4px; }}
  .bar-track {{ background: var(--input); border-radius: 4px; height: 8px;
                margin-top: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width .4s; }}

  /* RECOMMENDATION */
  .rec-box {{ background: var(--card); border: 1px solid var(--border);
              border-radius: 10px; padding: 14px 18px; font-size: .9rem; }}

  /* XAI CHART */
  .chart-wrap img {{ width: 100%; max-width: 720px; border-radius: 10px;
                     border: 1px solid var(--border); margin-top: 6px; }}

  /* LOADING */
  #spinner {{ display: none; color: var(--accent); margin: 30px auto;
              font-size: 1rem; text-align: center; }}

  /* FOOTER */
  footer {{ background: var(--panel); border-top: 1px solid var(--border);
            padding: 8px 24px; font-size: .78rem; color: var(--muted);
            text-align: center; }}
</style>
</head>
<body>

<header>
  <h1>🏥 AI Medical Diagnosis Assistant &nbsp;·&nbsp; B.Tech Mini-Project</h1>
  <span class="badge">Model Accuracy: {TEST_ACCURACY*100:.1f}%</span>
</header>

<div class="container">
  <!-- ═══ LEFT PANEL ═══ -->
  <div class="left">
    <div class="section-title">👤 Patient Information</div>
    <div class="info-row">
      <label>Age</label>
      <input type="number" id="age" value="25" min="1" max="120">
      <label>Gender</label>
      <select id="gender">
        <option>Male</option><option>Female</option><option>Other</option>
      </select>
    </div>

    <label class="live-toggle">
      <input type="checkbox" id="liveToggle" checked>
      ⚡ Live What-If update
    </label>

    <div class="section-title">🔍 Search Symptoms</div>
    <input type="text" id="search" placeholder="Type to filter…" oninput="filterSymptoms()">

    <div class="section-title">☑ Select Present Symptoms</div>
    <div class="cb-list" id="cbList">
      {symptom_checkboxes}
    </div>

    <div class="btn-row">
      <button class="btn btn-primary" onclick="diagnose()">🔬 DIAGNOSE</button>
      <button class="btn btn-secondary" onclick="clearAll()">🗑 Clear</button>
    </div>
  </div>

  <!-- ═══ RIGHT PANEL ═══ -->
  <div class="right" id="results">
    <div class="placeholder">
      <div class="icon">🩺</div>
      <div>Select symptoms on the left<br>and press <strong>DIAGNOSE</strong></div>
    </div>
  </div>
</div>

<footer>
  ⚠️ DISCLAIMER: This tool is for EDUCATIONAL PURPOSES ONLY and does NOT constitute medical advice.
  Always consult a qualified healthcare professional for diagnosis and treatment.
</footer>

<script>
function filterSymptoms() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.cb-item').forEach(el => {{
    el.style.display = el.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

function getSelected() {{
  return [...document.querySelectorAll('input[name=symptom]:checked')]
         .map(cb => cb.value);
}}

function liveUpdate() {{
  if (document.getElementById('liveToggle').checked) {{
    const sel = getSelected();
    if (sel.length > 0) diagnose();
  }}
}}

async function diagnose() {{
  const selected = getSelected();
  if (selected.length === 0) {{
    alert('Please select at least one symptom before diagnosing.');
    return;
  }}

  const panel = document.getElementById('results');
  panel.innerHTML = '<div id="spinner" style="display:block">⏳ Analysing…</div>';

  const resp = await fetch('/diagnose', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      symptoms: selected,
      age: document.getElementById('age').value,
      gender: document.getElementById('gender').value
    }})
  }});
  const data = await resp.json();
  renderResults(data, selected);
}}

function confidenceColor(p) {{
  if (p >= 0.70) return '#f85149';
  if (p >= 0.40) return '#d29922';
  return '#3fb950';
}}

function renderResults(data, selected) {{
  const top = data.top_diseases;
  let html = '';

  // Top diseases
  html += '<div class="sec-label">🏆 Top Predicted Diseases</div>';
  const rankCls = ['rank-1','rank-2','rank-3'];
  top.slice(0,3).forEach((d, i) => {{
    const pct = (d.prob * 100).toFixed(1);
    const col = confidenceColor(d.prob);
    html += `
      <div class="disease-card">
        <div class="card-top">
          <span class="rank ${{rankCls[i]}}">#${{i+1}}</span>
          <div>
            <div class="disease-name">${{d.disease}}</div>
            <div class="conf-text" style="color:${{col}}">${{pct}}% confidence</div>
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${{pct}}%;background:${{col}}"></div>
        </div>
      </div>`;
  }});

  // Recommendation
  html += '<div class="sec-label">💊 Recommendation</div>';
  const topProb = top[0].prob;
  let icon, msg, col;
  if (topProb >= 0.70) {{ icon='🔴'; msg='High confidence. Please consult a doctor immediately.'; col='#f85149'; }}
  else if (topProb >= 0.40) {{ icon='🟡'; msg='Moderate confidence. Consider seeing a healthcare professional.'; col='#d29922'; }}
  else {{ icon='🟢'; msg='Low confidence. Monitor symptoms and see a doctor if they worsen.'; col='#3fb950'; }}
  html += `<div class="rec-box" style="color:${{col}}">${{icon}} ${{msg}}</div>`;

  // XAI Chart
  html += '<div class="sec-label">🔍 Why This Diagnosis? (Symptom Contributions)</div>';
  html += `<div class="chart-wrap"><img src="data:image/png;base64,${{data.chart}}" alt="XAI Chart"></div>`;

  // What-If hint
  html += `<div class="sec-label">⚡ What-If Mode</div>
    <p style="color:var(--muted);font-size:.85rem;margin-top:4px">
      Toggle any symptom checkbox to instantly see how the prediction changes
      (requires "Live What-If update" checkbox to be ticked).
    </p>`;

  document.getElementById('results').innerHTML = html;
}}

function clearAll() {{
  document.querySelectorAll('input[name=symptom]').forEach(cb => cb.checked = false);
  document.getElementById('search').value = '';
  filterSymptoms();
  document.getElementById('results').innerHTML = `
    <div class="placeholder">
      <div class="icon">🩺</div>
      <div>Select symptoms on the left<br>and press <strong>DIAGNOSE</strong></div>
    </div>`;
}}
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════
#  HTTP REQUEST HANDLER
# ════════════════════════════════════════════════════════════════════════════
class DiagnosisHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass   # Silence default access log

    def do_GET(self):
        if urlparse(self.path).path in ("/", "/index.html"):
            body = build_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if urlparse(self.path).path == "/diagnose":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            symptoms = payload.get("symptoms", [])

            # ── Run model ────────────────────────────────────────────────
            top = predict_top_diseases(symptoms, k=5)
            top_disease = top[0][0] if top else None

            # ── XAI chart ────────────────────────────────────────────────
            chart_b64 = ""
            if top_disease:
                contribs = get_top_contributions(top_disease, symptoms, top_n=12)
                chart_b64 = make_xai_chart(contribs, top_disease)

            result = {
                "top_diseases": [{"disease": d, "prob": round(p, 4)} for d, p in top],
                "chart": chart_b64
            }
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def run_app():
    server = HTTPServer(("127.0.0.1", PORT), DiagnosisHandler)
    url = f"http://localhost:{PORT}"
    print(f"\n  🌐  Server running at  {url}")
    print(f"  📖  Opening in your browser …")
    print(f"  ⌨️   Press  Ctrl+C  to stop\n")

    # Open browser after a short delay (server needs to start first)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  👋  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    run_app()
