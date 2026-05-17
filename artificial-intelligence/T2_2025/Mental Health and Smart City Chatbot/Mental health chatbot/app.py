import os
import json
import glob
from typing import List, Dict, Tuple

from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import uuid
import datetime as dt
import csv, io, sqlite3
from urllib.parse import urlparse
try:
    import psycopg2
    _psycopg2_available = True
except Exception:
    _psycopg2_available = False


from flask import send_file
import io  
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet  
from reportlab.pdfbase.pdfmetrics import stringWidth  
import textwrap


# =========================
# Configuration (env vars)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Session + memory
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "100"))  # total messages kept (user+bot), 10 ~= last 5 exchanges

K = int(os.getenv("K", "3"))  
MIN_SIM = float(os.getenv("MIN_SIM", "0.05"))  
FALLBACK_PREFIX = os.getenv(
    "FALLBACK_PREFIX",
    "This question isn’t covered by the provided data, but here’s an answer from Gemini:"
)

DATA_GLOB = os.getenv("DATA_GLOB", "*.csv")
DATA_PATH = os.getenv("DATA_PATH", "")  

CSV_Q_COL = os.getenv("CSV_Q_COL", "")  
CSV_A_COL = os.getenv("CSV_A_COL", "") 

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Run: heroku config:set GEMINI_API_KEY=...")

app = Flask(__name__)
app.secret_key = SECRET_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# Globals for retrieval
_vectorizer: TfidfVectorizer = None
_matrix = None
_raw_items: List[Dict] = []
_display_blobs: List[str] = []


# =========================
# Helpers: dataset shaping
def _item_to_text_blob(item: Dict) -> str:
    """Build a searchable text blob from many possible schemas."""
    if isinstance(item, dict):
        preferred_keys = [
            "question", "q", "prompt", "title", "heading", "query", "input", "output"
            "answer", "a", "response", "content", "text", "body", "description", "context"
        ]
        parts = []
        for k in preferred_keys:
            if k in item and item[k] is not None:
                parts.append(str(item[k]))
        if not parts:
            parts = [str(v) for v in item.values() if v is not None]
        blob = " ".join(parts).strip()
        return blob if blob else json.dumps(item, ensure_ascii=False)
    return str(item)


def _item_to_context_block(item: Dict) -> str:
    """Human-readable block that we show Gemini as CONTEXT."""
    if not isinstance(item, dict):
        return str(item)
    q = item.get("input") or item.get("q") or item.get("prompt") or item.get("title") or ""
    a = item.get("output") or item.get("a") or item.get("response") or item.get("content") or item.get("text") or ""
    ctx = item.get("context") or item.get("description") or ""
    lines = []
    if ctx:
        lines.append(str(ctx))
    if q or a:
        lines.append(f"Q: {q}\nA: {a}")
    block = "\n".join([ln for ln in lines if ln.strip()])
    return block if block.strip() else _item_to_text_blob(item)


def _rows_from_csv(path: str) -> List[Dict]:
    """Read one CSV and map rows into dicts with best-guess Q/A fields."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    df.columns = [str(c).strip().lower() for c in df.columns]

    q_col = CSV_Q_COL.lower() if CSV_Q_COL else None
    a_col = CSV_A_COL.lower() if CSV_A_COL else None

    if not q_col:
        for cand in ["question", "q", "prompt", "title", "query", "input"]:
            if cand in df.columns:
                q_col = cand
                break
    if not a_col:
        for cand in ["answer", "a", "response", "content", "text", "reply", "output"]:
            if cand in df.columns:
                a_col = cand
                break

    rows: List[Dict] = []
    for _, r in df.iterrows():
        record: Dict = {}
        for c in df.columns:
            val = r.get(c)
            if pd.isna(val):
                continue
            record[c] = str(val)
        if q_col and q_col in df.columns:
            record.setdefault("question", str(r[q_col]))
        if a_col and a_col in df.columns:
            record.setdefault("answer", str(r[a_col]))
        rows.append(record)
    return rows


def load_dataset():
    """Load dataset from DATA_PATH (single file) or DATA_GLOB (many CSVs), then build TF-IDF index."""
    global _vectorizer, _matrix, _raw_items, _display_blobs

    items: List[Dict] = []

    if DATA_PATH:
        p = DATA_PATH
        if p.lower().endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                print(f"[INFO] Loaded JSON items from {p}: {len(items)}")
            except Exception as e:
                print(f"[WARN] Could not load JSON from {p}: {e}")
        elif p.lower().endswith(".csv"):
            try:
                batch = _rows_from_csv(p)
                items.extend(batch)
                print(f"[INFO] Loaded {len(batch)} CSV rows from {p}")
            except Exception as e:
                print(f"[WARN] Failed to load CSV {p}: {e}")
        else:
            print(f"[WARN] Unsupported DATA_PATH extension: {p}")
    else:
        paths = sorted(glob.glob(DATA_GLOB))
        if not paths:
            print(f"[INFO] No CSV files matched {DATA_GLOB}. Running pure Gemini mode.")
        total = 0
        for p in paths:
            try:
                batch = _rows_from_csv(p)
                items.extend(batch)
                total += len(batch)
                print(f"[INFO] Loaded {len(batch)} rows from {p}")
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")
        if total:
            print(f"[INFO] Total CSV rows loaded: {total}")

    _raw_items = items
    corpus = [_item_to_text_blob(it) for it in _raw_items]
    _display_blobs = [_item_to_context_block(it) for it in _raw_items]

    if corpus:
        _vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        _matrix = _vectorizer.fit_transform(corpus)
        print(f"[INFO] Indexed {len(corpus)} items.")
    else:
        _vectorizer, _matrix = None, None
        print("[INFO] No dataset items loaded; fallback to pure Gemini.")

load_dataset()


# =========================
# Retrieval + chat helpers
def retrieve_top_k(query: str, k: int) -> List[Tuple[int, float]]:
    """Return [(row_index, similarity), ...] for the top-k matches."""
    if not query or _vectorizer is None or _matrix is None:
        return []
    qv = _vectorizer.transform([query])
    sims = cosine_similarity(qv, _matrix).ravel()
    if sims.size == 0:
        return []
    idxs = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idxs]

COOKIE_NAME = os.getenv("JOURNAL_COOKIE_NAME", "journal_id")
COOKIE_MAX_DAYS = int(os.getenv("JOURNAL_COOKIE_DAYS", "180"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

def _use_postgres():
    return bool(DATABASE_URL) and DATABASE_URL.startswith(("postgres://","postgresql://")) and _psycopg2_available
def _pg_conn():
    assert _use_postgres(), "Postgres not available"
    return psycopg2.connect(DATABASE_URL, sslmode="require")
def _sqlite_conn():
    path = os.getenv("JOURNAL_DB_PATH", "journal.db")
    return sqlite3.connect(path, check_same_thread=False)
def _get_conn():
    return _pg_conn() if _use_postgres() else _sqlite_conn()

def _init_db():
    conn = _get_conn()
    try:
        cur = conn.cursor()
        if _use_postgres():
            cur.execute("""
                CREATE TABLE IF NOT EXISTS journal_entries(
                  id SERIAL PRIMARY KEY,
                  journal_id TEXT NOT NULL,
                  mood INTEGER NOT NULL CHECK (mood BETWEEN 1 AND 5),
                  emotions TEXT DEFAULT '',
                  note TEXT DEFAULT '',
                  ts TIMESTAMPTZ NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_journal_entries_jid_ts ON journal_entries (journal_id, ts);
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS journal_entries(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  journal_id TEXT NOT NULL,
                  mood INTEGER NOT NULL CHECK (mood BETWEEN 1 AND 5),
                  emotions TEXT DEFAULT '',
                  note TEXT DEFAULT '',
                  ts TEXT NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_journal_entries_jid_ts ON journal_entries (journal_id, ts);")
        conn.commit()
    finally:
        try: cur.close()
        except: pass
        conn.close()
_init_db()

def _now_utc(): return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
def _insert_entry(jid, mood, emotions_csv, note):
    conn = _get_conn()
    try:
        cur = conn.cursor()
        if _use_postgres():
            cur.execute("INSERT INTO journal_entries (journal_id,mood,emotions,note,ts) VALUES (%s,%s,%s,%s,%s);",
                        (jid, mood, emotions_csv, note, _now_utc()))
        else:
            cur.execute("INSERT INTO journal_entries (journal_id,mood,emotions,note,ts) VALUES (?,?,?,?,?);",
                        (jid, mood, emotions_csv, note, _now_utc().isoformat()))
        conn.commit()
    finally:
        try: cur.close()
        except: pass
        conn.close()

def _fetch_history(jid, limit=365):
    conn = _get_conn(); rows=[]
    try:
        cur = conn.cursor()
        if _use_postgres():
            cur.execute("SELECT mood,emotions,note,ts FROM journal_entries WHERE journal_id=%s ORDER BY ts ASC LIMIT %s;",
                        (jid, limit))
        else:
            cur.execute("SELECT mood,emotions,note,ts FROM journal_entries WHERE journal_id=? ORDER BY ts ASC LIMIT ?;",
                        (jid, limit))
        for r in cur.fetchall():
            mood, emotions, note, ts = r
            rows.append({"mood": int(mood), "emotions": emotions or "", "note": note or "",
                         "ts": (ts.isoformat() if hasattr(ts, "isoformat") else str(ts))})
    finally:
        try: cur.close()
        except: pass
        conn.close()
    return rows

def _get_or_make_journal_id():
    jid = request.cookies.get(COOKIE_NAME)
    new_cookie = False
    if not jid:
        jid = uuid.uuid4().hex
        new_cookie = True
    return jid, new_cookie


# =========================
# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    _data = request.get_json(silent=True) or {}
    _umsg = (_data.get("message") or "").strip()
    lowered = _umsg.lower()
    triggers = [
        "download my chat", "download my chats", "download this chat",
        "export my chat", "export my chats", "save this chat"
        "save my chat", "save my chats",
        "download conversation", "export conversation"
    ]
    if any(t in lowered for t in triggers):
        return jsonify({
            "response": "Here’s your download link for this chat session.",
            "download_url": "/history/pdf"
        }), 200

    user_message = _umsg
    if not user_message:
        return jsonify({"response": "Please type a message."}), 400

    history = session.get("history", [])

    pairs = retrieve_top_k(user_message, K)
    valid_pairs = [(i, s) for (i, s) in pairs if s >= MIN_SIM]
    use_dataset = len(valid_pairs) > 0

    if use_dataset:
        context_blocks = [_display_blobs[idx] for idx, _ in valid_pairs]
        context_text = "\n\n---\n\n".join([b for b in context_blocks if b.strip()])

        system_hint = (
            "You are a helpful health assistant. Answer the user using the information in the CONTEXT."
            "treat normal greetings as normal and respond them how you see fit"
            "If the question is out of context and isn't a greeting but is related to health or medical condition in any way, give a proper answer to the user."
            "If the question is out of context and isn't a greeting but isn't related to health or medical emergency in any way, give a generic short answer and prompt that you are just a health chatbot."
            "Do not fabricate details."
            "never say you are a health chatbot and  never direct the user to somewhere else. always take the question ang give a proper answer"
        )
        grounded_prompt = f"{system_hint}\n\nCONTEXT:\n{context_text}\n\nUser: {user_message}\nAnswer:"

        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(grounded_prompt)
            bot_reply = getattr(response, "text", None) or "Sorry, I couldn't generate a response."
        except Exception as e:
            bot_reply = f"Error: {e}"
    else:
        try:
            chat_session = model.start_chat(history=history)
            response = chat_session.send_message(user_message)
            gen_reply = getattr(response, "text", None) or "Sorry, I couldn't generate a response."
        except Exception as e:
            gen_reply = f"Error: {e}"
        bot_reply = f"{gen_reply}"

    history.append({"role": "user", "parts": [user_message]})
    history.append({"role": "model", "parts": [bot_reply]})
    session["history"] = history[-MAX_HISTORY:]

    return jsonify({"response": bot_reply})


def _render_history_to_pdf(history):
    """
    history: list of {"role": "user"|"model", "parts": [text]}
    returns: BytesIO of a generated PDF
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    left_margin = 2.0 * cm
    right_margin = 2.0 * cm
    top_margin = 2.0 * cm
    bottom_margin = 2.0 * cm
    max_width = width - left_margin - right_margin

    line_height = 14
    y = height - top_margin
    title = "Chat Transcript"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left_margin, y, title)
    y -= 20

    import datetime as _dt
    c.setFont("Helvetica", 9)
    c.drawString(left_margin, y, f"Generated: {_dt.datetime.utcnow().isoformat()}Z")
    y -= 16

    c.setFont("Helvetica", 11)
    for item in history:
        role = item.get("role", "model")
        parts = item.get("parts", [])
        text = str(parts[0]) if parts else ""
        prefix = "You: " if role == "user" else "Bot: "

        wrap_at = 100
        wrapped = []
        for line in (prefix + text).splitlines():
            wrapped.extend(textwrap.wrap(line, width=wrap_at) or [""])

        for line in wrapped:
            if y <= bottom_margin + line_height:
                c.showPage()
                y = height - top_margin
                c.setFont("Helvetica", 11)
            c.drawString(left_margin, y, line)
            y -= line_height
        y -= 6 

    c.showPage()
    c.save()
    buf.seek(0)
    return buf



@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return jsonify({"ok": True})


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.post("/journal/entry")
def journal_entry():
    payload = request.get_json(silent=True) or {}
    try:
        mood = int(payload.get("mood", 0))
    except Exception:
        return jsonify({"error": "mood must be an integer 1–5"}), 400
    if mood < 1 or mood > 5:
        return jsonify({"error": "mood must be between 1 and 5"}), 400

    emotions = payload.get("emotions", [])
    if isinstance(emotions, list):
        emotions_csv = ",".join([str(e).strip() for e in emotions if str(e).strip()])
    else:
        emotions_csv = str(emotions or "").strip()

    note = str(payload.get("note", "")).strip()[:2000]

    jid, need_cookie = _get_or_make_journal_id()
    _insert_entry(jid, mood, emotions_csv, note)

    resp = jsonify({"ok": True, "journal_id": jid})
    if need_cookie:
        expires = dt.datetime.utcnow() + dt.timedelta(days=COOKIE_MAX_DAYS)
        resp.set_cookie(COOKIE_NAME, jid, secure=True, httponly=True, samesite="Lax", expires=expires)
    return resp, 200


@app.get("/journal/history")
def journal_history():
    jid = request.cookies.get(COOKIE_NAME)
    if not jid:
        return jsonify({"entries": [], "note": "no journal cookie yet"}), 200
    entries = _fetch_history(jid)
    return jsonify({"entries": entries, "count": len(entries)}), 200


@app.get("/journal/export.csv")
def journal_export_csv():
    jid = request.cookies.get(COOKIE_NAME)
    if not jid:
        return jsonify({"error": "no journal cookie yet"}), 400
    entries = _fetch_history(jid, limit=10000)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["timestamp","mood","emotions","note"])
    for e in entries:
        w.writerow([e["ts"], e["mood"], e["emotions"], e["note"]])
    out = buf.getvalue().encode("utf-8")
    from flask import make_response
    resp = make_response(out)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="journal_export.csv"'
    return resp


@app.get("/history/pdf")
def history_pdf():
    history = session.get("history", [])
    pdf = _render_history_to_pdf(history or [])
    return send_file(
        pdf, as_attachment=True,
        download_name="chat_transcript.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
