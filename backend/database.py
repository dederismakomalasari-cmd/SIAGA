"""Database SQLite sederhana: user & log alert."""
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "siaga.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,           -- 'guard' | 'kai'
            name TEXT NOT NULL,
            crossing_id TEXT              -- null untuk role kai
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crossing_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            risk_level INTEGER NOT NULL,
            risk_label TEXT NOT NULL,
            tta_seconds REAL,
            snapshot_path TEXT,
            objects_json TEXT,
            acked INTEGER DEFAULT 0,
            acked_at TEXT,
            acked_by TEXT,
            ack_note TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tta_override (
            crossing_id TEXT PRIMARY KEY,
            tta_seconds REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Seed default users (ganti password di produksi!)
    users = [
        ("guard001", "guard123", "guard", "Petugas JPL-001", "JPL-001"),
        ("guard002", "guard123", "guard", "Petugas JPL-002", "JPL-002"),
        ("kai001",   "kai123",   "kai",   "Pengawas KAI Pusat", None),
    ]
    for u, p, r, n, cid in users:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, role, name, crossing_id) VALUES (?,?,?,?,?)",
            (u, hash_password(p), r, n, cid),
        )
    conn.commit()
    conn.close()


def authenticate(username: str, password: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND password_hash=?",
        (username, hash_password(password)),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_token(token: str):
    # token = username (sederhana, bisa diganti JWT di produksi)
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (token,)).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_alert(crossing_id, risk_level, risk_label, tta_seconds, snapshot_path, objects_json):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO alerts (crossing_id, timestamp, risk_level, risk_label,
           tta_seconds, snapshot_path, objects_json) VALUES (?,?,?,?,?,?,?)""",
        (crossing_id, datetime.utcnow().isoformat(), risk_level, risk_label,
         tta_seconds, snapshot_path, objects_json),
    )
    alert_id = cur.lastrowid
    conn.commit()
    conn.close()
    return alert_id


def ack_alert(alert_id, username, note):
    conn = get_conn()
    conn.execute(
        "UPDATE alerts SET acked=1, acked_at=?, acked_by=?, ack_note=? WHERE id=?",
        (datetime.utcnow().isoformat(), username, note, alert_id),
    )
    conn.commit()
    conn.close()


def get_alerts(crossing_id=None, limit=100):
    conn = get_conn()
    if crossing_id:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE crossing_id=? ORDER BY id DESC LIMIT ?",
            (crossing_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_tta_override(crossing_id, tta_seconds):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO tta_override (crossing_id, tta_seconds, updated_at) VALUES (?,?,?)",
        (crossing_id, tta_seconds, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_tta_override(crossing_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT tta_seconds FROM tta_override WHERE crossing_id=?", (crossing_id,)
    ).fetchone()
    conn.close()
    return row["tta_seconds"] if row else None
    