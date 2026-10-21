"""
Database module for storing scan history and feedback using SQLite.
"""

import sqlite3
import datetime
import os
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "spam_detector.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for scan history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        subject TEXT,
        preview TEXT,
        verdict TEXT NOT NULL,
        verdict_badge TEXT NOT NULL,
        spam_probability REAL NOT NULL,
        risk_level TEXT NOT NULL,
        url_count INTEGER DEFAULT 0,
        threat_count INTEGER DEFAULT 0
    )
    """)
    
    # Table for user feedback (active learning / model improvement)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id INTEGER,
        email_text TEXT,
        predicted_verdict TEXT,
        user_corrected_label TEXT NOT NULL,
        notes TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (scan_id) REFERENCES scans(id)
    )
    """)
    
    conn.commit()
    conn.close()

def save_scan(
    subject: str,
    preview: str,
    verdict: str,
    verdict_badge: str,
    spam_probability: float,
    risk_level: str,
    url_count: int,
    threat_count: int
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    cursor.execute("""
        INSERT INTO scans (
            timestamp, subject, preview, verdict, verdict_badge,
            spam_probability, risk_level, url_count, threat_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now,
        subject[:120] if subject else "No Subject",
        preview[:200] if preview else "",
        verdict,
        verdict_badge,
        spam_probability,
        risk_level,
        url_count,
        threat_count
    ))
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def get_recent_scans(limit: int = 15) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM scans ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_feedback(
    scan_id: Optional[int],
    email_text: str,
    predicted_verdict: str,
    user_corrected_label: str,
    notes: str = ""
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO feedback (
            scan_id, email_text, predicted_verdict, user_corrected_label, notes, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        scan_id,
        email_text[:3000] if email_text else "",
        predicted_verdict,
        user_corrected_label,
        notes,
        now
    ))
    fb_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return fb_id

def get_total_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM scans")
    total_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scans WHERE verdict = 'SPAM'")
    spam_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scans WHERE verdict LIKE 'HAM%'")
    ham_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM scans WHERE verdict = 'SUSPICIOUS'")
    suspicious_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM feedback")
    total_feedback = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_scans": total_scans,
        "spam_scans": spam_scans,
        "ham_scans": ham_scans,
        "suspicious_scans": suspicious_scans,
        "total_feedback": total_feedback
    }

# Initialize on import
init_db()
