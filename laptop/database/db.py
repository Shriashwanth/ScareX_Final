"""
Unified SQLite Database Logger for ScareX Integrated System
Stores bird deterrence activations, tomato crop maturity sessions, robot actuation events, and system telemetry.
"""
import os
import sqlite3
import threading
from datetime import datetime
try:
    import laptop.config as config
except ImportError:
    import config

class DatabaseLogger:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    species TEXT,
                    bird_confidence REAL,
                    tomato_counts TEXT,
                    harvest_priority TEXT,
                    robot_action TEXT,
                    details TEXT
                )
            ''')
            conn.commit()
            conn.close()

    def log_event(self, event_type: str, species: str = None, bird_confidence: float = None,
                  tomato_counts: str = None, harvest_priority: str = None,
                  robot_action: str = None, details: str = None):
        """
        Inserts a single event log entry.
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (
                        event_type, species, bird_confidence, tomato_counts,
                        harvest_priority, robot_action, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_type, species, bird_confidence,
                    str(tomato_counts) if tomato_counts else None,
                    harvest_priority, robot_action, details
                ))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"[Database] Error logging event: {e}")

    def get_recent_events(self, limit: int = 50) -> list:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, timestamp, event_type, species, bird_confidence,
                           tomato_counts, harvest_priority, robot_action, details
                    FROM events
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
                conn.close()

                events = []
                for r in rows:
                    events.append({
                        "id": r[0],
                        "timestamp": r[1],
                        "event_type": r[2],
                        "species": r[3] or "None",
                        "bird_confidence": round(float(r[4]), 2) if r[4] is not None else 0.0,
                        "tomato_counts": r[5] or "",
                        "harvest_priority": r[6] or "N/A",
                        "robot_action": r[7] or "None",
                        "details": r[8] or ""
                    })
                return events
        except Exception as e:
            print(f"[Database] Error retrieving events: {e}")
            return []

db = DatabaseLogger()
