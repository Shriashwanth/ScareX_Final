"""
ScareX Standalone Raspberry Pi 5 - Unified SQLite Database Logger
Stores bird detections, deterrence actuations, tomato crop maturity sessions,
and hardware telemetry in persistent local storage on the Pi.
"""
import os
import sqlite3
import threading
from datetime import datetime, date

try:
    import raspberry_pi.config as config
except ImportError:
    try:
        import laptop.config as config
    except ImportError:
        import config

class DatabaseLogger:
    def __init__(self, db_path=None):
        if db_path:
            self.db_path = db_path
        elif hasattr(config, "DB_PATH"):
            self.db_path = config.DB_PATH
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, "database", "scarex_local.db")

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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    date TEXT PRIMARY KEY,
                    bird_detections INTEGER DEFAULT 0,
                    deterrence_activations INTEGER DEFAULT 0,
                    tomatoes_observed INTEGER DEFAULT 0,
                    harvest_priority TEXT DEFAULT 'STANDBY'
                )
            ''')
            conn.commit()
            conn.close()

    def log_event(self, event_type: str, species: str = None, bird_confidence: float = None,
                  tomato_counts: str = None, harvest_priority: str = None,
                  robot_action: str = None, details: str = None):
        """
        Inserts a single event log entry in a thread-safe transaction.
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
                    ORDER BY id DESC
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

    def get_events_by_date_range(self, start_date: str = None, end_date: str = None) -> list:
        """
        Returns events within date range (YYYY-MM-DD format).
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                if start_date and end_date:
                    cursor.execute('''
                        SELECT id, timestamp, event_type, species, bird_confidence,
                               tomato_counts, harvest_priority, robot_action, details
                        FROM events
                        WHERE date(timestamp) BETWEEN date(?) AND date(?)
                        ORDER BY timestamp ASC
                    ''', (start_date, end_date))
                elif start_date:
                    cursor.execute('''
                        SELECT id, timestamp, event_type, species, bird_confidence,
                               tomato_counts, harvest_priority, robot_action, details
                        FROM events
                        WHERE date(timestamp) >= date(?)
                        ORDER BY timestamp ASC
                    ''', (start_date,))
                else:
                    cursor.execute('''
                        SELECT id, timestamp, event_type, species, bird_confidence,
                               tomato_counts, harvest_priority, robot_action, details
                        FROM events
                        ORDER BY timestamp DESC
                        LIMIT 100
                    ''')

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
            print(f"[Database] Error querying range: {e}")
            return []

    def get_summary_stats(self) -> dict:
        """
        Computes aggregate statistics across all logged events.
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM events")
                total_events = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM events WHERE event_type LIKE '%DETERRENCE%'")
                total_scares = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT species) FROM events WHERE species IS NOT NULL AND species != 'None'")
                distinct_species = cursor.fetchone()[0]

                conn.close()
                return {
                    "total_events": total_events,
                    "total_scares": total_scares,
                    "distinct_species_count": distinct_species
                }
        except Exception as e:
            return {"total_events": 0, "total_scares": 0, "distinct_species_count": 0}

# Global singleton
db = DatabaseLogger()
