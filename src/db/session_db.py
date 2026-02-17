"""
Lightweight SQLite session database for call persistence and analytics.
All writes are non-blocking (background thread) - zero latency impact on audio path.
Reads are direct (fast for SQLite).
"""

import sqlite3
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
from loguru import logger

DB_PATH = Path(__file__).parent.parent.parent / "data" / "calls.db"


class SessionDB:
    def __init__(self):
        self._write_queue = queue.Queue()
        self._writer_thread = None
        self._db_path = DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._start_writer()

    def _init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute('''CREATE TABLE IF NOT EXISTS calls (
            call_uuid TEXT PRIMARY KEY,
            plivo_uuid TEXT,
            phone TEXT,
            contact_name TEXT,
            client_name TEXT,
            status TEXT DEFAULT 'pending',
            started_at TEXT,
            ended_at TEXT,
            duration_seconds REAL,
            questions_completed INTEGER DEFAULT 0,
            total_questions INTEGER DEFAULT 0,
            transcript TEXT,
            call_summary TEXT,
            interest_level TEXT,
            collected_responses TEXT,
            objections_raised TEXT,
            webhook_url TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        # Cross-Call Memory: per-phone contact memory for repeat callers
        conn.execute('''CREATE TABLE IF NOT EXISTS contact_memory (
            phone TEXT PRIMARY KEY,
            name TEXT,
            persona TEXT,
            company TEXT,
            role TEXT,
            objections TEXT DEFAULT '[]',
            interest_areas TEXT DEFAULT '[]',
            key_facts TEXT DEFAULT '[]',
            call_count INTEGER DEFAULT 0,
            last_call_date TEXT,
            last_call_summary TEXT,
            last_call_outcome TEXT,
            all_call_uuids TEXT DEFAULT '[]',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        # Social Proof Engine: enrollment stats by company, city, role
        conn.execute('''CREATE TABLE IF NOT EXISTS social_proof_company (
            company_name TEXT PRIMARY KEY COLLATE NOCASE,
            enrollments_count INTEGER DEFAULT 0,
            last_enrollment_date TEXT,
            notable_outcomes TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS social_proof_city (
            city_name TEXT PRIMARY KEY COLLATE NOCASE,
            enrollments_count INTEGER DEFAULT 0,
            trending INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS social_proof_role (
            role_name TEXT PRIMARY KEY COLLATE NOCASE,
            enrollments_count INTEGER DEFAULT 0,
            avg_salary_increase TEXT,
            success_stories TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        # Migration: add linguistic_style column if missing
        try:
            conn.execute("ALTER TABLE contact_memory ADD COLUMN linguistic_style TEXT DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
        conn.close()
        logger.info(f"Session DB initialized: {self._db_path}")

    def _start_writer(self):
        """Start background writer thread - all writes go through here"""
        def writer():
            conn = sqlite3.connect(str(self._db_path))
            while True:
                try:
                    item = self._write_queue.get(timeout=2.0)
                    if item is None:
                        break
                    sql, params = item
                    conn.execute(sql, params)
                    conn.commit()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"DB write error: {e}")
            conn.close()

        self._writer_thread = threading.Thread(target=writer, daemon=True, name="session-db-writer")
        self._writer_thread.start()

    def create_call(self, call_uuid: str, phone: str, contact_name: str = "Customer",
                    client_name: str = "fwai", webhook_url: str = None, total_questions: int = 0):
        """Queue a new call record (non-blocking)"""
        self._write_queue.put((
            '''INSERT OR REPLACE INTO calls
               (call_uuid, phone, contact_name, client_name, status, webhook_url, total_questions, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)''',
            (call_uuid, phone, contact_name, client_name, webhook_url, total_questions,
             datetime.now().isoformat())
        ))

    def update_call(self, call_uuid: str, **kwargs):
        """Queue a call update (non-blocking). Pass any column as kwarg."""
        if not kwargs:
            return
        sets = []
        params = []
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                sets.append(f"{key} = ?")
                params.append(value)
        if not sets:
            return
        params.append(call_uuid)
        self._write_queue.put((
            f"UPDATE calls SET {', '.join(sets)} WHERE call_uuid = ?",
            tuple(params)
        ))

    def get_call(self, call_uuid: str) -> dict:
        """Read call record (blocking but fast ~1ms for SQLite)"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM calls WHERE call_uuid = ?", (call_uuid,)).fetchone()
        conn.close()
        if row:
            result = dict(row)
            # Parse JSON fields
            for field in ('collected_responses', 'objections_raised'):
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return result
        return None

    def get_recent_calls(self, limit: int = 50) -> list:
        """Get recent calls for dashboard/API"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT call_uuid, phone, contact_name, client_name, status, duration_seconds, "
            "questions_completed, total_questions, interest_level, call_summary, created_at "
            "FROM calls ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Cross-Call Memory
    # =========================================================================

    def get_contact_memory(self, phone: str) -> dict:
        """Load contact memory by phone number (blocking, fast ~1ms)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM contact_memory WHERE phone = ?", (phone,)).fetchone()
        conn.close()
        if row:
            result = dict(row)
            for field in ('objections', 'interest_areas', 'key_facts', 'all_call_uuids'):
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return result
        return None

    def save_contact_memory(self, phone: str, **kwargs):
        """Upsert contact memory (non-blocking). JSON fields are auto-serialized."""
        if not phone:
            return
        fields = {}
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                fields[key] = value
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        # Build upsert: INSERT ... ON CONFLICT(phone) DO UPDATE SET col = excluded.col
        columns = list(fields.keys())
        values = [phone] + [fields[c] for c in columns]
        placeholders = ", ".join(["?"] * len(values))
        col_list = "phone, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = excluded.{c}" for c in columns)
        self._write_queue.put((
            f"""INSERT INTO contact_memory ({col_list})
                VALUES ({placeholders})
                ON CONFLICT(phone) DO UPDATE SET {conflict_updates}""",
            tuple(values)
        ))

    def get_all_contact_memories(self, limit: int = 100) -> list:
        """List all contact memories for dashboard/API."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT phone, name, persona, company, role, call_count, "
            "last_call_date, last_call_outcome, updated_at "
            "FROM contact_memory ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def delete_contact_memory(self, phone: str):
        """Delete contact memory for a phone number (non-blocking)."""
        self._write_queue.put((
            "DELETE FROM contact_memory WHERE phone = ?", (phone,)
        ))

    # =========================================================================
    # Social Proof Stats
    # =========================================================================

    def get_social_proof_by_company(self, company_name: str) -> dict:
        """Look up company enrollment stats (blocking, fast ~1ms)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM social_proof_company WHERE company_name = ? COLLATE NOCASE",
            (company_name,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_social_proof_by_city(self, city_name: str) -> dict:
        """Look up city enrollment stats (blocking, fast ~1ms)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM social_proof_city WHERE city_name = ? COLLATE NOCASE",
            (city_name,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_social_proof_by_role(self, role_name: str) -> dict:
        """Look up role enrollment stats (blocking, fast ~1ms)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM social_proof_role WHERE role_name = ? COLLATE NOCASE",
            (role_name,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def get_social_proof_top(self, table: str, limit: int = 5) -> list:
        """Get top stats by enrollment count for summary generation."""
        valid_tables = {"social_proof_company", "social_proof_city", "social_proof_role"}
        if table not in valid_tables:
            return []
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY enrollments_count DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_social_proof_total(self, table: str) -> int:
        """Get total enrollment count from a stats table."""
        valid_tables = {"social_proof_company", "social_proof_city", "social_proof_role"}
        if table not in valid_tables:
            return 0
        conn = sqlite3.connect(str(self._db_path))
        row = conn.execute(f"SELECT SUM(enrollments_count) as total FROM {table}").fetchone()
        conn.close()
        return row[0] or 0 if row else 0

    def upsert_social_proof_company(self, company_name: str, **kwargs):
        """Upsert company stats (non-blocking via write queue)."""
        kwargs["updated_at"] = datetime.now().isoformat()
        columns = list(kwargs.keys())
        values = [company_name] + [kwargs[c] for c in columns]
        placeholders = ", ".join(["?"] * len(values))
        col_list = "company_name, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = excluded.{c}" for c in columns)
        self._write_queue.put((
            f"""INSERT INTO social_proof_company ({col_list})
                VALUES ({placeholders})
                ON CONFLICT(company_name) DO UPDATE SET {conflict_updates}""",
            tuple(values)
        ))

    def upsert_social_proof_city(self, city_name: str, **kwargs):
        """Upsert city stats (non-blocking via write queue)."""
        kwargs["updated_at"] = datetime.now().isoformat()
        columns = list(kwargs.keys())
        values = [city_name] + [kwargs[c] for c in columns]
        placeholders = ", ".join(["?"] * len(values))
        col_list = "city_name, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = excluded.{c}" for c in columns)
        self._write_queue.put((
            f"""INSERT INTO social_proof_city ({col_list})
                VALUES ({placeholders})
                ON CONFLICT(city_name) DO UPDATE SET {conflict_updates}""",
            tuple(values)
        ))

    def upsert_social_proof_role(self, role_name: str, **kwargs):
        """Upsert role stats (non-blocking via write queue)."""
        kwargs["updated_at"] = datetime.now().isoformat()
        columns = list(kwargs.keys())
        values = [role_name] + [kwargs[c] for c in columns]
        placeholders = ", ".join(["?"] * len(values))
        col_list = "role_name, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = excluded.{c}" for c in columns)
        self._write_queue.put((
            f"""INSERT INTO social_proof_role ({col_list})
                VALUES ({placeholders})
                ON CONFLICT(role_name) DO UPDATE SET {conflict_updates}""",
            tuple(values)
        ))

    def get_all_social_proof(self) -> dict:
        """Return all social proof data for dashboard/API."""
        return {
            "companies": self.get_social_proof_top("social_proof_company", limit=100),
            "cities": self.get_social_proof_top("social_proof_city", limit=100),
            "roles": self.get_social_proof_top("social_proof_role", limit=100),
        }

    def delete_social_proof_company(self, company_name: str):
        """Delete company stats (non-blocking)."""
        self._write_queue.put((
            "DELETE FROM social_proof_company WHERE company_name = ? COLLATE NOCASE",
            (company_name,)
        ))

    def delete_social_proof_city(self, city_name: str):
        """Delete city stats (non-blocking)."""
        self._write_queue.put((
            "DELETE FROM social_proof_city WHERE city_name = ? COLLATE NOCASE",
            (city_name,)
        ))

    def delete_social_proof_role(self, role_name: str):
        """Delete role stats (non-blocking)."""
        self._write_queue.put((
            "DELETE FROM social_proof_role WHERE role_name = ? COLLATE NOCASE",
            (role_name,)
        ))

    def cleanup_stale(self, max_age_minutes: int = 10):
        """Clean up stale pending calls older than max_age_minutes"""
        cutoff = datetime.now().timestamp() - (max_age_minutes * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        self._write_queue.put((
            "DELETE FROM calls WHERE status = 'pending' AND created_at < ?",
            (cutoff_iso,)
        ))

    def shutdown(self):
        """Graceful shutdown"""
        self._write_queue.put(None)
        if self._writer_thread:
            self._writer_thread.join(timeout=3.0)


# Singleton instance
session_db = SessionDB()
