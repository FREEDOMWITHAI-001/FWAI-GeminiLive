"""
PostgreSQL session database for call persistence and analytics.
All writes are non-blocking (background thread) - zero latency impact on audio path.
Reads use a connection pool for fast concurrent access.
"""

import json
import threading
import queue
import time
from datetime import datetime

import psycopg2
import psycopg2.pool
import psycopg2.extras
from loguru import logger

from src.core.config import config


class SessionDB:
    def __init__(self):
        self._write_queue = queue.Queue()
        self._writer_thread = None

        if not config.database_url:
            raise RuntimeError("DATABASE_URL is not set — cannot initialise SessionDB")

        self._dsn = config.database_url
        self._pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=5, dsn=self._dsn
        )
        self._init_db()
        self._start_writer()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_read_conn(self):
        conn = self._pool.getconn()
        # Validate — discard and recreate if the connection went stale
        try:
            conn.cursor().execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            logger.warning("DB read pool: stale connection detected — replacing")
            try:
                self._pool.putconn(conn, close=True)
            except Exception:
                pass
            # Recreate the pool so minconn=1 spawns a fresh connection
            try:
                self._pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1, maxconn=5, dsn=self._dsn
                )
            except Exception as exc:
                logger.error(f"DB read pool: failed to recreate pool: {exc}")
                raise
            conn = self._pool.getconn()
        return conn

    def _put_read_conn(self, conn):
        self._pool.putconn(conn)

    def _init_db(self):
        """Create tables if they don't exist (idempotent)."""
        from pathlib import Path
        schema_path = Path(__file__).parent / "scripts" / "init_schema.sql"
        schema_sql = schema_path.read_text()

        # Use a dedicated connection so we can freely set autocommit
        # (pool connections may have an implicit transaction from the keepalive
        # SELECT 1, which prevents changing session settings mid-transaction)
        conn = psycopg2.connect(self._dsn)
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(schema_sql)
            cur.close()
        finally:
            conn.close()
        logger.info("SessionDB initialized (PostgreSQL)")

    def _connect_writer(self) -> psycopg2.extensions.connection:
        """Open a fresh writer connection with TCP keepalives enabled."""
        conn = psycopg2.connect(
            self._dsn,
            keepalives=1,
            keepalives_idle=30,       # send keepalive after 30s idle
            keepalives_interval=10,   # retry every 10s
            keepalives_count=5,       # give up after 5 retries
        )
        return conn

    def _start_writer(self):
        """Start background writer thread with automatic reconnection.

        A single persistent connection is reused for all writes (no per-write
        overhead).  If the connection is lost (server restart, idle timeout,
        firewall TCP RST), the thread reconnects and retries the failed item.
        A periodic keep-alive ping is sent during idle periods so stale
        connections are detected early.
        """
        def writer():
            conn = None
            item = None  # track the current item so we can re-queue it on error

            def reconnect():
                nonlocal conn
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
                delay = 1
                while True:
                    try:
                        conn = self._connect_writer()
                        logger.info("DB writer: (re)connected to PostgreSQL")
                        return
                    except Exception as exc:
                        logger.error(f"DB writer: reconnect failed ({exc}), retrying in {delay}s")
                        time.sleep(delay)
                        delay = min(delay * 2, 30)

            reconnect()  # initial connection

            while True:
                try:
                    item = self._write_queue.get(timeout=5.0)
                    if item is None:
                        break  # shutdown signal
                    sql, params = item
                    cur = conn.cursor()
                    cur.execute(sql, params)
                    conn.commit()
                    cur.close()
                    item = None  # successfully written — clear so we don't re-queue

                except queue.Empty:
                    # Idle — send a keep-alive ping to detect dead connections early
                    try:
                        conn.cursor().execute("SELECT 1")
                    except Exception:
                        logger.warning("DB writer: keep-alive failed — reconnecting")
                        reconnect()

                except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                    logger.error(f"DB write error (connection lost): {exc} — reconnecting")
                    if item is not None:
                        # Re-queue the failed item so it's retried after reconnect
                        self._write_queue.put(item)
                        item = None
                    reconnect()

                except Exception as exc:
                    logger.error(f"DB write error: {exc}")
                    item = None
                    try:
                        conn.rollback()
                    except Exception:
                        reconnect()

            if conn is not None:
                conn.close()

        self._writer_thread = threading.Thread(target=writer, daemon=True, name="session-db-writer")
        self._writer_thread.start()

    # ==================================================================
    # Calls
    # ==================================================================

    def create_call(self, call_uuid: str, phone: str, contact_name: str = "Customer",
                    client_name: str = "fwai", webhook_url: str = None, total_questions: int = 0):
        """Queue a new call record (non-blocking)."""
        self._write_queue.put((
            '''INSERT INTO calls
               (call_uuid, phone, contact_name, client_name, status, webhook_url, total_questions, created_at)
               VALUES (%s, %s, %s, %s, 'pending', %s, %s, %s)
               ON CONFLICT (call_uuid) DO UPDATE SET
                 phone = EXCLUDED.phone,
                 contact_name = EXCLUDED.contact_name,
                 client_name = EXCLUDED.client_name,
                 status = 'pending',
                 webhook_url = EXCLUDED.webhook_url,
                 total_questions = EXCLUDED.total_questions,
                 created_at = EXCLUDED.created_at''',
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
                sets.append(f"{key} = %s")
                params.append(value)
        if not sets:
            return
        params.append(call_uuid)
        self._write_queue.put((
            f"UPDATE calls SET {', '.join(sets)} WHERE call_uuid = %s",
            tuple(params)
        ))

    def get_call(self, call_uuid: str) -> dict:
        """Read call record (blocking, uses connection pool)."""
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM calls WHERE call_uuid = %s", (call_uuid,))
            row = cur.fetchone()
            cur.close()
        finally:
            self._put_read_conn(conn)
        if row:
            result = dict(row)
            for field in ('collected_responses', 'objections_raised'):
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return result
        return None

    def get_recent_calls(self, limit: int = 50) -> list:
        """Get recent calls for dashboard/API."""
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT call_uuid, phone, contact_name, client_name, status, duration_seconds, "
                "questions_completed, total_questions, interest_level, call_summary, created_at "
                "FROM calls ORDER BY created_at DESC LIMIT %s", (limit,)
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            self._put_read_conn(conn)
        return [dict(row) for row in rows]

    def finalize_call(self, call_uuid: str, status: str = "completed", ended_at: datetime = None, duration_seconds: float = None):
        """Finalize a call record with end time, duration, and status (non-blocking)."""
        fields = {"status": status}
        if ended_at:
            fields["ended_at"] = ended_at.isoformat()
        if duration_seconds is not None:
            fields["duration_seconds"] = duration_seconds
        
        if not fields:
            return
        
        sets = []
        params = []
        for key, value in fields.items():
            sets.append(f"{key} = %s")
            params.append(value)
        params.append(call_uuid)
        
        self._write_queue.put((
            f"UPDATE calls SET {', '.join(sets)} WHERE call_uuid = %s",
            tuple(params)
        ))

    # ==================================================================
    # Cross-Call Memory
    # ==================================================================

    def get_contact_memory(self, phone: str) -> dict:
        """Load contact memory by phone number."""
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM contact_memory WHERE phone = %s", (phone,))
            row = cur.fetchone()
            cur.close()
        finally:
            self._put_read_conn(conn)
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
        columns = list(fields.keys())
        values = [phone] + [fields[c] for c in columns]
        placeholders = ", ".join(["%s"] * len(values))
        col_list = "phone, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
        self._write_queue.put((
            f"""INSERT INTO contact_memory ({col_list})
                VALUES ({placeholders})
                ON CONFLICT(phone) DO UPDATE SET {conflict_updates}""",
            tuple(values)
        ))

    def get_all_contact_memories(self, limit: int = 100) -> list:
        """List all contact memories for dashboard/API."""
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT phone, name, persona, company, role, call_count, "
                "last_call_date, last_call_outcome, updated_at "
                "FROM contact_memory ORDER BY updated_at DESC LIMIT %s", (limit,)
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            self._put_read_conn(conn)
        return [dict(row) for row in rows]

    def delete_contact_memory(self, phone: str):
        """Delete contact memory for a phone number (non-blocking)."""
        self._write_queue.put((
            "DELETE FROM contact_memory WHERE phone = %s", (phone,)
        ))

    # ==================================================================
    # Social Proof Stats
    # ==================================================================

    def get_social_proof_by_company(self, company_name: str) -> dict:
        """Look up company enrollment stats."""
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT * FROM social_proof_company WHERE LOWER(company_name) = LOWER(%s)",
                (company_name,)
            )
            row = cur.fetchone()
            cur.close()
        finally:
            self._put_read_conn(conn)
        return dict(row) if row else None

    def get_social_proof_by_city(self, city_name: str) -> dict:
        """Look up city enrollment stats."""
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT * FROM social_proof_city WHERE LOWER(city_name) = LOWER(%s)",
                (city_name,)
            )
            row = cur.fetchone()
            cur.close()
        finally:
            self._put_read_conn(conn)
        return dict(row) if row else None

    def get_social_proof_by_role(self, role_name: str) -> dict:
        """Look up role enrollment stats."""
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT * FROM social_proof_role WHERE LOWER(role_name) = LOWER(%s)",
                (role_name,)
            )
            row = cur.fetchone()
            cur.close()
        finally:
            self._put_read_conn(conn)
        return dict(row) if row else None

    def get_social_proof_top(self, table: str, limit: int = 5) -> list:
        """Get top stats by enrollment count for summary generation."""
        valid_tables = {"social_proof_company", "social_proof_city", "social_proof_role"}
        if table not in valid_tables:
            return []
        conn = self._get_read_conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                f"SELECT * FROM {table} ORDER BY enrollments_count DESC LIMIT %s",
                (limit,)
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            self._put_read_conn(conn)
        return [dict(row) for row in rows]

    def get_social_proof_total(self, table: str) -> int:
        """Get total enrollment count from a stats table."""
        valid_tables = {"social_proof_company", "social_proof_city", "social_proof_role"}
        if table not in valid_tables:
            return 0
        conn = self._get_read_conn()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT SUM(enrollments_count) as total FROM {table}")
            row = cur.fetchone()
            cur.close()
        finally:
            self._put_read_conn(conn)
        return row[0] or 0 if row else 0

    def upsert_social_proof_company(self, company_name: str, **kwargs):
        """Upsert company stats (non-blocking via write queue)."""
        kwargs["updated_at"] = datetime.now().isoformat()
        columns = list(kwargs.keys())
        values = [company_name] + [kwargs[c] for c in columns]
        placeholders = ", ".join(["%s"] * len(values))
        col_list = "company_name, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
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
        placeholders = ", ".join(["%s"] * len(values))
        col_list = "city_name, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
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
        placeholders = ", ".join(["%s"] * len(values))
        col_list = "role_name, " + ", ".join(columns)
        conflict_updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
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
            "DELETE FROM social_proof_company WHERE LOWER(company_name) = LOWER(%s)",
            (company_name,)
        ))

    def delete_social_proof_city(self, city_name: str):
        """Delete city stats (non-blocking)."""
        self._write_queue.put((
            "DELETE FROM social_proof_city WHERE LOWER(city_name) = LOWER(%s)",
            (city_name,)
        ))

    def delete_social_proof_role(self, role_name: str):
        """Delete role stats (non-blocking)."""
        self._write_queue.put((
            "DELETE FROM social_proof_role WHERE LOWER(role_name) = LOWER(%s)",
            (role_name,)
        ))

    # ==================================================================
    # Maintenance
    # ==================================================================

    def cleanup_stale(self, max_age_minutes: int = 10):
        """Clean up stale pending calls older than max_age_minutes."""
        cutoff = datetime.now().timestamp() - (max_age_minutes * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        self._write_queue.put((
            "DELETE FROM calls WHERE status = 'pending' AND created_at < %s",
            (cutoff_iso,)
        ))

    def shutdown(self):
        """Graceful shutdown."""
        self._write_queue.put(None)
        if self._writer_thread:
            self._writer_thread.join(timeout=3.0)
        if self._pool:
            self._pool.closeall()


# Singleton instance
session_db = SessionDB()
