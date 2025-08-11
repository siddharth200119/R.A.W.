import sqlite3
import json
from typing import List, Optional
from RAW.models import Message
from RAW.utils import Logger

class SQLiteStorage:
    def __init__(self, db_path: str = "longterm_memory.db", logger: Logger = Logger()):
        self.db_path = db_path
        self.logger = logger
        self._init_db()

    def _init_db(self):
        """Create tables if not exist"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            is_tool_call INTEGER DEFAULT 0,
            tool_name TEXT,
            tool_params TEXT,
            tool_output TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            user_id TEXT PRIMARY KEY,
            summary TEXT
        )
        """)

        conn.commit()
        conn.close()

    def store_message(
    self,
    user_id: str,
    message: Message,
    is_tool_call: int = 0,
    tool_name: Optional[str] = None,
    tool_params: Optional[str] = None,
    tool_output: Optional[str] = None
):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        content = json.dumps(message.content) if not isinstance(message.content, str) else message.content
        cur.execute(
            "INSERT INTO messages (user_id, role, content, is_tool_call, tool_name, tool_params, tool_output) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, message.role, content, is_tool_call, tool_name, tool_params, tool_output)
        )
        conn.commit()
        conn.close()
        print(f"Stored message for user {user_id}, tool_call={is_tool_call}, tool_name={tool_name}")

    def get_all_messages(self, user_id: str) -> List[Message]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY id ASC", (user_id,))
        rows = cur.fetchall()
        conn.close()
        return [Message(role=row[0], content=row[1]) for row in rows]

    def store_summary(self, user_id: str, summary: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO summaries (user_id, summary) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary
        """, (user_id, summary))
        conn.commit()
        conn.close()
        print(f"Stored/Updated summary for user {user_id}")

    def get_summary(self, user_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT summary FROM summaries WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    
    def delete_summary(self, user_id: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM summaries WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        (f"Deleted summary for user {user_id}")

