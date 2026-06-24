"""Content store with on-demand loading.

Node *content* (the text shown to the LLM) is kept out of the hot path. In
memory mode it is a dict; in sqlite mode it is a single local file and content
is fetched lazily — only fired nodes are ever materialized. `loads` counts how
many content fetches happened, so a benchmark can prove that a query over a huge
corpus only touches a tiny working set ("按需加载").
"""
import sqlite3
from typing import Dict, Optional


class NodeStore:
    def __init__(self, db_path: Optional[str] = None):
        self.loads = 0
        self._mem: Dict[str, str] = {}
        self._conn: Optional[sqlite3.Connection] = None
        if db_path:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("CREATE TABLE IF NOT EXISTS content (id TEXT PRIMARY KEY, body TEXT)")
            self._conn.commit()

    def put(self, node_id: str, content: str) -> None:
        if self._conn is not None:
            self._conn.execute("INSERT OR REPLACE INTO content (id, body) VALUES (?, ?)",
                               (node_id, content))
        else:
            self._mem[node_id] = content

    def commit(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    def get(self, node_id: str) -> Optional[str]:
        """Lazy fetch. Increments the on-demand load counter."""
        self.loads += 1
        if self._conn is not None:
            row = self._conn.execute("SELECT body FROM content WHERE id = ?", (node_id,)).fetchone()
            return row[0] if row else None
        return self._mem.get(node_id)

    def reset_counter(self) -> None:
        self.loads = 0
