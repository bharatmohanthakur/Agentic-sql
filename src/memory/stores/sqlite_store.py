"""
SQLite Memory Store Implementation

SQLite provides a lightweight, file-based storage option that requires no external server.
This is the default memory store for simple deployments.

Usage:
    from memory.stores import SQLiteMemoryStore

    store = SQLiteMemoryStore(
        db_path="./memories.db",
    )
    await store.connect()
"""

import json
import logging
import aiosqlite
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from .base import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class SQLiteConfig:
    """Configuration for SQLite storage"""
    db_path: str = "./agentic_sql_memories.db"


class SQLiteMemoryStore(MemoryStore):
    """
    SQLite Memory Store for lightweight, file-based memory storage.

    Features:
    - No external server required
    - File-based persistence
    - Full-text search support
    - Simple deployment
    """

    def __init__(self, config: SQLiteConfig):
        self.config = config
        self._connection = None

    async def connect(self) -> None:
        """Connect to SQLite database and create tables"""
        try:
            self._connection = await aiosqlite.connect(self.config.db_path)

            # Enable WAL mode for better concurrent access
            await self._connection.execute("PRAGMA journal_mode=WAL")

            # Create memories table
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    entity_id TEXT,
                    process_id TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    relevance_score REAL DEFAULT 0.5,
                    embedding TEXT
                )
            """)

            # Create FTS5 virtual table for full-text search
            await self._connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='rowid'
                )
            """)

            # Create triggers to keep FTS in sync
            await self._connection.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
                END
            """)

            await self._connection.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', OLD.rowid, OLD.content);
                END
            """)

            await self._connection.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', OLD.rowid, OLD.content);
                    INSERT INTO memories_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
                END
            """)

            # Create indexes
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)"
            )
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity_id)"
            )
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)"
            )
            await self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)"
            )

            await self._connection.commit()
            logger.info(f"Connected to SQLite at {self.config.db_path}")

        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    async def disconnect(self) -> None:
        """Close SQLite connection"""
        if self._connection:
            await self._connection.close()

    async def store(self, memory: "Memory") -> bool:
        """Store memory in SQLite"""
        try:
            embedding_json = json.dumps(memory.embedding) if memory.embedding else None
            metadata_json = json.dumps(memory.metadata) if memory.metadata else "{}"

            await self._connection.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, content, type, priority, entity_id, process_id, session_id,
                 metadata, created_at, access_count, relevance_score, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(memory.id),
                    memory.content,
                    memory.type.value,
                    memory.priority.value,
                    memory.entity_id,
                    memory.process_id,
                    memory.session_id,
                    metadata_json,
                    memory.created_at.isoformat(),
                    memory.access_count,
                    memory.relevance_score,
                    embedding_json,
                )
            )
            await self._connection.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to store memory in SQLite: {e}")
            return False

    async def retrieve(
        self,
        query: str,
        memory_type: Optional["MemoryType"] = None,
        limit: int = 10,
        use_fts: bool = True,
        **kwargs,
    ) -> List["Memory"]:
        """
        Retrieve memories using full-text search or LIKE query.

        Args:
            query: Text query for search
            memory_type: Filter by memory type
            limit: Max results to return
            use_fts: Use FTS5 full-text search (default True)
        """
        from ..manager import Memory, MemoryType, MemoryPriority

        try:
            if use_fts and query:
                # Use FTS5 for better text search
                base_query = """
                    SELECT m.id, m.content, m.type, m.priority, m.entity_id,
                           m.process_id, m.session_id, m.metadata, m.created_at,
                           m.access_count, m.relevance_score, m.embedding
                    FROM memories m
                    JOIN memories_fts fts ON m.rowid = fts.rowid
                    WHERE memories_fts MATCH ?
                """
                params = [query]
            else:
                # Fallback to LIKE query
                base_query = """
                    SELECT id, content, type, priority, entity_id,
                           process_id, session_id, metadata, created_at,
                           access_count, relevance_score, embedding
                    FROM memories
                    WHERE content LIKE ?
                """
                params = [f"%{query}%"]

            if memory_type:
                base_query += " AND type = ?"
                params.append(memory_type.value)

            base_query += " ORDER BY relevance_score DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await self._connection.execute(base_query, params)
            rows = await cursor.fetchall()

            memories = []
            for row in rows:
                metadata = json.loads(row[7]) if row[7] else {}
                embedding = json.loads(row[11]) if row[11] else None

                memories.append(Memory(
                    id=UUID(row[0]),
                    content=row[1],
                    type=MemoryType(row[2]),
                    priority=MemoryPriority(row[3]),
                    entity_id=row[4],
                    process_id=row[5],
                    session_id=row[6],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row[8]),
                    access_count=row[9],
                    relevance_score=row[10],
                    embedding=embedding,
                ))

            return memories

        except Exception as e:
            logger.error(f"Failed to search SQLite: {e}")
            return []

    async def delete(self, memory_id: UUID) -> bool:
        """Delete memory by ID"""
        try:
            await self._connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (str(memory_id),)
            )
            await self._connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete from SQLite: {e}")
            return False

    async def update(self, memory: "Memory") -> bool:
        """Update memory (upsert)"""
        return await self.store(memory)

    async def count(self) -> int:
        """Get total memory count"""
        try:
            cursor = await self._connection.execute(
                "SELECT COUNT(*) FROM memories"
            )
            row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    async def get_by_type(
        self,
        memory_type: "MemoryType",
        limit: int = 100,
    ) -> List["Memory"]:
        """Get all memories of a specific type"""
        from ..manager import Memory, MemoryType, MemoryPriority

        try:
            cursor = await self._connection.execute(
                """
                SELECT id, content, type, priority, entity_id,
                       process_id, session_id, metadata, created_at,
                       access_count, relevance_score, embedding
                FROM memories
                WHERE type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (memory_type.value, limit)
            )
            rows = await cursor.fetchall()

            memories = []
            for row in rows:
                metadata = json.loads(row[7]) if row[7] else {}
                embedding = json.loads(row[11]) if row[11] else None

                memories.append(Memory(
                    id=UUID(row[0]),
                    content=row[1],
                    type=MemoryType(row[2]),
                    priority=MemoryPriority(row[3]),
                    entity_id=row[4],
                    process_id=row[5],
                    session_id=row[6],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row[8]),
                    access_count=row[9],
                    relevance_score=row[10],
                    embedding=embedding,
                ))

            return memories

        except Exception as e:
            logger.error(f"Failed to get memories by type: {e}")
            return []

    async def clear_all(self) -> bool:
        """Clear all memories (use with caution)"""
        try:
            await self._connection.execute("DELETE FROM memories")
            await self._connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False
