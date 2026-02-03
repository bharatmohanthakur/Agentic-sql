"""
Base Memory Store Interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..manager import Memory, MemoryType


class MemoryStore(ABC):
    """Abstract base class for memory storage backends"""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the store"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the store"""
        pass

    @abstractmethod
    async def store(self, memory: Memory) -> bool:
        """Store a memory"""
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        **kwargs,
    ) -> List[Memory]:
        """Retrieve memories matching the query"""
        pass

    @abstractmethod
    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory by ID"""
        pass

    @abstractmethod
    async def update(self, memory: Memory) -> bool:
        """Update an existing memory"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return total number of memories"""
        pass

    async def health_check(self) -> bool:
        """Check if store is healthy"""
        try:
            await self.count()
            return True
        except Exception:
            return False
