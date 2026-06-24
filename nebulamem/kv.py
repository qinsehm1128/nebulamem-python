import json
import logging
from typing import Any, Optional
import lmdb

logger = logging.getLogger("nebulamem.kv")

class LmdbKVStore:
    """
    A super-fast, memory-mapped transactional Key-Value store using LMDB.
    This module contains NO fallback/mock code and is designed for high-performance production workloads.
    """
    def __init__(self, db_path: str = "./lmdb_data"):
        self.db_path = db_path
        # Map up to 10MB database mapping
        self.env = lmdb.open(self.db_path, map_size=10485760)
        logger.info(f"Native LMDB environment opened at '{self.db_path}'")

    def set(self, key: str, value: Any) -> None:
        """
        Puts a JSON-serializable value associated with a key in a fast transaction.
        """
        val_bytes = json.dumps(value).encode('utf-8')
        with self.env.begin(write=True) as txn:
            txn.put(key.encode('utf-8'), val_bytes)

    def get(self, key: str) -> Optional[Any]:
        """
        Gets and deserializes the value associated with a key.
        """
        with self.env.begin() as txn:
            val_bytes = txn.get(key.encode('utf-8'))
            if val_bytes is None:
                return None
            return json.loads(val_bytes.decode('utf-8'))

    def delete(self, key: str) -> None:
        """
        Removes a key-value record atomically.
        """
        with self.env.begin(write=True) as txn:
            txn.delete(key.encode('utf-8'))
