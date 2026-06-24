import logging
from typing import List, Dict, Any
import zvec

logger = logging.getLogger("nebulamem.vector")

class ZvecVectorStore:
    """
    A pure, production-grade Python wrapper around Alibaba's 'zvec' embedded vector database.
    This module contains NO fallback/mock code and is designed to run directly in enterprise production
    environments with native '@zvec/zvec' binary packages.
    """
    def __init__(self, db_path: str = "./zvec_data", dimension: int = 64):
        self.db_path = db_path
        self.dimension = dimension
        
        # Setup Zvec Schema
        schema = zvec.CollectionSchema(
            name="nebulamem_vectors",
            fields=[
                zvec.FieldSchema(name="id", type=zvec.DataType.STRING, is_primary_key=True),
                zvec.FieldSchema(name="content", type=zvec.DataType.STRING),
                zvec.FieldSchema(name="node_type", type=zvec.DataType.STRING),
                zvec.FieldSchema(name="embedding", type=zvec.DataType.FLOAT_VECTOR, dimension=self.dimension, metric_type=zvec.MetricType.COSINE)
            ]
        )
        # Open or Create Zvec collection (single file WAL-persisted on-disk database)
        self.zvec_collection = zvec.ZVecOpen(self.db_path, schema)
        logger.info(f"Native Zvec Vector collection opened at '{self.db_path}'")

    def insert(self, node_id: str, embedding: List[float], content: str, node_type: str) -> None:
        """
        Inserts or updates a vector record in Zvec.
        """
        if len(embedding) != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
            
        self.zvec_collection.insert(data={
            "id": node_id,
            "embedding": embedding,
            "content": content,
            "node_type": node_type
        })

    def delete(self, node_id: str) -> None:
        """
        Removes a vector record from the HNSW index.
        """
        self.zvec_collection.delete(node_id)

    def search(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs Approximate Nearest Neighbor (ANN) or exact Cosine Similarity search.
        Returns a list of dictionaries containing 'id' and 'similarity'.
        """
        results = self.zvec_collection.search(vector=query_embedding, limit=limit)
        # Map Zvec results into standardized format
        return [{"id": r["id"], "similarity": r["score"]} for r in results]
