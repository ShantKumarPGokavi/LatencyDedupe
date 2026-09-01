import os
from pathlib import Path
from typing import Any, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "qdrant_db"


class QuestionVectorStore:

    def __init__(
        self,
        collection_name: str = "quora_questions",
        path: str = str(DEFAULT_DB_PATH),
    ):
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self._init_collection()

    def _init_collection(self, vector_size: int = 384):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )

    def upsert_questions(
        self,
        ids: List[int],
        questions: List[str],
        embeddings: List[List[float]],
    ):
        points = [
            PointStruct(id=idx, vector=vector, payload={"question": q})
            for idx, q, vector in zip(ids, questions, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_similar(
        self, query_vector: List[float], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )
        results = []
        for res in response.points:
            # Safe payload key extraction
            q_text = res.payload.get("question", "Unknown Question")
            results.append({"question": q_text, "score": round(res.score, 4)})
        return results