import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.embeddings import ONNXQuestionEmbedder, ReRanker
from src.preprocessing import clean_text
from src.vector_store import QuestionVectorStore

# Load ONNX Engine
embedder = ONNXQuestionEmbedder()
reranker = ReRanker()
vector_store = QuestionVectorStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="ONNX-Optimized Semantic Search Engine", lifespan=lifespan
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    rerank: bool = True


class MatchResponse(BaseModel):
    question: str
    score: float
    rerank_score: float = None


class QueryResponse(BaseModel):
    query: str
    matches: List[MatchResponse]


@app.post("/search", response_model=QueryResponse)
def find_duplicates(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400, detail="Question string cannot be empty"
        )

    cleaned_query = clean_text(request.question)
    query_vector = embedder.encode([cleaned_query])[0].tolist()

    fetch_k = max(request.top_k, 20) if request.rerank else request.top_k
    candidates = vector_store.search_similar(
        query_vector=query_vector, top_k=fetch_k
    )

    if request.rerank and candidates:
        ranked_candidates = reranker.rank(
            query=request.question, candidates=candidates
        )
        matches = ranked_candidates[: request.top_k]
    else:
        matches = candidates[: request.top_k]

    return QueryResponse(query=request.question, matches=matches)