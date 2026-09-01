import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.preprocessing import clean_text
from src.vector_store import QuestionVectorStore


def run_ingestion(max_samples: int = 10000, batch_size: int = 256):
    print("Fetching Quora dataset directly via Hugging Face...")
    ds = load_dataset("AlekseyKorshuk/quora-question-pairs", split="train")
    df = pd.DataFrame(ds)

    # Extract unique questions
    q1 = df["question1"].dropna().astype(str)
    q2 = df["question2"].dropna().astype(str)
    all_questions = (
        pd.concat([q1, q2]).drop_duplicates().reset_index(drop=True)
    )

    if max_samples and len(all_questions) > max_samples:
        all_questions = all_questions.iloc[:max_samples]

    print(f"Total unique questions to index: {len(all_questions)}")

    # Use PyTorch model directly for batch embedding generation
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vector_store = QuestionVectorStore()

    for i in tqdm(
        range(0, len(all_questions), batch_size), desc="Ingesting Batches"
    ):
        batch_questions = all_questions.iloc[i : i + batch_size].tolist()
        batch_ids = list(range(i, i + len(batch_questions)))

        cleaned_batch = [clean_text(q) for q in batch_questions]
        # SentenceTransformer automatically normalizes vectors for Cosine distance
        embeddings = embedder.encode(
            cleaned_batch, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        vector_store.upsert_questions(
            ids=batch_ids, questions=batch_questions, embeddings=embeddings
        )

    print(
        "\nBatch Ingestion Complete! Data is safely stored inside data/qdrant_db/"
    )


if __name__ == "__main__":
    run_ingestion(max_samples=10000, batch_size=256)