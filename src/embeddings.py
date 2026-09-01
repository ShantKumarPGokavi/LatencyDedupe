from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import onnxruntime as ort
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer

DEFAULT_ONNX_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "onnx_model"
    / "model.onnx"
)


class ONNXQuestionEmbedder:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        onnx_path: Path = DEFAULT_ONNX_PATH,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model file not found at {onnx_path}. Run src/export_onnx.py first."
            )
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded, token_embeddings.shape
        )
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(
            input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None
        )
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="np"
        )
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, onnx_inputs)
        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

        # L2 Normalize vectors for Cosine Similarity
        norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)


class ReRanker:

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, item["question"]] for item in candidates]
        scores = self.model.predict(pairs)

        for item, score in zip(candidates, scores):
            item["rerank_score"] = float(round(score, 4))

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)