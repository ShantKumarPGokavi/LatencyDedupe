import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILE_PATH = PROJECT_ROOT / "data" / "onnx_model" / "model.onnx"

def export_and_benchmark():
    ONNX_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    # Create dummy inputs for ONNX tracing
    dummy_text = ["How do I start learning software engineering?"]
    inputs = tokenizer(dummy_text, padding=True, truncation=True, return_tensors="pt")

    print(f"Exporting PyTorch model to ONNX: {ONNX_FILE_PATH}")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(ONNX_FILE_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        dynamo=False,  # Uses the stable TorchScript ONNX exporter
    )
    print("Export Complete!")

    # Benchmark ONNX Runtime Session
    print("\nRunning Inference Benchmark using ONNX Runtime...")
    session = ort.InferenceSession(str(ONNX_FILE_PATH), providers=["CPUExecutionProvider"])
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }

    # Warmup
    _ = session.run(None, onnx_inputs)

    # Benchmark 50 iterations
    start = time.perf_counter()
    for _ in range(50):
        _ = session.run(None, onnx_inputs)
    onnx_time = (time.perf_counter() - start) / 50 * 1000

    print(f"Average ONNX CPU Inference Time: {onnx_time:.2f} ms per query")

if __name__ == "__main__":
    export_and_benchmark()