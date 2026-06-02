# scripts/train.py
import subprocess
import sys
from pathlib import Path

# ---------- CONFIG ----------
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DATA_FILE = Path("data/train.jsonl").resolve()
ADAPTER_PATH = Path("adapters/peachko").resolve()
ITERS = 1000          # more iterations can capture style better, adjust as needed
BATCH_SIZE = 2        # small batch for memory
# ----------------------------

def train():
    print(f"🚀 Starting LoRA fine‑tuning on {len(open(DATA_FILE).readlines())} examples...")
    cmd = [
        "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--data", str(DATA_FILE),
        "--train",
        "--iters", str(ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--adapter-path", str(ADAPTER_PATH),
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Training complete. Adapter saved to {ADAPTER_PATH}")

if __name__ == "__main__":
    train()