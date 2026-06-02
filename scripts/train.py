# scripts/train.py
import subprocess, json, os, random
from pathlib import Path
from mlx_lm import load

# ---------- CONFIG ----------
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
RAW_DATA = Path("data/train.jsonl")          # from clean.py
TRAIN_DIR = Path("data/train_data")          # MLX expects a directory
TRAIN_FILE = TRAIN_DIR / "train.jsonl"
VALID_FILE = TRAIN_DIR / "valid.jsonl"
ADAPTER_PATH = Path("adapters/peachko")
ITERS = 1000
BATCH_SIZE = 2
VALID_SPLIT = 0.05                           # 5% for validation
# ----------------------------

def prepare_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"❌ {RAW_DATA} not found. Run clean.py first.")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📝 Loading tokenizer from {BASE_MODEL}...")
    model, tokenizer = load(BASE_MODEL)

    # Convert all examples to text
    all_texts = []
    with open(RAW_DATA, 'r', encoding='utf-8') as fin:
        for line in fin:
            ex = json.loads(line)
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            all_texts.append(text)

    print(f"🔄 Converting {len(all_texts)} examples...")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_texts)
    n_valid = max(1, int(len(all_texts) * VALID_SPLIT))
    valid_data = all_texts[:n_valid]
    train_data = all_texts[n_valid:]

    # Write train.jsonl
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for text in train_data:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

    # Write valid.jsonl
    with open(VALID_FILE, 'w', encoding='utf-8') as f:
        for text in valid_data:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

    print(f"✅ Created {TRAIN_FILE} ({len(train_data)} examples)")
    print(f"✅ Created {VALID_FILE} ({len(valid_data)} examples)")

def train():
    if not (TRAIN_FILE.exists() and VALID_FILE.exists()):
        prepare_data()

    print(f"🚀 Starting LoRA fine‑tuning on {TRAIN_DIR}...")
    cmd = [
        "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--data", str(TRAIN_DIR.resolve()),
        "--train",
        "--iters", str(ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--adapter-path", str(ADAPTER_PATH.resolve()),
    ]
    print(f"   Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"✅ Training complete! Adapter saved to {ADAPTER_PATH}")

if __name__ == "__main__":
    train()