# scripts/clean.py
import pandas as pd
import json
from pathlib import Path

# ---------- CONFIG ----------
INPUT_CSV = Path("data/group_chat.csv")
OUTPUT_JSONL = Path("data/train.jsonl")
PHONE_MAP_FILE = Path("data/phone_map.json")
TARGET = "Peachko"                 # who the model will speak as
CONTEXT_LENGTH = 6                 # how many previous messages to include as user prompt
# ----------------------------

# 1. Name mapping (kept out of source control, see data/phone_map.json)
with open(PHONE_MAP_FILE, encoding='utf-8') as f:
    PHONE_MAP = json.load(f)

# 2. Load data
df = pd.read_csv(INPUT_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Loaded {len(df)} messages")

# 3. Map senders to names
df['sender_name'] = df['sender'].astype(str).map(PHONE_MAP)
unknown = df['sender_name'].isna().sum()
if unknown > 0:
    print(f"⚠️  {unknown} messages from unknown senders (not in mapping). These will be dropped.")
    df = df.dropna(subset=['sender_name'])
print(f"After mapping: {len(df)} messages")

# 4. Remove system messages (common patterns)
system_phrases = [
    "left the chat",
    "added",
    "removed",
    "changed the group photo",
    "named the group",
    "turned on",
    "turned off",
    "encryption",
    "you are now",
    "started a call",
    "is now",
    "joined the conversation",
    "message from",
]
def is_system(text):
    if not isinstance(text, str):
        return True   # drop non-strings (shouldn't happen)
    low = text.lower()
    return any(phrase in low for phrase in system_phrases)

df = df[~df['text'].apply(is_system)]
print(f"After removing system messages: {len(df)}")

# 5. Remove empty texts
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != '']
print(f"After removing empty texts: {len(df)}")

# 6. Sort by time (should already be, but just in case)
df = df.sort_values('timestamp')

# 7. Build conversation examples
examples = []
# Iterate over all messages
for i, row in df.iterrows():
    if row['sender_name'] != TARGET:
        continue   # we only create examples where Peachko is the responder

    # Find the preceding messages (up to CONTEXT_LENGTH)
    # We use all messages before this one (chronologically)
    prev = df[df['timestamp'] < row['timestamp']].tail(CONTEXT_LENGTH)

    # Build the user prompt as a chat script
    user_lines = []
    for _, p_row in prev.iterrows():
        speaker = p_row['sender_name']
        text = p_row['text'].strip()
        user_lines.append(f"{speaker}: {text}")

    if not user_lines:
        # skip if no context (e.g., first message of the whole chat)
        continue

    user_prompt = "\n".join(user_lines)
    assistant_reply = row['text'].strip()

    # Create the JSON structure for MLX chat template
    example = {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_reply}
        ]
    }
    examples.append(example)

print(f"Generated {len(examples)} training examples")

# 8. Save as JSONL
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"✅ Training data saved to {OUTPUT_JSONL}")