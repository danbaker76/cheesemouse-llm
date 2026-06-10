# 🧀 cheesemouse-llm – Full Training Guide

> Fine‑tune an LLM on your iMessage group chat – completely private, on Apple Silicon.

This guide walks you through every step: extracting your iMessage database, identifying the group chat, cleaning the data, fine‑tuning a model with LoRA, and chatting with your clone. All processing stays on your Mac.

---

## Prerequisites

- **Mac with Apple Silicon** (M1, M2, M3) and at least **16 GB of RAM**
- **Python 3.9+** (and `pip`)
- **iMessage data** synced via **Messages in iCloud** (or an iPhone backup)
- **Full Disk Access** granted to your terminal (System Settings → Privacy & Security → Full Disk Access)

---

## 1. Set up the project

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/danbaker76/cheesemouse-llm.git
cd cheesemouse-llm
python3 -m venv cheesemouse-venv
source cheesemouse-venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Copy your iMessage database

If **Messages in iCloud** is enabled on your Mac, the database already exists locally. Copy it to the project’s data folder to avoid permission issues later:

```bash
cp ~/Library/Messages/chat.db data/chat.db
```

If you don’t use iCloud sync, you’ll need to extract `chat.db` from an iPhone backup (e.g., using iMazing or iExplorer) and place it in `data/chat.db`.

---

## 3. Find your group chat ID

Group names change, but the **chat_id** is stable. We identify it using the participants.

Open a terminal and run:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/chat.db')
cursor = conn.execute('''
    SELECT c.ROWID AS chat_id, c.display_name, h.id AS handle
    FROM chat c
    JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
    JOIN handle h ON chj.handle_id = h.ROWID
    ORDER BY chat_id, handle;
''')
for row in cursor.fetchall():
    print(f'chat_id={row[0]}, name={row[1]}, handle={row[2]}')
conn.close()
"
```

Look for the list of phone numbers / emails that belong to your group. Note the **`chat_id`** (e.g., `377`).

---

## 4. Extract the messages

Open `scripts/extract.py` and set:

```python
CHAT_ID = 999   # replace with your actual chat ID
```

Run the extraction:

```bash
python scripts/extract.py
```

This creates `data/group_chat.csv` with columns: `timestamp`, `sender`, `text`.

---

## 5. (Optional) Quick data exploration

See who sent how many messages:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/group_chat.csv')
print(df['sender'].value_counts())
print('Earliest message:', df['timestamp'].min())
"
```

---

## 6. Map phone numbers to names

`scripts/clean.py` loads its sender mapping from `data/phone_map.json` (this file is gitignored, so your real phone numbers never get committed).

Create `data/phone_map.json` with the exact sender strings from your CSV (often without a leading `+`):

```json
{
    "15555555555": "User1",
    "16666666666": "User2"
}
```
etc.

---

## 7. Choose your target speaker

In the same `clean.py`, set the person you want to clone:

```python
TARGET = "Peachko"
```

You can also adjust `CONTEXT_LENGTH` – the number of previous chat lines the model sees before replying (default is `6`).

---

## 8. Generate training data

```bash
python scripts/clean.py
```

This produces `data/train.jsonl` – each line contains a `messages` array with:
- `user` – the preceding group messages (with speaker names)
- `assistant` – the target speaker’s reply

---

## 9. Fine‑tune the model (LoRA)

Training runs entirely on your Mac using Apple’s MLX framework. No data leaves your machine.

```bash
python scripts/train.py
```

What happens:
1. Downloads the base model (~2 GB) if not cached.
2. Converts your training data to MLX format.
3. Splits a small validation set.
4. Runs 1,000 LoRA iterations (adjustable inside `train.py`).

On an M1 Mac this takes **1–3 hours**. The adapter is saved in `adapters/peachko/`.

---

## 10. Chat with your clone

```bash
python scripts/chat.py
```

Type messages in the same format used during training – start with the speaker’s name:

```
Rico: Who's bringing snacks tonight?
```

The model replies as your chosen target. Type `quit` to end the chat.

---

## Customisation

| What to change | Where |
|----------------|-------|
| Target speaker | `TARGET` in `clean.py` |
| Context window (messages) | `CONTEXT_LENGTH` in `clean.py` |
| Training iterations | `ITERS` in `train.py` |
| Base model | `BASE_MODEL` in `train.py` and `chat.py` |
| Batch size (memory) | `BATCH_SIZE` in `train.py` |

---

## Project structure after all steps

```
cheesemouse-llm/
├── data/
│   ├── chat.db                  (ignored – your raw database)
│   ├── phone_map.json           (ignored – your phone number → name mapping)
│   ├── group_chat.csv
│   ├── train.jsonl
│   └── train_data/
│       ├── train.jsonl
│       └── valid.jsonl
├── adapters/
│   └── peachko/                 (ignored – trained weights)
├── scripts/
│   ├── extract.py
│   ├── clean.py
│   ├── train.py
│   └── chat.py
├── requirements.txt
└── README.md
```

---

## Privacy and safety

- **Never commit `data/chat.db` or `data/phone_map.json`** – they contain raw private conversations and real phone numbers, and are automatically ignored by `.gitignore`.
- **Never share the adapter weights** (`adapters/peachko/`) – they may leak real names and conversation snippets.
- Always **inform your group members** before training a model on their messages.
- Keep the model local – do **not** deploy it as a public bot.

---

## Troubleshooting

**“Operation not permitted” during extraction**  
→ Grant Full Disk Access to your terminal (System Settings → Privacy & Security → Full Disk Access).

**“Training set not found or empty”**  
→ Ensure `data/train_data/train.jsonl` and `valid.jsonl` exist. Run `clean.py` then `train.py` again.

**Out of memory**  
→ Reduce `BATCH_SIZE` in `train.py` to `1`, or switch to a smaller model (e.g., `Llama-3.2-1B-Instruct-4bit`).

**Model responses feel generic**  
→ Increase `ITERS` (e.g., 2000), or add more messages. Set your iPhone’s “Keep Messages” to **Forever** to preserve full history.

---

## Example chat session

```
$ python scripts/chat.py
🔄 Loading model + adapter...
🍑 Peachko is ready. Type your message (or 'quit' to stop).

You: Dank: Anyone around?
Peachko: Always 😏 what's up?
You: Mass: Taco Bell run?
Peachko: I'm in if you grab me a Baja Blast
```

---

Enjoy your private, fine‑tuned group chat clone!
```