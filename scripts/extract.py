import sqlite3
import pandas as pd
from pathlib import Path

# --- Settings ---
DB_PATH = Path("data/chat/chat.db")          # your copied database
OUTPUT_CSV = Path("data/group_chat.csv")
CHAT_ID = 1194                           # the group chat you identified

# --- Extraction ---
conn = sqlite3.connect(str(DB_PATH))

query = f"""
SELECT
    m.text,
    m.date / 1000000000 + 978307200 AS date_unix,   -- convert to Unix time
    h.id AS sender,
    c.display_name
FROM message m
JOIN handle h ON m.handle_id = h.ROWID
JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
JOIN chat c ON cmj.chat_id = c.ROWID
WHERE c.ROWID = {CHAT_ID}               -- directly using chat ID 377
ORDER BY m.date ASC
"""

df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print(f"❌ No messages found for chat_id {CHAT_ID}. Is data/chat.db up to date?")
    exit()

# Clean up columns
df['timestamp'] = pd.to_datetime(df['date_unix'], unit='s')
df = df[['timestamp', 'sender', 'text']]

# Save
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Exported {len(df)} messages to {OUTPUT_CSV}")