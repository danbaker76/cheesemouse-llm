import pandas as pd

# Load your exported data
df = pd.read_csv("data/group_chat.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Basic overview
print("=" * 50)
print("OVERVIEW")
print("=" * 50)
print(f"Total messages: {len(df)}")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Unique senders: {df['sender'].nunique()}")
print()

# 2. Messages per sender
print("=" * 50)
print("MESSAGES PER SENDER")
print("=" * 50)
sender_counts = df['sender'].value_counts()
print(sender_counts)
print()

# 3. Sample messages from each sender (top 3)
print("=" * 50)
print("SAMPLE MESSAGES (up to 3 per sender)")
print("=" * 50)
for sender, group in df.groupby('sender'):
    print(f"\n--- {sender} ({len(group)} messages) ---")
    for i, (_, row) in enumerate(group.head(3).iterrows()):
        text = str(row['text']) if pd.notna(row['text']) else "[empty]"
        # Truncate for readability
        print(f"  [{i+1}] {text[:120]}")
print()

# 4. Null/empty texts
empty_texts = df['text'].isna().sum() + (df['text'] == '').sum()
print(f"Empty/null messages: {empty_texts}")

# 5. Time distribution
df['month'] = df['timestamp'].dt.to_period('M')
print("\nMessages per month:")
print(df['month'].value_counts().sort_index())