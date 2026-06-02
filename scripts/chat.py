# scripts/chat.py
from mlx_lm import load, generate

# ---------- CONFIG ----------
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER_PATH = "adapters/peachko"
# ----------------------------

print("🔄 Loading model + adapter (this takes a moment)...")
model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)

print("🍑 Peachko is ready. Type your message (or 'quit' to stop).")
print("    Format like the group: 'Rico: Who's bringing snacks?'\n")

messages = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Build the chat template just like training
    messages.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    print(f"Peachko: {response}\n")

    # Add the model's reply to the history for multi‑turn
    messages.append({"role": "assistant", "content": response})