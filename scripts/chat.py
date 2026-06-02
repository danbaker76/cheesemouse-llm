# scripts/chat.py
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit",
                        adapter_path="adapters/peachko")

messages = []
print("Chat with Peachko (type 'quit' to stop)")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    messages.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    print(f"Peachko: {response}")
    messages.append({"role": "assistant", "content": response})