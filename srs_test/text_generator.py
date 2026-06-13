from transformers import AutoTokenizer
import random
import string

# Ensure token count < 131072, the max sequence length for meta-llama/Llama-3.1-8B
target_size_bytes = 250_000  

words = []
while sum(len(w) + 1 for w in words) < target_size_bytes:  # +1 for space
    word_len = random.randint(3, 10)
    word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
    words.append(word)

text = ' '.join(words)

with open("random_text.txt", "w") as f:
    f.write(text)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokens = tokenizer.encode(text)

print(f"Text size: {len(text.encode('utf-8'))} bytes")
print(f"Token count: {len(tokens)}")
print(f"First 200 chars: {text[:200]}...")
