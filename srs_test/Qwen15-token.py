from transformers import AutoTokenizer
import random
import string

target_size_bytes = 382_757  

words = []
while sum(len(w) + 1 for w in words) < target_size_bytes:  # +1 for space
    word_len = random.randint(3, 10)
    word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
    words.append(word)

text = ' '.join(words)

with open("random_text.txt", "w") as f:
    f.write(text)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B", trust_remote_code=True)
tokens = tokenizer.encode(text)

print(f"Text size: {len(text.encode('utf-8'))} bytes")
print(f"Token count: {len(tokens)}")
print(f"First 200 chars: {text[:200]}...")
