import time
from openai import OpenAI
from transformers import AutoTokenizer

client = OpenAI(
    api_key="dummy-key",  # required by OpenAI client even for local servers
    base_url="http://localhost:8000/v1"
)

models = client.models.list()
model = models.data[0].id

long_context = ""
with open("random_text.txt", "r") as f:
    long_context = f.read()

# a truncation of the long context for the --max-model-len
# if you increase the --max-model-len, you can decrease the truncation i.e.
# use more of the long context
long_context = long_context[:15000]

tokenizer = AutoTokenizer.from_pretrained(model)
question = "Summarize the text in 2 sentences."

prompt = f"{long_context}\n\n{question}"

print(f"Number of tokens in prompt: {len(tokenizer.encode(prompt))}")

def query_and_measure_ttft():
    start = time.perf_counter()
    ttft = None

    completion = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=0.7,
        stream=True,
    )

    for chunk in completion:
        chunk_message = chunk.choices[0].text
        if chunk_message is not None:
            if ttft is None:
                ttft = time.perf_counter()
            print(chunk_message, end="", flush=True)

    print("\n")  # New line after streaming
    return ttft - start

print("Querying vLLM server with cold LMCache CPU Offload")
cold_ttft = query_and_measure_ttft()
print(f"Cold TTFT: {cold_ttft:.3f} seconds")

print("\nQuerying vLLM server with warm LMCache CPU Offload")
warm_ttft = query_and_measure_ttft()
print(f"Warm TTFT: {warm_ttft:.3f} seconds")

print(f"\nTTFT Improvement: {(cold_ttft - warm_ttft):.3f} seconds \
    ({(cold_ttft/warm_ttft):.1f}x faster)")