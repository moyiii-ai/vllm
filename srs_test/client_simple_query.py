import time
from openai import OpenAI
from transformers import AutoTokenizer

client = OpenAI(
    api_key="dummy-key",  # required by OpenAI client even for local servers
    base_url="http://localhost:8000/v1"
)

models = client.models.list()
model = models.data[0].id

tokenizer = AutoTokenizer.from_pretrained(model)
question = "Hello, I'm a simple query."

prompt = question


def send_query():
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=0.7,
        stream=True,
    )

    for chunk in completion:
        chunk_message = chunk.choices[0].text
        if chunk_message is not None:
            print(chunk_message, end="", flush=True)

    print("\n")

send_query()