from verusdb.settings import Settings
from verusdb.client import VerusClient
import os
from verusdb.embeddings.openai import OpenAIEmbeddingsEngine


settings = Settings(
    engine="postgres",
    postgres={
        "host": "localhost",
        "port": 5432,
        "db": "verus",
        "username": "verus",
        "password": "verus",
        "table": "verusdb",
    },
    embeddings=OpenAIEmbeddingsEngine(
        api_key=os.getenv("OPENAI_API_KEY", ""), fake=True
    ),
)

client = VerusClient(settings)


client.add(
    collection="MyCollection",
    texts=["This is my first document", "This is my second document"],
    metadata=[{"source": "input", "pages": "3"}, {"source": "input", "pages": "5"}],
)

response = client.search(text="what is my first document?", collection="MyCollection")

print(response)
# [{'collection': 'MyCollection', 'text': 'This is my first document', 'metadata__source': 'input', 'metadata__pages': '3', 'cosine_similarity': 0.937450967393278}, {'collection': 'MyCollection', 'text': 'This is my second document', 'metadata__source': 'input', 'metadata__pages': '5', 'cosine_similarity': 0.8927016698439493}]
