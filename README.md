# Verus

Verus is a powerful, lightweight and flexible vector store that is designed to work seamlessly with multiple databases. It is built with LLM in mind, making it an ideal choice for natural language processing, information retrieval, and recommendation systems.

One of the key benefits of Verus is its agnostic design, which allows it to integrate with a wide range of databases. This means that you can use Verus with your existing database infrastructure, without having to worry about compatibility issues.

## TODOs
- [ ] Redis Integration
- [ ] PostgreSQL Integration
- [ ] DuckDB Integration
- [ ] CosmosDB Intergration
- [ ] Benchmarking


## Installation

You can install VerusDB using pip:

`pip install verusdb`

## Usage

```python
from verusdb.settings import Settings
from verusdb.client import VerusClient

# Create a new VerusDB client
settings = Settings(folder='data', engine='polars')
client = VerusClient(settings)

# Add some documents to the database
client.add(
    collection='MyCollection',
    texts=['This is my first document', 'This is my second document'],
    metadata=[{'source': 'input', 'pages': '3'}, {'source': 'input', 'pages': '5'}]
)

# Search for documents
response = client.search(text='what is my first document?')
```

This will output a list of documents that match the search query, along with their metadata and a cosine similarity score. You can adjust the the number of results you obtain by relevance


```json
[
  {
    "collection": "MyCollection",
    "text": "This is my first document",
    "metadata": {
      "source": "input",
      "pages": "5"}
    ,
    "score": 0.937450967393278
  },
  {
    "collection": "MyCollection",
    "text": "This is my second document",
    "metadata": {
      "source": "input",
      "pages": "5"}
    ,
    "score": 0.8927016698439493
  }
]
```

# Configuration

You can configure VerusDB by passing a Settings object to the VerusClient constructor. The Settings object allows you to specify the folder where the database files will be stored, the storage engine to use (currently Polars and Redis are supported), 

## Polars

```python
from verusdb.settings import Settings
from verusdb.client import VerusClient
from verusdb.embeddings.openai import OpenAIEmbeddingsEngine

# Create a new VerusDB client with custom settings
settings = Settings(
    folder='data',
    engine='polars',
    embeddings=OpenAIEmbeddingsEngine(key='my-openai-api-key')
)
client = VerusClient(settings)
```
## Redis

```python
from verusdb.settings import Settings
from verusdb.client import VerusClient
from verusdb.embeddings.openai import OpenAIEmbeddingsEngine

# Create a new VerusDB client with custom settings
settings = Settings(
    engine='redis',
    redis = {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'password': None,
                'prefix': 'doc:',
                'index': 'verusdb'
    },     
    embeddings=OpenAIEmbeddingsEngine(key='my-openai-api-key')
)
client = VerusClient(settings)
```

# Contributing
If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/verusdb/verusdb). Pull requests are also welcome!

# License
VerusDB is licensed under the [MIT License](https://opensource.org/licenses/MIT).