# VerusDB

VerusDB is a lightweight document database that allows you to store and search for documents based on their text content and metadata using embeddings. It is designed to be easy to use and integrate into your existing Python projects and infrastructure.


## TODOs

- [ ] DuckDB Integration
- [ ] Redis Integration
- [ ] Benchmarking
- [ ] SQLAlchemy Storage
- [ ] MongoDB Storage


## Installation

You can install VerusDB using pip:

`pip install verusdb`

## Usage

```python
from verusdb.settings import Settings
from verusdb.client import VerusClient

# Create a new VerusDB client
settings = Settings(folder='data', engine='polars', store='parquet')
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
    "metadata__source": "input",
    "metadata__pages": "3",
    "cosine_similarity": 0.937450967393278
  },
  {
    "collection": "MyCollection",
    "text": "This is my second document",
    "metadata__source": "input",
    "metadata__pages": "5",
    "cosine_similarity": 0.8927016698439493
  }
]
```

# Configuration

You can configure VerusDB by passing a Settings object to the VerusClient constructor. The Settings object allows you to specify the folder where the database files will be stored, the storage engine to use (currently only Polars is supported), and the file format to use (currently only Parquet is supported).

```python
from verusdb.settings import Settings
from verusdb.client import VerusClient
from verusdb.embeddings.openai import OpenAIEmbeddingsEngine

# Create a new VerusDB client with custom settings
settings = Settings(
    folder='data',
    engine='polars',
    store='parquet',
    embeddings=OpenAIEmbeddingsEngine(key='my-openai-api-key')
)
client = VerusClient(settings)
```
## Contributing
If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/verusdb/verusdb). Pull requests are also welcome!

## License
VerusDB is licensed under the [MIT License](https://opensource.org/licenses/MIT).