from __future__ import annotations
import redis
import numpy as np
from redis.commands.search.field import TagField, VectorField, NumericField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from verusdb.engines import BaseEngine
from verusdb.settings import Settings

class RedisEngine(BaseEngine):
    """
    PolarsEngine class

    """

    def __init__(self, settings: Settings):
        """
        Create a new PolarsEngine instance
        """
        self.settings = settings
        self.embeddings_engine = settings.embeddings
        self.redis_host = settings.redis_host
        self.redis_port = settings.redis_port
        self.redis_db = settings.redis_db
        self.redis_index = settings.redis_index
        self.redis_password = settings.redis_password
        self.redis_doc_prefix = settings.redis_doc_prefix
        
        self.dimensions = self.embeddings_engine.get_dimensions() # type: ignore

    def load(self):
        
        # connect to redis
        self.store = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db, password=self.redis_password)

        
        try:
            # check to see if index exists
            self.store.ft(self.redis_index).info()

        except:
            # schema
            schema = (
                TextField("collection"),                # Tag Field Name
                TextField("text"),                     # Text Field Name
                TagField("metadata"),                # Tag Field Name
                VectorField("embeddings",               # Vector Field Name
                    "HNSW", {                          # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",             # FLOAT32 or FLOAT64
                        "DIM": self.dimensions,        # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",   # Vector Search Distance Metric
                    }
                ),
            )

            # index Definition
            definition = IndexDefinition(prefix=[self.redis_doc_prefix ], index_type=IndexType.HASH)

            # create Index
            self.store.ft(self.redis_index).create_index(fields=schema, definition=definition)
        
        


    def add(self, texts: list[str],  collection: str | None = None, embeddings: list[list[float]] | None = None, metadata: list[dict[str, str]] | None = None):
        """
        Add a document to the dataframe
        """
     
        data = []
        
             
        if embeddings is None:
            embeddings = [self.embeddings_engine.encode(text) for text in texts] # type: ignore
        
        if metadata is None:
            metadata = [{}] * len(texts)

        
        for text, embedding, meta in zip(texts, embeddings, metadata):
            data.append({
                'collection': collection,
                'text': text,
                'embeddings': np.array(embedding).astype(np.float32).tobytes(),
                'metadata': ','.join([f'{key}:{value}' for key, value in meta.items()])
            })

        # expand the metadata and add it to the data
     
        
        # add the data to the redis index
        pipe = self.store.pipeline()
        
        for i, item in enumerate(data):
            
            pipe.hset(f"doc:{i}", mapping=item)
        

        pipe.execute()
            
            
        
    def clear(self):
        self.store.flushdb()            
        

      

    def delete(self, filters: dict[str, str] | None = None):
        """
        Delete documents from the index based on filters
        """
        self.store.
        
        pass

    

    def search(self,  embedding: list[float], collection: str | None = None, filters: dict[str, str] | None = None, top_k: int = 10, return_objects: bool = False):
        
        # build the query
        query_string = f"@collection:{collection}"
        # query_string = '*'
        
        
        return_fields = ['collection', 'text', 'metadata', 'score']
        
        if filters:
            for key, value in filters.items():
                query_string += f" @metadata:{key}:{value}"
        
            
        # create the query for the embedding, filters and collection
            
        query = (Query(f"({query_string})=>[KNN {top_k} @embeddings $vec as score]")
             .sort_by("score")
             .return_fields(
                *return_fields)
             .dialect(2)
             
             )
        
        
        query_params = {"vec": np.array(embedding).astype(np.float32).tobytes()}
        
        results = self.store.ft(self.redis_index).search(query, query_params) # type: ignore
                
        if return_objects:
            return results
        
        return self._serialize(results.docs)
        
    

    def search_text(self, text: str, collection: str | None = None,  filters: dict[str, str] | None = None, top_k: int = 10, return_object: bool = False):
        """
        Search the index for a text string
        """
        # calulate the embedding
        if self.embeddings_engine is None:
            raise ValueError('Embeddings Engine is not set')
        
        embedding : list[float] = self.embeddings_engine.encode(text)

        #perform the search
        results  = self.search(embedding, collection, filters, top_k, True)

        # Return the top k results
        if return_object:
            return results
        
        return self._serialize(results.docs) # type: ignore
        
        # Return the top k results

    def _serialize(self, documents) -> list[dict[str,str] | dict[str,dict[str,str]]]:
        """
        Serialize the redis output, merging the metadata
        
        """
        
        data: list[dict[str,str] | dict[str,dict[str,str]]] = []
        
        for doc in documents:
            
            data.append({
                'collection': doc['collection'],
                'text': doc['text'],
                'metadata': {key: value for key, value in [item.split(':') for item in doc['metadata'].split(',')]} if doc['metadata'] else {},
                'score': doc['score']
            })
            
            
        return data
        
        
       
    
    def save(self):
        """
        No need to save the index
        """
        
        pass

    