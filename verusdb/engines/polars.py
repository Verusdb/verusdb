from __future__ import annotations
import numpy as np
import polars as pl
from verusdb.engines import BaseEngine
from verusdb.settings import Settings
import os


class PolarsEngine(BaseEngine):
    """
    PolarsEngine class

    """

    def __init__(self, settings: Settings):
        """
        Create a new PolarsEngine instance
        """
        self.store = pl.DataFrame()
        self.settings = settings
        self.embeddings = settings.embeddings

        
    def __get_blank_store(self):
        """
        Get a blank dataframe
        """
        return pl.DataFrame(schema=
            [
                ('collection', pl.Utf8),
                ('text', pl.Utf8),
                ('embeddings', pl.List(pl.Float64)),
            ]
        )

    def load(self):
        """
        Load the dataframe from disk
        """
        if not self.settings.persist:

            self.store = self.__get_blank_store()

            return
        
        if os.path.exists(self.settings.folder+'/verusdb.parquet'):
            self.store = pl.read_parquet(self.settings.folder+'/verusdb.parquet')
        else:
            self.store = self.__get_blank_store()

            
        


    def add(self, collection: str, texts: list[str], embeddings: list[list[float]] | None = None, metadata: list[dict[str, str]] | None = None):
        """
        Add a document to the dataframe
        """
     
        metadata_df = pl.DataFrame(metadata)

        data = {
            'collection': collection,
            'text': texts,
            'embeddings': embeddings if embeddings else [self.embeddings.encode(text) for text in texts],
        }

        #add the metadata to the dataframe
        for key in metadata_df.columns:
            data['metadata__'+key] = metadata_df[key]

        # add the columns starting with metadata__ from store to the data dict
        for key in self.store.columns:
            if key.startswith('metadata__') and key not in data.keys():
                data[key] = ''


        # add to self.store the new columns with '' as the value
        self.store = self.store.with_columns(
            (pl.lit('')).alias('metadata__'+key) for key in metadata_df.columns
        )

        # add the new dataframe to the existing dataframe
        self.store = self.store.vstack(pl.DataFrame(data))

    def delete(self, filters: dict[str, str] | None = None):
        """
        Delete documents from the store
        """
        if filters is None:
            self.store = pl.DataFrame(schema=
            [
                ('collection', pl.Utf8),
                ('text', pl.Utf8),
                ('embeddings', pl.List(pl.Float64)),
            ])
        else:
            for key, value in filters.items():
                self.store = self.store.filter(pl.col(f'metadata__{key}') != value)


    def _cosine_similarity(self, embedding: list[float], filters: dict[str, str] | None = None) -> pl.DataFrame:

        #expand the metadata column
        temp = self.store

        #filter the dataframe
        if filters is not None:
            for key, value in filters.items():
                temp = temp.filter(pl.col('metadata__'+key) == value)

        norm =  np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError('The embedding cannot be a zero vector')
        
        # Calculate the cosine similarity
        # TODO: This is not the most efficient way to do this, there is a pylint warning
        temp = temp.with_columns(
            pl.col('embeddings').apply(lambda x: (x.dot(embedding))/ (norm * np.linalg.norm(x))).alias('cosine_similarity')
        )
        
        # Return the dataframe sorted by cosine similarity
        return temp.sort('cosine_similarity', descending=True)

    

    def search(self, embedding: list[float], filters: dict[str, str] | None = None, top_k: int = 10) -> list[dict[str, str]]:
        """
        Search the dataframe
        """
        # Calculate the cosine similarity
        results = self._cosine_similarity(embedding, filters).drop('embeddings')

        # Return the top k results
        return self._serialize(results.head(top_k))
    

    def search_text(self, text: str, filters: dict[str, str] | None = None, top_k: int = 10) -> list[dict[str, str]]:
        """
        Search the dataframe
        """
        # calulate the embedding
        embedding : list[float] = self.embeddings.encode(text)

        #perform the search
        results = self._cosine_similarity(embedding=embedding, filters=filters).drop('embeddings')

        # Return the top k results
        return self._serialize(results.head(top_k))

    def _serialize(self, documents: pl.DataFrame) -> list[dict[str, str]]:
        return documents.to_dicts()
    
    def save(self):
        """
        Save the dataframe
        """
        self.store.write_parquet(self.settings.file)

    