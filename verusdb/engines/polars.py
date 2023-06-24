from __future__ import annotations
import numpy as np
import polars as pl
from verusdb.engines import BaseEngine
from verusdb.settings import Settings
from verusdb.utils import generate_uuid
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
        self.embeddings_engine = settings.embeddings

    def __get_blank_store(self):
        """
        Get a blank dataframe
        """
        return pl.DataFrame(
            schema=[
                ("uuid", str),
                ("collection", pl.Utf8),
                ("text", pl.Utf8),
                ("embeddings", pl.List(pl.Float64)),
            ]
        )

    def load(self):
        """
        Load the dataframe from disk
        """
        if not self.settings.persist:
            self.store = self.__get_blank_store()
            return

        if self.settings.folder and os.path.exists(
            self.settings.folder + "/verusdb.parquet"
        ):
            self.store = pl.read_parquet(self.settings.folder + "/verusdb.parquet")
        else:
            self.store = self.__get_blank_store()

    def clear(self):
        self.store = self.__get_blank_store()

    def add(
        self,
        texts: list[str],
        collection: str | None = None,
        embeddings: list[list[float]] | None = None,
        metadata: list[dict[str, str]] | None = None,
    ):
        """
        Add a document to the dataframe
        """
        metadata_df = pl.DataFrame(metadata)

        if embeddings is None:
            if self.embeddings_engine is None:
                raise ValueError("Embeddings engine not set")

            embeddings = [self.embeddings_engine.encode(text) for text in texts]

        data = {
            "uuid": generate_uuid(dimension=len(texts)),
            "collection": collection,
            "text": texts,
            "embeddings": embeddings,
        }

        # Add the metadata to the dataframe
        for key in metadata_df.columns:
            data["metadata__" + key] = metadata_df[key]

        # Add the columns starting with metadata__ from store to the data dict
        for key in self.store.columns:
            if key.startswith("metadata__") and key not in data.keys():
                data[key] = ""

        # Add to self.store the new columns with '' as the value
        self.store = self.store.with_columns(
            (pl.lit("")).alias("metadata__" + key) for key in metadata_df.columns
        )

        # Add the new dataframe to the existing dataframe
        self.store = self.store.vstack(pl.DataFrame(data))

    def delete(self, uuid: str | None = None, collection: str | None = None, filters: dict[str, str] | None = None):
        """
        Delete documents from the store
        """
        if uuid is None and filters is None and collection is None:
            ValueError("uuid, collection or filters must be provided")
            
        if collection is not None:
            self.store = self.store.filter(pl.col("collection") != collection)

        if uuid is not None:
            self.store = self.store.filter(pl.col("uuid") != uuid)

        if filters is not None:
            for key, value in filters.items():
                self.store = self.store.filter(pl.col(f"metadata__{key}") != value)

    def _cosine_similarity(
        self,
        embedding: list[float],
        collection: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """
        Calculate the cosine similarity
        """
        temp = self.store

        if collection is not None:
            temp = temp.filter(pl.col("collection") == collection)

        # filter the dataframe
        if filters is not None:
            for key, value in filters.items():
                temp = temp.filter(pl.col("metadata__" + key) == value)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("The embedding cannot be a zero vector")

        # Calculate the cosine similarity
        # TODO: This is not the most efficient way to do this, there is a pylint warning
        temp = temp.with_columns(
            pl.col("embeddings").apply(lambda x: (x.dot(embedding)) / (norm * np.linalg.norm(x))).alias("score")  # type: ignore
        )

        # Return the dataframe sorted by cosine similarity
        return temp.sort("score", descending=True)

    def search(
        self,
        embedding: list[float],
        collection: str | None = None,
        filters: dict[str, str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, str]]:
        """
        Search the dataframe
        """

        # Calculate the cosine similarity
        results = self._cosine_similarity(embedding, collection, filters).drop(
            "embeddings"
        )

        # Return the top k results
        return self._serialize(results.head(top_k), include_score=True)

    def search_text(
        self,
        text: str,
        collection: str | None = None,
        filters: dict[str, str] | None = None,
        top_k: int = 10,
        return_object: bool = False,
    ):
        """
        Search the dataframe
        """

        if self.embeddings_engine is None:
            raise ValueError("Embeddings Engine is not set")

        # calulate the embedding
        embedding: list[float] = self.embeddings_engine.encode(text)

        # perform the search
        results = self._cosine_similarity(
            embedding=embedding, collection=collection, filters=filters
        ).drop("embeddings")

        # Return the top k results
        if return_object:
            return results.head(top_k)

        return self._serialize(results.head(top_k), include_score=True)

    def _serialize(
        self, documents: pl.DataFrame, include_score: bool = False
    ) -> list[dict[str, str]]:
        if not include_score and "score" in documents.columns:
            documents = documents.drop("score")

        serialized_data = documents.to_dicts()

        for data in serialized_data:
            metadata = {}
            keys_to_remove = []
            for key, value in data.items():
                if key.startswith("metadata__"):
                    metadata[key.split("__")[1]] = value
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                data.pop(key)
            data["metadata"] = metadata

        return serialized_data

    def get_documents(self, collection: str | None = None):
        if collection is None:
            return self._serialize(self.store)

        return self._serialize(self.store.filter(pl.col("collection") == collection))

    def get_document(self, uuid: str):
        document = self._serialize(
            self.store.filter(pl.col("uuid") == uuid).drop("embeddings")
        )
        return document[0] if len(document) > 0 else None

    def update(self, uuid: str, metadata: dict[str, str]):
        """
        Update the metadata of a document
        """

        removed_uuid = self.store.filter(pl.col("uuid") != uuid)
        uuid_df = self.store.filter(pl.col("uuid") == uuid)

        # replace all metadata columns with empty strings from the uuid_df
        for key in uuid_df.columns:
            if key.startswith("metadata__"):
                uuid_df = uuid_df.with_columns(pl.lit("").alias(key))

        # add the metadata to the uuid_df
        for key, value in metadata.items():
            uuid_df = uuid_df.with_columns(pl.lit(value).alias("metadata__" + key))

        # combine the two dataframes
        self.store = removed_uuid.vstack(uuid_df)

        return self.get_document(uuid)

    def save(self):
        """
        Save the dataframe
        """
        self.store.write_parquet(self.settings.file)
