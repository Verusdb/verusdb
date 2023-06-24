from __future__ import annotations
import psycopg2
import polars as pl
import numpy as np
import json
from psycopg2.extras import RealDictCursor
from verusdb.engines import BaseEngine
from verusdb.settings import Settings
from verusdb.utils import generate_uuid


class PostgreSQLEngine(BaseEngine):
    def __init__(self, settings: Settings):
        """
        Create a new PolarsEngine instance
        """
        self.settings = settings
        self.embeddings_engine = settings.embeddings
        self.pg_host = settings.pg_host
        self.pg_port = settings.pg_port
        self.pg_db = settings.pg_db
        self.username = settings.username
        self.pg_password = settings.pg_password
        self.pg_table = settings.pg_table

        self.dimensions = self.embeddings_engine.get_dimensions()  # type: ignore

        self.connection = self.__get_connection()

    def __get_connection(self):
        return psycopg2.connect(
            f"dbname={self.pg_db} user={self.username} password={self.pg_password} host={self.pg_host} port={self.pg_port}"
        )

    def load(self):        
        cursor = self.connection.cursor()

        cursor.execute(
            f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{self.pg_table}');"
        )
        results = cursor.fetchone()

        if results and not results[0]:
            cursor.execute(
                f"CREATE TABLE {self.pg_table} (uuid varchar(250), collection text, text text, metadata JSON, embeddings vector({self.dimensions}));"
            )
            self.connection.commit()
            
        cursor.close()


    def _serialize(self, documents):
        return super()._serialize(documents)

    def add(
        self,
        texts: list[str],
        collection: str | None = None,
        embeddings: list[list[float]] | None = None,
        metadata: list[dict[str, str]] | None = None,
    ):
        cursor = self.connection.cursor()

        if embeddings is None:
            embeddings = [self.embeddings_engine.encode(text) for text in texts]  # type: ignore

        if metadata is None:
            metadata = [{}] * len(texts)

        uuids = generate_uuid(len(texts))

        query = f"INSERT INTO {self.pg_table} (uuid, collection, text, metadata, embeddings) VALUES "
        for uuid, text, meta, embedding in zip(uuids, texts, metadata, embeddings):
            query += f"('{uuid}','{collection}', '{text}', '{json.dumps(meta)}', '{embedding}'),"

        query = query[:-1] + ";"

        cursor.execute(query)
        self.connection.commit()
        cursor.close()
        
    def search(
        self,
        embedding: list[float],
        collection: str | None = None,
        filters: dict[str, str] | None = None,
        top_k: int = 10,
    ) :

        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        metadata_filters = ""
        
        if filters:
            metadata_filters = "AND metadata->>'{}' = '{}'".format(
                list(filters.keys())[0], list(filters.values())[0]
            )

        cursor.execute(f"SELECT uuid, collection, text, metadata  FROM {self.pg_table} WHERE collection = '{collection}' {metadata_filters} ORDER BY embeddings <-> '{embedding}' LIMIT {top_k};")
        results = cursor.fetchall()  
        cursor.close()
        
        return results

        

    def clear(self):
        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM {self.pg_table};")
        self.connection.commit() 
        cursor.close()


    def get_documents(self, collection: str | None = None):
        if collection is None:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f"SELECT * FROM {self.pg_table};")
        else:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                f"SELECT * FROM {self.pg_table} WHERE collection = '{collection}';"
            )

        result = cursor.fetchall()
        
        cursor.close()


        return result

    def get_document(self, uuid: str):
        cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(f"SELECT * FROM {self.pg_table} WHERE uuid = '{uuid}';")
        result = cursor.fetchone()
        cursor.close()
        self.connection.close()

        return result

    def update(self, uuid, metadata):
        cursor = self.connection.cursor()
        cursor.execute(
            f"UPDATE {self.pg_table} SET metadata = '{json.dumps(metadata)}' WHERE uuid = '{uuid}';"
        )
        self.connection.commit()
        cursor.close()
        self.connection.close()

    def delete(self, uuid):
        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM {self.pg_table} WHERE uuid = '{uuid}';")
        self.connection.commit()
        cursor.close()
        self.connection.close()
