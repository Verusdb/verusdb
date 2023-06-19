from __future__ import annotations
import unittest
import os
import polars as pl
from verusdb.settings import Settings
from verusdb.client import VerusClient
from verusdb.embeddings.openai import OpenAIEmbeddingsEngine

class TestVerusClient(unittest.TestCase):

    def setUp(self):
        
        self.settings = Settings(
            folder='tests/data',
            engine='polars',
            store='parquet',
            embeddings=OpenAIEmbeddingsEngine(
                api_key=os.environ['OPENAI_API_KEY'],
                fake=True
            )
        )

        self.client = VerusClient(self.settings)

        # delete the file if it exists
        if self.settings.get_file():
            if os.path.exists(self.settings.get_file()):
                os.remove(self.settings.get_file())

        
    def test_create_or_open(self):
        self.assertIsInstance(self.client.get_store(), pl.DataFrame)
        self.assertEqual(len(self.client.get_store().columns), 3)
        self.assertListEqual(list(self.client.get_store().columns), ['collection', 'text', 'embeddings'])

    def test_add_documents(self):
        self.client.add(
            collection='test',
            texts=['test'],
            embeddings=[[1.0, 2.0, 3.0]],
            metadata=[{'test': 'test', 'test2': 'test2'}]
        )
        self.assertEqual(len(self.client.get_store()), 1) # type: ignore


    def test_search(self):
        self.client.add(
            collection='test',
            texts=['test'],
            embeddings=[[1.0, 2.0, 3.0]],
            metadata=[{'test': 'test'}]
        )
        temp = self.client.search(embedding=[1.0, 2.0, 3.0], collection='test')

        self.assertIsInstance(temp, list)
        self.assertEqual(len(temp), 4)
        self.assertEqual(temp[0]['collection'], 'test')
        self.assertEqual(temp[0]['text'], 'test')


    def test_search_multiple_entries(self):
        self.client.add(
            collection='test',
            texts=['test', 'test2', 'test3'],
            embeddings=[[1.0, 2.0, 3.0], [1.0, 5.0, 63.0], [71.0, 2.0, 1.0]],
            metadata=[{'test': 'test'}, {'test': 'test2'}, {'test': 'test3'}]
        )
        
        temp = self.client.search(embedding=[1.0, 2.0, 3.0], collection='test')
        self.assertIsInstance(temp, list)
        self.assertEqual(len(temp), 3)
        self.assertEqual(temp[0]['collection'], 'test')
        self.assertEqual(temp[0]['text'], 'test')
        self.assertEqual(temp[0]['metadata']['test'], 'test') # type: ignore

    def test_search_multiple_entries_and_filter(self):
        self.client.add(
            collection='test',
            texts=['test', 'test2', 'test3'],
            embeddings=[[1.0, 2.0, 3.0], [1.0, 5.0, 63.0], [71.0, 2.0, 1.0]],
            metadata=[{'test': 'test'}, {'test': 'test2'}, {'test': 'test3'}]
        )
        
        temp = self.client.search(embedding=[1.0, 2.0, 3.0], filters={'test': 'test'}, collection='test')
        self.assertIsInstance(temp, list)
        self.assertEqual(len(temp), 1) # type: ignore
        self.assertEqual(temp[0]['collection'], 'test') # type: ignore

    
    def test_save(self):
        self.client.add(
            collection='test',
            texts=['test', 'test2', 'test3'],
            embeddings=[[1.0, 2.0, 3.0], [1.0, 5.0, 63.0], [71.0, 2.0, 1.0]],
            metadata=[{'test': 'test'}, {'test': 'test2'}, {'test': 'test3'}]
        )
        self.client.save()
        self.assertEqual(len(self.client.get_store()), 3)
        self.assertListEqual(list(self.client.get_store().columns), ['collection', 'text', 'embeddings', 'metadata__test'])

    def test_delete(self):
        self.client.add(
            collection='test',
            texts=['test', 'test2', 'test3'],
            embeddings=[[1.0, 2.0, 3.0], [1.0, 5.0, 63.0], [71.0, 2.0, 1.0]],
            metadata=[{'test': 'test'}, {'test': 'test2'}, {'test': 'test3'}]
        )
        self.client.delete(filters={'test': 'test'})
        self.assertEqual(len(self.client.get_store()), 2)
        self.assertListEqual(list(self.client.get_store().columns), ['collection', 'text', 'embeddings', 'metadata__test'])
        self.assertEqual(self.client.get_store().shape[0], 2)


    def test_embeddings_creation(self):
        self.client.add(
            collection='test',
            texts=['This is the first test', 'This is the second test', 'This is the third test'],
            metadata=[{'test': 'test1'}, {'test': 'test2'}, {'test': 'test3'}]
        )

        result = self.client.search(text='What is the first test?', collection='test')

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3) # type: ignore
        # self.assertEqual(result[0]['collection'], 'test')
        # self.assertEqual(result[0]['text'], 'This is the first test')
        