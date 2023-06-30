from __future__ import annotations
import unittest
import os
import numpy as np
from verusdb.settings import Settings
from verusdb.client import VerusClient
from verusdb.embeddings.openai import OpenAIEmbeddingsEngine

def generate_fake_embeddings(length):
    
    # return a random vector of length  length normalized to 1
    vector = np.random.rand(length).astype(np.float32)
    return (vector / np.linalg.norm(vector)).tolist()    

class TestVerusPostgreSQLClient(unittest.TestCase):
    

    def setUp(self):
        
        self.settings = Settings(
            engine='postgres',
            postgres = {
                "host": "localhost",
                "port": 5432,
                "db": "verus",
                "username": "verus",
                "password": "verus",
                "table": "verusdb"
            },
            embeddings=OpenAIEmbeddingsEngine(
                api_key=os.getenv('OPENAI_API_KEY', ''),
                fake=True
            ),
            
                
        )

        self.client = VerusClient(self.settings)
        self.client.engine.clear()
        self.client.engine.load()
        
        self.dimensions = self.settings.embeddings.get_dimensions() # type: ignore
        
    def test_add_documents(self):
                
        self.client.add(
            collection='test',
            texts=['test'],
            embeddings=[generate_fake_embeddings(self.dimensions)],
            metadata=[{'test': 'test', 'test2': 'test2'}]
        )
        
        self.assertEqual(len(self.client.get_documents(collection='test')), 1) # type: ignore
        
    def test_search_with_embedding(self):
        
        embedding = generate_fake_embeddings(self.dimensions)
        self.client.add(
            collection='test',
            texts=['test'],
            embeddings=[embedding],
            metadata=[{'test': 'test'}]
        )
        
        
        temp = self.client.search(embedding=embedding, collection='test')
        
        self.assertIsInstance(temp, list)


    def test_search_with_text(self):
        
        self.client.add(
            collection='test_search_with_text',
            texts=['This is my first document', 'This is my second document'],
        )
        
        temp = self.client.search(text='what is my first document?', collection='test_search_with_text')
        
        self.assertIsInstance(temp, list)
        self.assertEqual(len(temp), 2) # type: ignore
        self.assertEqual(temp[0]['collection'], 'test_search_with_text') # type: ignore
                
        