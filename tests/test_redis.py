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

class TestVerusRedisClient(unittest.TestCase):
    

    def setUp(self):
        
        self.settings = Settings(
            engine='redis',
            redis = {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'password': None,
                'prefix': 'doc:',
                'index': 'verusdb'
            },
            embeddings=OpenAIEmbeddingsEngine(
                api_key=os.environ['OPENAI_API_KEY'],
                fake=True
            ),
            
                
        )

        self.client = VerusClient(self.settings)
        
        self.dimensions = self.settings.embeddings.get_dimensions() # type: ignore
        
        self.client.engine.clear()
        self.client.engine.load()
        

    def test_add_documents(self):
                
        self.client.add(
            collection='test',
            texts=['test'],
            embeddings=[generate_fake_embeddings(self.dimensions)],
            metadata=[{'test': 'test', 'test2': 'test2'}]
        )
        
        self.assertEqual(self.client.get_store().dbsize(), 1) # type: ignore
        

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
        self.assertEqual(len(temp), 1) # type: ignore
        self.assertEqual(temp[0]['collection'], 'test') # type: ignore
        
    def test_search_with_text(self):
        
        
        self.client.add(
            collection='test_search_with_text',
            texts=['This is my first document', 'This is my second document'],
        )
        
        temp = self.client.search(text='what is my first document?', collection='test_search_with_text')
        
        self.assertIsInstance(temp, list)
        self.assertEqual(len(temp), 2) # type: ignore
        self.assertEqual(temp[0]['collection'], 'test_search_with_text') # type: ignore
                
        
      