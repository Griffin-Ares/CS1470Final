from openai import OpenAI
import os
import json
import pickle
from multiprocessing import Pool, Lock, Manager
from tqdm import tqdm
import random


client = OpenAI(api_key='sk-proj-yUDloSnJVhVKxtHMQjccT3BlbkFJ2ONuFc6pbPOxQ6GA5cxh')

def get_embedding(description, max_retries=5, base_delay=10.0):
    """Utility function to retry embedding API calls with exponential backoff."""
    for attempt in range(max_retries):
        try:
            embedding = client.embeddings.create(
                input=[description],
                model="text-embedding-3-small",
                dimensions=128
            ).data[0].embedding
            return embedding
        except Exception as e:
            if 'rate_limit' in str(e).lower():
                wait_time = base_delay * (2 ** attempt)
                print(f"Rate limit reached, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Embedding API request failed after several retries.")