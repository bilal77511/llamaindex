from os import environ
from datasets import load_dataset
import pandas as pd
import json
import pprint
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.response.notebook_utils import display_response
import pymongo


OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
MONGO_URI = environ.get("MONGO_URI")

embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=256)
llm = OpenAI()
Settings.llm = llm
Settings.embed_model = embed_model


if not MONGO_URI:
    print("MONGO_URI not set in environment variables")


def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


mongo_client = get_mongo_client(MONGO_URI)

DB_NAME = "movies"
COLLECTION_NAME = "movies_records"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]


vector_store = MongoDBAtlasVectorSearch(mongo_client, db_name=DB_NAME, collection_name=COLLECTION_NAME, index_name="vector_index")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents([], storage_context=storage_context)

query_engine = index.as_query_engine(similarity_top_k=3)

query = "Recommend a romantic movie suitable for the christmas season and justify your selecton"

response = query_engine.query(query)
print(response)
# pprint.pprint(response.source_nodes)