import streamlit as st
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


def generate_embedding(text):
    return embed_model.get_text_embedding(text)

# Title of the app
st.title('TicketAI')

# Header
st.header('AI-powered ticketing system')

query = ""
query = st.text_input('Enter the prompt:', 'Type here')

if st.button('Submit'):
    results = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "queryVector": generate_embedding(query),
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": 4,
                    "index": "vector_index",
                }
            }
        ]
    )

    for document in results:
        print(document["text"])
        print("\n")

    # Display the user input
    # put this in a block that can be copied easily
    documentnumber = document["metadata"]["ticket-number"].strip('"')
    st.write(f'Ticket Number: {documentnumber}')
    st.write(f'{document["text"]}')

    

