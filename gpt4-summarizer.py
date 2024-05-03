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
from openai import OpenAI

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
MONGO_URI = environ.get("MONGO_URI")

client = OpenAI(api_key=OPENAI_API_KEY)

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

DB_NAME = "itdata"
COLLECTION_NAME = "it_support_data"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]


def generate_embedding(text):
    return embed_model.get_text_embedding(text)



# Function to query GPT4 with all the data
def summary(query):
    results = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "queryVector": generate_embedding(query),
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": 3,
                    "index": "vector_index",
                }
            }
        ]
    )


    fullanswer = "Top 3 results: \n"
    i = 0
    for document in results:
        i += 1
        text = document["text"]
        fullanswer +=  "#" + str(i) + ": " + text + "\n"
    
    input = query + "this is the raw answer use it in your response" + fullanswer
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant for Boston University IT Support, mention this when greeted.
             Your role is to answer questions about Bu Login, Blackboard and other IT related issues.""",
            },
            {"role": "user", "content": input},
        ],
    )
    return response.choices[0].message.content

# Example usage
query = "Who is your favorite superhero?"
gpt4_response = summary(query)
print("summary:", gpt4_response)
