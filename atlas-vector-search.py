import pymongo
import requests
from os import environ

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = environ.get("MONGO_URI")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi("1"))
db = client.sample_mflix
collection = db.movies


hf_token = environ.get("HF_TOKEN")
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


def generate_embedding(text):

    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text},
    )

    if response.status_code != 200:
        raise ValueError(
            f"Request failed with status code {response.status_code}: {response.text}"
        )
    return response.json()


def create_embeddings_for_first_50_movies():
    for doc in collection.find({'plot':{"$exists": True}}).limit(50):
        doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
        collection.replace_one({'_id': doc['_id']}, doc)
        

query = "imaginary characters from outer space at war"

print(len(generate_embedding(query)))

# results = collection.aggregate([
#   {"$vectorSearch": {
#     "queryVector": generate_embedding(query),
#     "path": "plot_embedding_hf",
#     "numCandidates": 100,
#     "limit": 4,
#     "index": "PlotSemanticSearch",
#       }}
# ])

# for document in results:
#     print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')