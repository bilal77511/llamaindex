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

collection.delete_many({})

# https://huggingface.co/datasets/MongoDB/embedded_movies
dataset = load_dataset("MongoDB/embedded_movies")
dataset_df = pd.DataFrame(dataset["train"])

# data cleaning, preparation, and loading
dataset_df = dataset_df.dropna(subset=["plot"])
dataset_df = dataset_df.dropna(subset=["fullplot"])

print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())
dataset_df = dataset_df.drop(columns=["plot_embedding"])

embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=256)

llm = OpenAI()

Settings.llm = llm

Settings.embed_model = embed_model

# Convert the DataFrame to a JSON string representation
documents_json = dataset_df.to_json(orient="records")
# Load the JSON string into a Python list of dictionaries
documents_list = json.loads(documents_json)

print("Number of documents: ", len(documents_list))


llama_documents = []

for i in range(100):

    document = documents_list[i]

    # Value for metadata must be one of (str, int, float, None)
    document["writers"] = json.dumps(document["writers"])
    document["languages"] = json.dumps(document["languages"])
    document["genres"] = json.dumps(document["genres"])
    document["cast"] = json.dumps(document["cast"])
    document["directors"] = json.dumps(document["directors"])
    document["countries"] = json.dumps(document["countries"])
    document["imdb"] = json.dumps(document["imdb"])
    document["awards"] = json.dumps(document["awards"])

    #   Create a Document object with the text and excluded metadata for llm and embedding models
    llama_document = Document(
        text=document["fullplot"],
        metadata=document,
        excluded_llm_metadata_keys=["fullplot", "metacritic"],
        excluded_embed_metadata_keys=[
            "fullplot",
            "metacritic",
            "poster",
            "num_mflix_comments",
            "runtime",
            "rated",
        ],
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    )

    llama_documents.append(llama_document)

# Observing an example of what the LLM and Embedding model receive as input
print(
    "\nThe LLM sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.LLM),
)
print(
    "\nThe Embedding model sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.EMBED),
)


parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(llama_documents)
print("Nodes count: ", len(nodes))

i = 0
# for node in nodes:

#     print(i)
#     i += 1
#     node_embedding = embed_model.get_text_embedding(
#         node.get_content(metadata_mode="all")
#     )
#     node.embedding = node_embedding


# DATA INGESTION TO VECTOR DATABASE

vector_store = MongoDBAtlasVectorSearch(
    mongo_client,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    index_name="vector_index",
)
index = VectorStoreIndex.from_vector_store(vector_store)


# QUERY ENGINE STARTED!
query_engine = index.as_query_engine(similarity_top_k=3)
query = "Recommend a romantic movie suitable for the christmas season and justify your selecton"
response = query_engine.query(query)
display_response(response)
pprint.pprint(response.source_nodes)
