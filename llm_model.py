from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

INDEX_NAME = "langchain-index"
# initialize pinecone index
def init_pinecone(api_key, index_name=INDEX_NAME):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            #The OpenAI embedding model `text-embedding-ada-002' uses 1536 dimensions
            dimension=1536,
            metric="euclidean",
            spec={'pod': {'environment': 'gcp-starter',
                               'pod_type': 'starter',
                               'pods': 1,
                               'replicas': 1,
                               'shards': 1}},
        ) 
    index = pc.Index(index_name)
    return index


# create vector db from text file and upload to pinecone
def create_vector_db_from_text_file(file_path, api_key, index_name=INDEX_NAME):
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    docs = [str(doc) for doc in docs]
    
    embeddings = OpenAIEmbeddings()
    embeds = embeddings.embed_documents(docs)

    # Create a list of dictionaries where each dictionary has an 'id' and 'values'
    vectors = [{"id": f'doc_{i}', "values": embed} for i, embed in enumerate(embeds)]

    index = init_pinecone(api_key, index_name)
    index.upsert(vectors=vectors)
    return index

# search and retrieve from vector db
def search_and_retrieve(query, api_key, index_name=INDEX_NAME):
    #get vector representation of query
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)
    index = init_pinecone(api_key, index_name)
    search = index.query(vector=query_vector, top_k=4, include_values=True)

    return search