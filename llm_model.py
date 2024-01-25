from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
import json

INDEX_NAME = "langchain-index"
# initialize pinecone index
def init_pinecone(index_name=INDEX_NAME):
    pc = Pinecone()
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
def create_vector_db_from_text_file(file_path, index_name=INDEX_NAME):
    loader = TextLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    docs = [str(doc) for doc in docs]
    
    embeddings = OpenAIEmbeddings()
    embeds = embeddings.embed_documents(docs)

    # Create a list of dictionaries where each dictionary has an 'id' and 'values'
    vectors = [{"id": f'doc_{i}', "values": embed} for i, embed in enumerate(embeds)]

    index = init_pinecone(index_name)
    index.upsert(vectors=vectors)
    # Save the mapping of doc ids and their corresponding text to a json file
    mappings = {f'doc_{i}': embed  for i, embed in enumerate(docs)}
    with open('data/mapping.json', 'w') as fp:
        json.dump(mappings, fp)

    return index

# search and retrieve from vector db
def search_and_retrieve(query, index_name=INDEX_NAME,max_len=3750):
    #get vector representation of query
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)
    index = init_pinecone(index_name)
    search_res = index.query(vector=query_vector, top_k=4, include_metadata=True)
    cur_len = 0
    contexts = []
    mappings = json.load(open('data/mapping.json', 'r'))
    for row in search_res['matches']:
        text = mappings[row['id']]
        cur_len += len(text) + 4
        if cur_len < max_len:
            contexts.append(text)
        else:
            cur_len -= len(text) + 4
            if max_len - cur_len < 200:
                break
    return "\\\\n\\\\n###\\\\n\\\\n".join(contexts)

# get response from LLM model based on query and docs
def get_response_from_llm(query, docs_page_content):
    llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.0
    )

    prompt = PromptTemplate(
        input_variables=["query", "docs"],   
        template = """
        You are a helpful assistant that uses the following pieces of context to answer the users question.
        Answer the following question: {question}
        by searching the following context:{docs}
        
        Answer the question based on the context provided. If the
        question cannot be answered using the information provided answer
        with "I don't know".

        Your answer should be detailed.
        """
    )

    chain = LLMChain(llm = llm, prompt = prompt)
    response = chain.generate(input_list=[{"question": query, "docs": docs_page_content}])
    answer = response.generations[0][0].text
    return answer