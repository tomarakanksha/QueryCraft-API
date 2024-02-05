from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()

class llm_model_class:
    INDEX_NAME = "langchain-index"
    index = None
    # initialize pinecone index
    def __init__(self, index_name=INDEX_NAME):
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if(index_name in pc.list_indexes().names()):
            pc.delete_index(index_name)
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
        self.index = index


    # create vector db from text file and upload to pinecone
    def create_vector_db_from_file(self, file_content, OPENAI_API_KEY, index_name=INDEX_NAME):
        docs = file_content.decode('utf-8')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.create_documents([docs])
        docs = [str(doc) for doc in docs]
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        embeds = embeddings.embed_documents(docs)

        # Create a list of dictionaries where each dictionary has an 'id' and 'values'
        vectors = [{"id": f'doc_{i}', "values": embed} for i, embed in enumerate(embeds)]

        self.index.upsert(vectors=vectors)
        # Save the mapping of doc ids and their corresponding text to a json file
        mappings = {f'doc_{i}': embed  for i, embed in enumerate(docs)}
        if not os.path.exists('data'):
            os.makedirs('data')
        with open('data/mapping.json', 'w') as fp:
            json.dump(mappings, fp)

    # search and retrieve from vector db
    def search_and_retrieve(self, query,OPENAI_API_KEY, index_name=INDEX_NAME,max_len=3750):
        #get vector representation of query
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        query_vector = embeddings.embed_query(query)
        search_res = self.index.query(vector=query_vector, top_k=4, include_metadata=True)
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
    def get_response_from_llm(self, query, docs_page_content, OPENAI_API_KEY):
        llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
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