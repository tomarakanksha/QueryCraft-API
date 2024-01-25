from fastapi import FastAPI, Query
from llm_model import *
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()

# Endpoint for user queries
@app.get("/answer_query")
def answer_query(query: str = Query(..., title="User Query", max_length=50)):
    if not query:
        return {"answer": "Please provide a query"}
    else:
        create_vector_db_from_text_file("Data/TechData.txt", os.getenv("PINECONE_API_KEY"))
        top_K_docs = search_and_retrieve(query, api_key=os.getenv("PINECONE_API_KEY"), k = 4)
    
    return {"answer": "Final Result after search and retrieval"}
        