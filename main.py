from fastapi import FastAPI, HTTPException, Query
from llm_model import *
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

load_dotenv()


class Item(BaseModel):
    api_key: str
    collection_name: Optional[str] = "Test"
    user_query: str

# Endpoint for user queries
@app.get("/answer_query")
def answer_query(query: str = Query(..., title="User Query", max_length=50)):
    if not query:
        return {"answer": "Please provide a query"}
    else:
        create_vector_db_from_text_file("Data/TechData.txt")
        docs_page_content = search_and_retrieve(query)
        try:
            response = get_response_from_llm(query, docs_page_content)
            return {"answer": response}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail="Error occurred while getting response from LLM model")
        
@app.post("/predict")
def predict_intent(item: Item):
    if not item.user_query:
        return {"answer": "Please provide a query"}
    elif not item.api_key:
        return {"answer": "Please provide a valid API key"}
    else:
        if(item.collection_name == None or item.collection_name == ""):
            item.collection_name = "Test"
        OPENAI_API_KEY = item.api_key
        try:
            top_three_responses = ['response1', 'response2', 'response3']
            return {"intents": top_three_responses}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail="Error occurred while getting response from LLM model")