from fastapi import FastAPI, Query
from llm_model import *
from dotenv import load_dotenv
from fastapi import HTTPException

app = FastAPI()

load_dotenv()

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
        
        