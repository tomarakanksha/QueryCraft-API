from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from llm_model import llm_model_class

app = FastAPI()

# Endpoint for user queries
@app.post("/answer_query_from_text_file")
async def answer_query_from_text_file(query: str = Query(..., title="User Query", max_length=50), file: UploadFile = File(...), api_key: str = Query(..., title="API Key")):
    if not query:
        return {"answer": "Please provide a query"}
    elif not file or not file.filename.endswith('.txt'):
        return {"answer": "Please upload a .txt file"}
    elif api_key == "" or api_key == None:
        return {"answer": "Please provide OPENAI API key"}
    else:
        file_contents = await file.read()
        try:
            response = get_response_from_llm(query, file_contents, api_key)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error occurred while getting response from LLM model")
        
def get_response_from_llm(query, file_contents, api_key):
    OPENAI_API_KEY = api_key
    llm_model = llm_model_class()
    llm_model.create_vector_db_from_file(file_contents, OPENAI_API_KEY)
    docs_page_content = llm_model.search_and_retrieve(query, OPENAI_API_KEY)
    response = llm_model.get_response_from_llm(query, docs_page_content, OPENAI_API_KEY)
    return {"answer": response}
    
        