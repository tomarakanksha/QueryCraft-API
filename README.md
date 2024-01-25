# QueryCraft-API

This repository contains an API that leverages a Large Language Model (LLM) to provide relevant answers to user queries based on text data stored in a vector database.

## Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Load your environment variables: `cp .env.example .env` (Linux/Mac) or `copy .env.example .env` (Windows)

## Usage

1. Fill in the required API keys in the code.
2. Add your txt file in '/Data' folder.
3. Run the FastAPI application: `uvicorn main:app --reload`
4. Visit `http://127.0.0.1:8000/docs` in your browser to interact with the API using Swagger documentation.

## Environment Variables

Create a `.env` file and add the following:

```env
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```
