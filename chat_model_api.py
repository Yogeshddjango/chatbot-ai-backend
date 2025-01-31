import os
import json
import uvicorn
import logging
import requests
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from rag_folder.question_answer import ChatBot
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from database.organisation_database import DatabaseManager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from organisation_embedding_creation.embedding_generation import CreateDataEmbedding

app = FastAPI()
load_dotenv()

logger = logging.getLogger("fastapi_app")
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/")
async def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.post("/api/organisation_database/")
async def upload_file(organisation_id: str = Query(..., description="Organisation ID is required")):
    # organisation_data = requests.get(EXTERNAL_API_URL.format(org_id=org_id))

    data = {
            "organization_details": {
                "name": "Tech Innovators Ltd.",
                "address": {
                "street": "123 Innovation Drive",
                "city": "San Francisco",
                "state": "CA",
                "zip": "94107",
                "country": "USA"
                },
                "contact": {
                "phone": "+1-415-123-4567",
                "email": "contact@techinnovators.com",
                "website": "https://www.techinnovators.com"
                },
                "departments": [
                {
                    "name": "Engineering",
                    "head": "John Doe",
                    "employees": 120
                },
                {
                    "name": "Marketing",
                    "head": "Jane Smith",
                    "employees": 45
                },
                {
                    "name": "Human Resources",
                    "head": "Robert Brown",
                    "employees": 30
                }
                ],
                "established_year": 2010,
                "employees_count": 500,
                "revenue": "50M USD",
                "industry": "Technology"
            }
        }
    organisation_data_from_frontend = json.dumps(data)
    organisation_data = {
        "organisation_id": organisation_id,
        "organisation_data": organisation_data_from_frontend,
        "ai_embeddings_status": "Pending",
        "ai_embeddings_reason": "Initial processing"
    }

    orgainsation_database_object = DatabaseManager()
    orgainsation_database_object.connect()
    orgainsation_database_object.insert_or_update_data(organisation_data)

    organisation_vector_database = CreateDataEmbedding()
    embedding_status = organisation_vector_database._create_embedding_selection(organisation_data)

    orgainsation_database_object.insert_or_update_data(embedding_status)
    orgainsation_database_object.close()

    return JSONResponse(content={
            "message": embedding_status['ai_embeddings_reason'],
            "status": embedding_status['ai_embeddings_status']
        })


@app.get("/api/organisation_chatbot/{organisation_id}")
async def get_organisation_data(    
                        organisation_id,
                        user_query: str = Body(..., embed=True),
                    )-> JSONResponse:

    if not user_query:
        raise HTTPException(status_code=400, detail="Missing query")

    data = {
        "user_query": user_query,
        "organisation_id": organisation_id
    }
    
    chatbot = ChatBot()
    answer = chatbot.get_response(data)
    return JSONResponse(content=answer)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')
