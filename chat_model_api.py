import os
import json
import uvicorn
import logging
import requests
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from src.rag_folder.question_answer import ChatBot
from starlette.middleware.cors import CORSMiddleware
from src.database.organisation_database import DatabaseManager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from src.organisation_embedding_creation.embedding_generation import CreateDataEmbedding

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
async def upload_file(
        organisation_id: str = Query(..., description="Organisation ID is required"),
        organisation_data: dict = Body(..., embed=True)
    ):
    organisation_data_from_frontend = json.dumps(organisation_data)
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


@app.get("/api/organisation_chatbot/")
async def get_organisation_data(    
                        organisation_id: str = Query(..., description="Organisation ID is required"),
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