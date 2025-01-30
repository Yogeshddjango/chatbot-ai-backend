import os
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

@app.get("/api/organisation_database/")
async def upload_file(
    organisation_id: str = Query(..., description="ID of the organisation")
):
    organisation_data = requests.get(EXTERNAL_API_URL.format(org_id=org_id))

    organisation_data = {
        "organisation_id": organisation_id,
        "organisation_data": organisation_data,
        "ai_embeddings_status": "Pending",
        "ai_embeddings_reason": "Initial processing"
    }

    orgainsation_database_object = DatabaseManager()
    orgainsation_database_object.connect()
    orgainsation_database_object.insert_or_update_data(organisation_data)


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
