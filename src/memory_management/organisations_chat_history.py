import os
import uuid
import logging
import psycopg
from dotenv import load_dotenv
from langchain_postgres import PostgresChatMessageHistory

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
load_dotenv()

DBNAME = os.getenv("DBNAME")
DBUSER = os.getenv("DBUSER")
DBPW = os.getenv("DBPW")
DBHOST = os.getenv("DBHOST")
DBPORT = os.getenv("DBPORT")
CHATHISTORY_SETTINGS = f"postgresql://{DBUSER}:{DBPW}@{DBHOST}:{DBPORT}/{DBNAME}"

table_name = "message_store"
PostgresChatMessageHistory.create_tables(psycopg.connect(CHATHISTORY_SETTINGS), table_name)

class ChatHistory:
    def __init__(self, organisation_id: str) -> None:
        self.connection = psycopg.connect(CHATHISTORY_SETTINGS)
        self.session_id = uuid.UUID(organisation_id.replace('-', '').ljust(32, '0'))

    def session_based_chat_history(self):
        """Retrieve the chat history for the session."""
        chat_history = PostgresChatMessageHistory(
            table_name,
            str(self.session_id),
            sync_connection=self.connection 
        )
        return chat_history
