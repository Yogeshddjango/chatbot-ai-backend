import os
import uuid
import psycopg
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from psycopg import Connection

load_dotenv()

DBNAME = os.getenv("DBNAME")
DBUSER = os.getenv("DBUSER")
DBPW = os.getenv("DBPW")
DBHOST = os.getenv("DBHOST")
DBPORT = os.getenv("DBPORT")


class OrganiationHistoryManager:
    def __init__(self):
        """Initialize the database manager with configuration."""
        self.db_config = {
            "dbname": DBNAME,
            "user": DBUSER,
            "password": DBPW,
            "host": DBHOST,
            "port": DBPORT
        }
        self.conn: Optional[Connection] = None

    def connect(self) -> None:
        """Establish a connection to the PostgreSQL database."""
        if not self.conn:
            try:
                self.conn = psycopg.connect(**self.db_config)
                print("Database connection established.")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to the database: {e}")

    def close(self) -> None:
        """Close the connection to the PostgreSQL database."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                print("Database connection closed.")
            except Exception as e:
                raise ConnectionError(f"Failed to close the database connection: {e}")

    def check_organisation_in_session(self, organisation_id: str) -> bool:
        """Check if a given organisation_id is present in the session_id of the message_store table.

        Args:
            organisation_id (str): The organisation ID to check.

        Returns:
            bool: True if found, False otherwise.
        """
        organisation_id = uuid.UUID(organisation_id.replace('-', '').ljust(32, '0'))
        query = """
            SELECT EXISTS (
                SELECT 1 FROM message_store
                WHERE session_id::TEXT LIKE %s  -- Cast session_id to TEXT before using LIKE
            )
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (f"%{organisation_id}%",))
                result = cur.fetchone()
                return result[0] if result else False
        except Exception as e:
            raise RuntimeError(f"Failed to check organisation in session: {e}")