import os
import psycopg
from datetime import datetime
from dotenv import load_dotenv
from psycopg import Connection
from typing import Any, Optional, Dict

load_dotenv()

DBNAME = os.getenv("DBNAME")
DBUSER = os.getenv("DBUSER")
DBPW = os.getenv("DBPW")
DBHOST = os.getenv("DBHOST")
DBPORT = os.getenv("DBPORT")

class DatabaseManager:
    def __init__(self):
        """Initialize the database manager with configuration.

        Args:
            db_config (Dict[str, str]): A dictionary containing database connection details like host, database, user, and password.
        """
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

    def create_table_if_not_exists(self) -> None:
        """Create the table if it does not already exist."""
        query = (
            """
            CREATE TABLE IF NOT EXISTS organisation_data (
                organisation_id TEXT PRIMARY KEY,
                organisation_data TEXT NOT NULL,
                ai_embeddings_status TEXT NOT NULL,
                ai_embeddings_reason TEXT,
                created_at TIMESTAMP NOT NULL,
                modified_at TIMESTAMP NOT NULL
            )
            """
        )
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                self.conn.commit()
                print("Table created or already exists.")
        except Exception as e:
            raise RuntimeError(f"Failed to create table: {e}")

    def insert_or_update_data(self, data: Dict[str, Any]) -> None:
        """Insert or update data in the table.

        Args:
            data (Dict[str, Any]): A dictionary containing the data to be inserted or updated.
        """
        query_check = (
            "SELECT organisation_id FROM organisation_data WHERE organisation_id = %s"
        )
        query_insert = (
            """
            INSERT INTO organisation_data (
                organisation_id, organisation_data, ai_embeddings_status,
                ai_embeddings_reason, created_at, modified_at
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """
        )
        query_update = (
            """
            UPDATE organisation_data
            SET ai_embeddings_status = %s,
                ai_embeddings_reason = %s,
                modified_at = %s
            WHERE organisation_id = %s
            """
        )

        now = datetime.now()
        try:
            with self.conn.cursor() as cur:
                cur.execute(query_check, (data["organisation_id"],))
                result = cur.fetchone()

                if result:
                    # Update existing record
                    cur.execute(
                        query_update,
                        (
                            data["ai_embeddings_status"],
                            data["ai_embeddings_reason"],
                            now,
                            data["organisation_id"],
                        ),
                    )
                    print("Record updated.")
                else:
                    # Insert new record
                    cur.execute(
                        query_insert,
                        (
                            data["organisation_id"],
                            data["organisation_data"],
                            data["ai_embeddings_status"],
                            data["ai_embeddings_reason"],
                            now,
                            now,
                        ),
                    )
                    print("Record inserted.")

                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to insert or update data: {e}")

# # Example Usage
# if __name__ == "__main__":
#     db_manager = DatabaseManager()
#     try:
#         db_manager.connect()
#         db_manager.create_table_if_not_exists()

#     finally:
#         db_manager.close()
