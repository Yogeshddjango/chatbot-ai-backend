import os
import uuid
import psycopg2
from dotenv import load_dotenv
from typing import List, Any, Dict
from langchain_postgres import PGVector
from psycopg2.extensions import register_adapter, AsIs

load_dotenv()

DBNAME = os.getenv("DBNAME")
DBUSER = os.getenv("DBUSER")
DBPW = os.getenv("DBPW")
DBHOST = os.getenv("DBHOST")
DBPORT = os.getenv("DBPORT")
CONNECTION_SETTINGS = f"postgresql+psycopg2://{DBUSER}:{DBPW}@{DBHOST}:{DBPORT}/{DBNAME}"

def adapt_uuid(uuid_val: uuid.UUID) -> AsIs:
    return AsIs(f"'{str(uuid_val)}'")

register_adapter(uuid.UUID, adapt_uuid)

class VectorStorePostgresVector:
    def __init__(self, collection_name: str, embeddings: Any) -> None:
        """
        Initializes the VectorStorePostgresVector object.

        Args:
            collection_name (str): The name of the collection.
            embeddings (Any): The embeddings function.
            connection_settings (str): The connection settings for the database.
        """
        self.collection_name: str = collection_name
        self.connection: str = CONNECTION_SETTINGS
        self.embeddings: Any = embeddings

    def get_or_create_collection(self) -> PGVector:
        """
        Retrieves an existing collection or creates a new one.

        Returns:
            PGVector: The PGVector collection.
        """
        return PGVector(
                        embeddings=self.embeddings,
                        collection_name=self.collection_name,
                        connection=CONNECTION_SETTINGS,
                        use_jsonb=True,
                        create_extension=True
                    )

    def store_docs_to_collection(self, organisation_id: str, docs: List[Any]) -> bool:
        """
        Stores documents into the vector store collection.

        Args:
            document_id (str): The ID of the document.
            docs (List[Any]): The list of documents to store.
            document_path (str): The path to the document.

        Returns:
            bool: True if documents are stored successfully.
        """
        try:
            vector_db = self.get_or_create_collection()

            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []

            for doc in docs:
                metadata: Dict[str, Any] = {
                    'id': organisation_id
                }
                doc.metadata = metadata
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
                ids.append(f"{organisation_id}")

            vector_db.add_texts(texts, metadatas, ids=ids)
            return {
                    "status": True,
                    "organisation_id": organisation_id,
                    "ai_embeddings_status": "Completed",
                    "ai_embeddings_reason": f"Embeddings of organisation id: {organisation_id} is generated successfully"
                }
        except Exception as e:
            return {
                    "status": False,
                    "organisation_id": organisation_id,
                    "ai_embeddings_status": "Failed",
                    "ai_embeddings_reason": f"{e}"
                }

    def delete_documents_from_collection(self, organisation_id: str) -> None:
        """
        Deletes documents from the collection.

        Args:
            document_id (str): The ID of the document to delete.
        """
        vector_db = self.get_or_create_collection()
        vector_db.delete([organisation_id])

    def check_if_record_exist(self, organisation_id: str) -> Dict[str, bool]:
        """
        Checks if a record with the given document ID exists in the database.

        Args:
            document_id (str): The ID of the document to check.

        Returns:
            Dict[str, bool]: A dictionary indicating whether the record exists.
        """
        is_rec_exist: bool = False
        try:
            with psycopg2.connect(host=DBHOST, database=DBNAME, user=DBUSER, password=DBPW) as db:
                cursor = db.cursor()
                cursor.execute("SELECT EXISTS (SELECT 1 FROM langchain_pg_embedding WHERE id = %s LIMIT 1)", (organisation_id,))
                record = cursor.fetchone()
                is_rec_exist = record[0] if record else False
        except Exception as e:
            print(f"Error checking record existence: {e}")
        return {"is_rec_exist": is_rec_exist}

    # def delete_file_embeddings_from_collection(self, pdf_id: str) -> Dict[str, bool]:
    #     """
    #     Deletes file embeddings associated with the given PDF ID.

    #     Args:
    #         pdf_id (str): The ID of the PDF whose embeddings need to be deleted.

    #     Returns:
    #         Dict[str, bool]: A dictionary indicating whether the record was deleted successfully.
    #     """
    #     is_rec_exist: bool = False
    #     try:
    #         with psycopg2.connect(host=DBHOST, database=DBNAME, user=DBUSER, password=DBPW) as db:
    #             cursor = db.cursor()
    #             cursor.execute("DELETE FROM langchain_pg_embedding WHERE custom_id = %s", (pdf_id,))
    #             is_rec_exist = True
    #     except Exception as e:
    #         print(f"Error deleting embeddings: {e}")
    #     return {'is_rec_exist': is_rec_exist}