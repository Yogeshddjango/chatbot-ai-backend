import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.database.organisation_vector_database import VectorStorePostgresVector

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
DIMENSION = int(os.getenv("DIMENSION", 768))
LOGGER = logging.getLogger(__name__)

class CreateDataEmbedding:
    def __init__(self, use_gpu: bool = False) -> None:
        self.embedding_model = OpenAIEmbeddings(
                                    model="text-embedding-3-large",
                                    api_key=OPENAI_API_KEY,
                                    dimensions=DIMENSION,
                                )
        self.text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=CHUNK_SIZE,
                                chunk_overlap=CHUNK_OVERLAP,
                                length_function=len,
                                is_separator_regex=False,
                            )

    def _clean_extraction_data(self, extraction_data: str) -> List[str]:
        lines = extraction_data.splitlines()

        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                cleaned_lines.append(stripped_line)
        
        return cleaned_lines

    def _get_docs_split(self, organisation_data: str):
        output_document = []
        cleaned_extraction_data = self._clean_extraction_data(organisation_data)
        if isinstance(cleaned_extraction_data, list):
            page_content = "\n".join(cleaned_extraction_data)
        else:
            page_content = cleaned_extraction_data
        metadata = {
            'format': "Text"
        }
        output_document.append(Document(metadata=metadata, page_content=page_content))
        return output_document    

    def _create_embedding_selection(
                self, data: dict
            ) -> dict[str, int]:
        doc_split = self._get_docs_split(data['organisation_data'])
        vector_store = VectorStorePostgresVector("organisation_embeddings", self.embedding_model)
        if not vector_store.check_if_record_exist(data['organisation_id'])['is_rec_exist']:
            status = vector_store.store_docs_to_collection(str(data['organisation_id']), doc_split)
        else:
            vector_store.delete_documents_from_collection(str(data['organisation_id']))
            status = vector_store.store_docs_to_collection(str(data['organisation_id']), doc_split)
            if status['status']:
                status["ai_embeddings_reason"] = f"Embeddings of {data['organisation_id']} is updated successfully"
        
        return status

    def check_if_all_files_can_be_sent(self, vector_store, organisation_id):
        import anthropic
        threshold = 20000
        try:
            all_docs = vector_store.similarity_search("", k=10 ** 9, filter={
                                                        'id': {'in': organisation_id}}    
                                                    )
            LOGGER.info("ALL DOCS LENGTH: %s" % len(all_docs))
            vo = anthropic.Client()
            total_tokens = 0
            for doc in all_docs:
                tokens = vo.count_tokens(doc.page_content)
                total_tokens += tokens
            LOGGER.info("Checking if all files can be sent, Total tokens: %s" % total_tokens)   
            return True if 0 < total_tokens < threshold else False
        except Exception as e:
            LOGGER.info("Exception in check_if_all_files_can_be_sent: %s" % e)
            return False