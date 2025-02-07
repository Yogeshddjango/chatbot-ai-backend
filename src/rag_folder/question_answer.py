import os
import uuid
import logging
from typing import Dict
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_openai import OpenAIEmbeddings
from src.organisation_prompts.prompts import *
from langchain.globals import set_verbose, set_debug
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.memory_management.organisations_chat_history import ChatHistory
from src.database.organisation_vector_database import VectorStorePostgresVector
from src.database.organisation_retrieval_history import OrganiationHistoryManager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_TEMPERATURE = int(os.getenv("OPENAI_TEMPERATURE", 0))
DIMENSION = int(os.getenv("DIMENSION", 768))
LOGGER = logging.getLogger(__name__)

# set_debug(True)
# set_verbose(True)


class ChatBot:
    def __init__(self, temperature: float = 0.7):
        self.chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=OPENAI_TEMPERATURE)
        self.embedding_model = OpenAIEmbeddings(
                                    model="text-embedding-3-large",
                                    api_key=OPENAI_API_KEY,
                                    dimensions=DIMENSION,
                                )

    def _vectorstore_retriever(self, organisation_id):
        try:
            vector_store = VectorStorePostgresVector("organisation_embeddings", self.embedding_model)
            return vector_store.get_or_create_collection().as_retriever(
                                                                search_type="mmr",
                                                                search_kwargs={"metadata.id": str(organisation_id)},
                                                            )
        except Exception:
            return None

    def get_response(self, data: dict) -> str:
        """
        Get a response from the chatbot.

        :param data['user_query']: The message input from the user.
        :return: The chatbot's response in JSON Format.
        """
        history_db_manager = OrganiationHistoryManager()
        history_db_manager.connect()
        chat_history = ChatHistory(data['organisation_id'])
        chat_history_object = chat_history.session_based_chat_history()
        organisation_id = uuid.UUID(data['organisation_id'].replace('-', '').ljust(32, '0'))
        if not history_db_manager.check_organisation_in_session(data['organisation_id']):
            chat_history_object.add_user_message(HumanMessage(
                                                    name=data['organisation_id'],
                                                    content="oragnisation_data",
                                                ))
            chat_history_object.add_ai_message(AIMessage(
                                                    name=data['organisation_id'],
                                                    content="oragnisation_data",
                                                ))
        retriever = self._vectorstore_retriever(data['organisation_id'])
        documents = retriever.get_relevant_documents(data['user_query'])
        filtered_docs = [doc for doc in documents if doc.id == str(data['organisation_id'])]

        act_prompt = ACT_PROMPT
        act_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                act_prompt,
                            ),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{question}"),
                        ]
                    )
        rag_chain = act_prompt | self.chat_model | JsonOutputParser()

        chain_with_message_history = RunnableWithMessageHistory(
                                rag_chain,
                                lambda session_id: chat_history_object,
                                input_messages_key="question",
                                history_messages_key="chat_history",
                            )
        generation = chain_with_message_history.invoke(
                {"question": data['user_query'], "context": filtered_docs},
                {"configurable": {"session_id": data['organisation_id']}},
            )

        return {'message': 'Query processed successfully', 'status': 200, 'question': data["user_query"], 'answer': generation.get('answer')}
