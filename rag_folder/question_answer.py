import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from organisation_prompts.prompts import *
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from memory_management.organisations_chat_history import ChatHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from database.organisation_retrieval_history import OrganiationHistoryManager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_TEMPERATURE = int(os.getenv("OPENAI_TEMPERATURE", 0))


class ChatBot:
    def __init__(self, temperature: float = 0.7):
        self.chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=OPENAI_TEMPERATURE)

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
                        {"question": data["user_query"]},
                        {"configurable": {"session_id": organisation_id}},
                    )
        return {'message': 'Query processed successfully', 'status': 200, 'question': data["user_query"], 'answer': generation.get('answer')}
