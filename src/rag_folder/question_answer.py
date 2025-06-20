# # import os
# # import uuid
# # import logging
# # from typing import Dict
# # from dotenv import load_dotenv
# # from typing_extensions import TypedDict
# # from langchain_openai import ChatOpenAI
# # from langgraph.graph import END, StateGraph
# # from langchain_openai import OpenAIEmbeddings
# # from src.organisation_prompts.prompts import *
# # from langchain.globals import set_verbose, set_debug
# # from langchain.schema import HumanMessage, AIMessage
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.pydantic_v1 import BaseModel, Field
# # from langchain_core.output_parsers import JsonOutputParser
# # from langchain_core.runnables.history import RunnableWithMessageHistory
# # from src.memory_management.organisations_chat_history import ChatHistory
# # from src.database.organisation_vector_database import VectorStorePostgresVector
# # from src.database.organisation_retrieval_history import OrganiationHistoryManager
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate

# # from langchain_core.tools import tool

# # @tool
# # def call_task_tool(task_creation: bool) -> int:
# #     """provide the status of task creation.

# #     Args:
# #         task_creation: True for the task creation and False for the task creation. 
# #     """
# #     return task_creation


# # load_dotenv()

# # OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# # OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# # OPENAI_TEMPERATURE = int(os.getenv("OPENAI_TEMPERATURE", 0))
# # DIMENSION = int(os.getenv("DIMENSION", 768))
# # LOGGER = logging.getLogger(__name__)

# # set_debug(True)
# # set_verbose(True)


# # class ChatBot:
# #     def __init__(self, temperature: float = 0.7):
# #         self.chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=OPENAI_TEMPERATURE)
# #         self.embedding_model = OpenAIEmbeddings(
# #                                     model="text-embedding-3-large",
# #                                     api_key=OPENAI_API_KEY,
# #                                     dimensions=DIMENSION,
# #                                 )
# #         self.chat_model_json = self.chat_model.bind(response_format={"type": "json_object"})


# #     def _vectorstore_retriever(self, organisation_id):
# #         try:
# #             vector_store = VectorStorePostgresVector("organisation_embeddings", self.embedding_model)
# #             return vector_store.get_or_create_collection().as_retriever(
# #                                                                 search_type="mmr",
# #                                                                 search_kwargs={"metadata.id": str(organisation_id)},
# #                                                             )
# #         except Exception:
# #             return None

# #     def get_response(self, data: dict) -> str:
# #         """
# #         Get a response from the chatbot.

# #         :param data['user_query']: The message input from the user.
# #         :return: The chatbot's response in JSON Format.
# #         """
# #         history_db_manager = OrganiationHistoryManager()
# #         history_db_manager.connect()
# #         chat_history = ChatHistory(data['organisation_id'])
# #         chat_history_object = chat_history.session_based_chat_history()
# #         organisation_id = uuid.UUID(data['organisation_id'].replace('-', '').ljust(32, '0'))
# #         if not history_db_manager.check_organisation_in_session(data['organisation_id']):
# #             chat_history_object.add_user_message(HumanMessage(
# #                                                     name=data['organisation_id'],
# #                                                     content="oragnisation_data",
# #                                                 ))
# #             chat_history_object.add_ai_message(AIMessage(
# #                                                     name=data['organisation_id'],
# #                                                     content="oragnisation_data",
# #                                                 ))
# #         retriever = self._vectorstore_retriever(data['organisation_id'])
# #         documents = retriever.get_relevant_documents(data['user_query'])
# #         filtered_docs = [doc for doc in documents if doc.id == str(data['organisation_id'])]
# #         if filtered_docs != []:
# #             for i in range(len(filtered_docs)):
# #                 content = "".join(filtered_docs[i].page_content)
# #         else:
# #             content = ""

# #         examples = [
# #             # Example 1: User requests task creation explicitly
# #             HumanMessage("Can you create a task for this?", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "Would you like to create a task for this?"}',
# #                 name="example_assistant",
# #             ),
# #             HumanMessage("Yes", name="example_user"),
# #             AIMessage(
# #                 "",
# #                 name="example_assistant",
# #                 tool_calls=[
# #                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "1"}
# #                 ],
# #             ),
# #             ToolMessage("Task created successfully.", tool_call_id="1"),
# #             AIMessage(
# #                 '{"answer": "Alright, I\'m creating a task for you."}',
# #                 name="example_assistant",
# #             ),

# #             # Example 2: User implicitly confirms task creation
# #             HumanMessage("Please go ahead and create a task.", name="example_user"),
# #             AIMessage(
# #                 "",
# #                 name="example_assistant",
# #                 tool_calls=[
# #                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "2"}
# #                 ],
# #             ),
# #             ToolMessage("Task created successfully.", tool_call_id="2"),
# #             AIMessage(
# #                 '{"answer": "Alright, I\'m creating a task for you."}',
# #                 name="example_assistant",
# #             ),

# #             # Example 3: User declines task creation
# #             HumanMessage("Can you create a task for this?", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "Would you like to create a task for this?"}',
# #                 name="example_assistant",
# #             ),
# #             HumanMessage("No", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "thankyou"}',
# #                 name="example_assistant",
# #             ),

# #             # Example 4: Agreement leading to task creation
# #             HumanMessage("I agree with this.", name="example_user"),
# #             AIMessage(
# #                 "",
# #                 name="example_assistant",
# #                 tool_calls=[
# #                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "3"}
# #                 ],
# #             ),
# #             ToolMessage("Task created successfully.", tool_call_id="3"),
# #             AIMessage(
# #                 '{"answer": "Alright, I\'m creating a task for you."}',
# #                 name="example_assistant",
# #             ),

# #             # Example 5: Disagreement, no task creation
# #             HumanMessage("I disagree with this.", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "thankyou"}',
# #                 name="example_assistant",
# #             ),
# #         ]

# #         self.chat_model_json  = self.chat_model_json.bind_tools(tools)
# #         act_prompt = ACT_PROMPT
# #         act_prompt = ChatPromptTemplate.from_messages(
# #                         [
# #                             (
# #                                 "system",
# #                                 act_prompt,
# #                             ),
# #                             *examples,
# #                             MessagesPlaceholder(variable_name="chat_history"),
# #                             ("human", "{question}"),
# #                         ]
# #                     )
# #         rag_chain = act_prompt | self.chat_model_json | JsonOutputParser()

# #         chain_with_message_history = RunnableWithMessageHistory(
# #                                 rag_chain,
# #                                 lambda session_id: chat_history_object,
# #                                 input_messages_key="question",
# #                                 history_messages_key="chat_history",
# #                             )
# #         import pdb;pdb.set_trace()
# #         generation = chain_with_message_history.invoke(
# #                 {"question": data['user_query'], "context": content},
# #                 {"configurable": {"session_id": data['organisation_id']}},
# #             )
# #         return {'message': 'Query processed successfully', 'status': 200, 'question': data["user_query"], 'answer': generation.get('answer'), 'task_creation': generation.get('task_creation')}


# import os
# import uuid
# import logging
# from typing import Dict
# from dotenv import load_dotenv
# from typing_extensions import TypedDict
# from langchain_openai import ChatOpenAI
# from langgraph.graph import END, StateGraph
# from langchain_openai import OpenAIEmbeddings
# from src.organisation_prompts.prompts import *
# from langchain.globals import set_verbose, set_debug
# from langchain.schema import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from src.memory_management.organisations_chat_history import ChatHistory
# from src.database.organisation_vector_database import VectorStorePostgresVector
# from src.database.organisation_retrieval_history import OrganiationHistoryManager
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate

# load_dotenv()

# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# OPENAI_TEMPERATURE = int(os.getenv("OPENAI_TEMPERATURE", 0))
# DIMENSION = int(os.getenv("DIMENSION", 768))
# LOGGER = logging.getLogger(__name__)

# # set_debug(True)
# # set_verbose(True)


# class ChatBot:
#     def __init__(self, temperature: float = 0.7):
#         self.chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=OPENAI_TEMPERATURE)
#         self.embedding_model = OpenAIEmbeddings(
#                                     model="text-embedding-3-large",
#                                     api_key=OPENAI_API_KEY,
#                                     dimensions=DIMENSION,
#                                 )
#         # self.chat_model_json = self.chat_model.bind(response_format={"type": "json_object"})

#     def _vectorstore_retriever(self, organisation_id):
#         try:
#             vector_store = VectorStorePostgresVector("organisation_embeddings", self.embedding_model)
#             return vector_store.get_or_create_collection().as_retriever(
#                                                                 search_type="mmr",
#                                                                 search_kwargs={"metadata.id": str(organisation_id)},
#                                                             )
#         except Exception:
#             return None

#     def get_response(self, data: dict) -> str:
#         """
#         Get a response from the chatbot.

#         :param data['user_query']: The message input from the user.
#         :return: The chatbot's response in JSON Format.
#         """
#         history_db_manager = OrganiationHistoryManager()
#         history_db_manager.connect()
#         chat_history = ChatHistory(data['organisation_id'])
#         chat_history_object = chat_history.session_based_chat_history()
#         organisation_id = uuid.UUID(data['organisation_id'].replace('-', '').ljust(32, '0'))
#         if not history_db_manager.check_organisation_in_session(data['organisation_id']):
#             chat_history_object.add_user_message(HumanMessage(
#                                                     name=data['organisation_id'],
#                                                     content="oragnisation_data",
#                                                 ))
#             chat_history_object.add_ai_message(AIMessage(
#                                                     name=data['organisation_id'],
#                                                     content="oragnisation_data",
#                                                 ))
#         retriever = self._vectorstore_retriever(data['organisation_id'])
#         try:
#             documents = retriever.get_relevant_documents(data['user_query'])
#         except Exception:
#             documents = []
#         filtered_docs = [doc for doc in documents if doc.id == str(data['organisation_id'])]

#         act_prompt = ACT_PROMPT
#         act_prompt = ChatPromptTemplate.from_messages(
#                         [
#                             (
#                                 "system",
#                                 act_prompt,
#                             ),
#                             MessagesPlaceholder(variable_name="chat_history"),
#                             ("human", "{question}"),
#                         ]
#                     )
#         rag_chain = act_prompt | self.chat_model | JsonOutputParser()

#         chain_with_message_history = RunnableWithMessageHistory(
#                                 rag_chain,
#                                 lambda session_id: chat_history_object,
#                                 input_messages_key="question",
#                                 history_messages_key="chat_history",
#                             )
#         generation = chain_with_message_history.invoke(
#                 {"question": data['user_query'], "context": filtered_docs},
#                 {"configurable": {"session_id": data['organisation_id']}},
#             )

#         return {'message': 'Query processed successfully', 'status': 200, 'question': data["user_query"], 'answer': generation.get('answer')}

# import os
# # import uuid
# # import logging
# # from typing import Dict
# # from dotenv import load_dotenv
# # from typing_extensions import TypedDict
# # from langchain_openai import ChatOpenAI
# # from langgraph.graph import END, StateGraph
# # from langchain_openai import OpenAIEmbeddings
# # from src.organisation_prompts.prompts import *
# # from langchain.globals import set_verbose, set_debug
# # from langchain.schema import HumanMessage, AIMessage
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.pydantic_v1 import BaseModel, Field
# # from langchain_core.output_parsers import JsonOutputParser
# # from langchain_core.runnables.history import RunnableWithMessageHistory
# # from src.memory_management.organisations_chat_history import ChatHistory
# # from src.database.organisation_vector_database import VectorStorePostgresVector
# # from src.database.organisation_retrieval_history import OrganiationHistoryManager
# # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate

# # from langchain_core.tools import tool

# # @tool
# # def call_task_tool(task_creation: bool) -> int:
# #     """provide the status of task creation.

# #     Args:
# #         task_creation: True for the task creation and False for the task creation. 
# #     """
# #     return task_creation


# # load_dotenv()

# # OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# # OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# # OPENAI_TEMPERATURE = int(os.getenv("OPENAI_TEMPERATURE", 0))
# # DIMENSION = int(os.getenv("DIMENSION", 768))
# # LOGGER = logging.getLogger(__name__)

# # set_debug(True)
# # set_verbose(True)


# # class ChatBot:
# #     def __init__(self, temperature: float = 0.7):
# #         self.chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=OPENAI_TEMPERATURE)
# #         self.embedding_model = OpenAIEmbeddings(
# #                                     model="text-embedding-3-large",
# #                                     api_key=OPENAI_API_KEY,
# #                                     dimensions=DIMENSION,
# #                                 )
# #         self.chat_model_json = self.chat_model.bind(response_format={"type": "json_object"})


# #     def _vectorstore_retriever(self, organisation_id):
# #         try:
# #             vector_store = VectorStorePostgresVector("organisation_embeddings", self.embedding_model)
# #             return vector_store.get_or_create_collection().as_retriever(
# #                                                                 search_type="mmr",
# #                                                                 search_kwargs={"metadata.id": str(organisation_id)},
# #                                                             )
# #         except Exception:
# #             return None

# #     def get_response(self, data: dict) -> str:
# #         """
# #         Get a response from the chatbot.

# #         :param data['user_query']: The message input from the user.
# #         :return: The chatbot's response in JSON Format.
# #         """
# #         history_db_manager = OrganiationHistoryManager()
# #         history_db_manager.connect()
# #         chat_history = ChatHistory(data['organisation_id'])
# #         chat_history_object = chat_history.session_based_chat_history()
# #         organisation_id = uuid.UUID(data['organisation_id'].replace('-', '').ljust(32, '0'))
# #         if not history_db_manager.check_organisation_in_session(data['organisation_id']):
# #             chat_history_object.add_user_message(HumanMessage(
# #                                                     name=data['organisation_id'],
# #                                                     content="oragnisation_data",
# #                                                 ))
# #             chat_history_object.add_ai_message(AIMessage(
# #                                                     name=data['organisation_id'],
# #                                                     content="oragnisation_data",
# #                                                 ))
# #         retriever = self._vectorstore_retriever(data['organisation_id'])
# #         documents = retriever.get_relevant_documents(data['user_query'])
# #         filtered_docs = [doc for doc in documents if doc.id == str(data['organisation_id'])]
# #         if filtered_docs != []:
# #             for i in range(len(filtered_docs)):
# #                 content = "".join(filtered_docs[i].page_content)
# #         else:
# #             content = ""

# #         examples = [
# #             # Example 1: User requests task creation explicitly
# #             HumanMessage("Can you create a task for this?", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "Would you like to create a task for this?"}',
# #                 name="example_assistant",
# #             ),
# #             HumanMessage("Yes", name="example_user"),
# #             AIMessage(
# #                 "",
# #                 name="example_assistant",
# #                 tool_calls=[
# #                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "1"}
# #                 ],
# #             ),
# #             ToolMessage("Task created successfully.", tool_call_id="1"),
# #             AIMessage(
# #                 '{"answer": "Alright, I\'m creating a task for you."}',
# #                 name="example_assistant",
# #             ),

# #             # Example 2: User implicitly confirms task creation
# #             HumanMessage("Please go ahead and create a task.", name="example_user"),
# #             AIMessage(
# #                 "",
# #                 name="example_assistant",
# #                 tool_calls=[
# #                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "2"}
# #                 ],
# #             ),
# #             ToolMessage("Task created successfully.", tool_call_id="2"),
# #             AIMessage(
# #                 '{"answer": "Alright, I\'m creating a task for you."}',
# #                 name="example_assistant",
# #             ),

# #             # Example 3: User declines task creation
# #             HumanMessage("Can you create a task for this?", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "Would you like to create a task for this?"}',
# #                 name="example_assistant",
# #             ),
# #             HumanMessage("No", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "thankyou"}',
# #                 name="example_assistant",
# #             ),

# #             # Example 4: Agreement leading to task creation
# #             HumanMessage("I agree with this.", name="example_user"),
# #             AIMessage(
# #                 "",
# #                 name="example_assistant",
# #                 tool_calls=[
# #                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "3"}
# #                 ],
# #             ),
# #             ToolMessage("Task created successfully.", tool_call_id="3"),
# #             AIMessage(
# #                 '{"answer": "Alright, I\'m creating a task for you."}',
# #                 name="example_assistant",
# #             ),

# #             # Example 5: Disagreement, no task creation
# #             HumanMessage("I disagree with this.", name="example_user"),
# #             AIMessage(
# #                 '{"answer": "thankyou"}',
# #                 name="example_assistant",
# #             ),
# #         ]

# #         self.chat_model_json  = self.chat_model_json.bind_tools(tools)
# #         act_prompt = ACT_PROMPT
# #         act_prompt = ChatPromptTemplate.from_messages(
# #                         [
# #                             (
# #                                 "system",
# #                                 act_prompt,
# #                             ),
# #                             *examples,
# #                             MessagesPlaceholder(variable_name="chat_history"),
# #                             ("human", "{question}"),
# #                         ]
# #                     )
# #         rag_chain = act_prompt | self.chat_model_json | JsonOutputParser()

# #         chain_with_message_history = RunnableWithMessageHistory(
# #                                 rag_chain,
# #                                 lambda session_id: chat_history_object,
# #                                 input_messages_key="question",
# #                                 history_messages_key="chat_history",
# #                             )
# #         import pdb;pdb.set_trace()
# #         generation = chain_with_message_history.invoke(
# #                 {"question": data['user_query'], "context": content},
# #                 {"configurable": {"session_id": data['organisation_id']}},
# #             )
# #         return {'message': 'Query processed successfully', 'status': 200, 'question': data["user_query"], 'answer': generation.get('answer'), 'task_creation': generation.get('task_creation')}


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
        # self.chat_model_json = self.chat_model.bind(response_format={"type": "json_object"})

    def _vectorstore_retriever(self, organisation_id):
        try:
            vector_store = VectorStorePostgresVector("organisation_embeddings", self.embedding_model)
            # import pdb;pdb.set_trace()
            return vector_store.get_or_create_collection().as_retriever(
                                                                search_type="mmr",
                                                                search_kwargs={
                                                                    "filter": {"id": {"$eq": str(organisation_id)}}
                                                                }
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
        try:
            documents = retriever.get_relevant_documents(data['user_query'])
        except Exception:
            documents = []
        # filtered_docs = [doc for doc in documents if doc.id == str(data['organisation_id'])]

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

        # import pdb;pdb.set_trace()
        generation = chain_with_message_history.invoke(
                {"question": data['user_query'], "context": documents},
                {"configurable": {"session_id": data['organisation_id']}},
            )

        return {'message': 'Query processed successfully', 'status': 200, 'question': data["user_query"], 'answer': generation.get('answer'), 'task_creation': generation.get('task_creation')}

# import os
# import uuid
# import logging
# from typing import Dict
# from dotenv import load_dotenv
# from typing_extensions import TypedDict
# from langchain_openai import ChatOpenAI
# from langgraph.graph import END, StateGraph
# from langchain_openai import OpenAIEmbeddings
# from src.organisation_prompts.prompts import *
# from langchain.globals import set_verbose, set_debug
# from langchain.schema import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from src.memory_management.organisations_chat_history import ChatHistory
# from src.database.organisation_vector_database import VectorStorePostgresVector
# from src.database.organisation_retrieval_history import OrganiationHistoryManager
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate
# from langchain_core.messages import ToolMessage
# from langchain_core.tools import tool

# @tool
# def call_task_tool(task_creation: bool) -> int:
#     """provide the status of task creation.
#     Args:
#         task_creation: True for the task creation and False for the task creation. 
#     """
#     return task_creation

# load_dotenv()

# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# OPENAI_TEMPERATURE = int(os.getenv("OPENAI_TEMPERATURE", 0))
# DIMENSION = int(os.getenv("DIMENSION", 768))
# LOGGER = logging.getLogger(__name__)

# set_debug(True)
# set_verbose(True)

# class ChatBot:
#     def __init__(self, temperature: float = 0.7):
#         self.chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=OPENAI_TEMPERATURE)
#         self.embedding_model = OpenAIEmbeddings(
#                                     model="text-embedding-3-large",
#                                     api_key=OPENAI_API_KEY,
#                                     dimensions=DIMENSION,
#                                 )
#         self.chat_model_json = self.chat_model.bind(response_format={"type": "json_object"})
#     def _vectorstore_retriever(self, organisation_id):
#         try:
#             vector_store = VectorStorePostgresVector("organisation_embeddings", self.embedding_model)
#             return vector_store.get_or_create_collection().as_retriever(
#                                                                 search_type="mmr",
#                                                                 search_kwargs={"metadata.id": str(organisation_id)},
#                                                             )
#         except Exception:
#             return None
#     def get_response(self, data: dict) -> str:
#         """
#         Get a response from the chatbot.
#         :param data['user_query']: The message input from the user.
#         :return: The chatbot's response in JSON Format.
#         """
#         history_db_manager = OrganiationHistoryManager()
#         history_db_manager.connect()
#         chat_history = ChatHistory(data['organisation_id'])
#         chat_history_object = chat_history.session_based_chat_history()
#         organisation_id = uuid.UUID(data['organisation_id'].replace('-', '').ljust(32, '0'))
#         if not history_db_manager.check_organisation_in_session(data['organisation_id']):
#             chat_history_object.add_user_message(HumanMessage(
#                                                     name=data['organisation_id'],
#                                                     content="oragnisation_data",
#                                                 ))
#             chat_history_object.add_ai_message(AIMessage(
#                                                     name=data['organisation_id'],
#                                                     content="oragnisation_data",
#                                                 ))
#         retriever = self._vectorstore_retriever(data['organisation_id'])
#         try:
#             documents = retriever.get_relevant_documents(data['user_query'])
#         except Exception:
#             documents = []
#         filtered_docs = [doc for doc in documents if doc.id == str(data['organisation_id'])]
#         if filtered_docs != []:
#             for i in range(len(filtered_docs)):
#                 content = "".join(filtered_docs[i].page_content)
#         else:
#             content = ""
    
#         examples = [
#             # Example 1: User requests task creation explicitly
#             HumanMessage("Can you create a task for this?", name="example_user"),
#             AIMessage(
#                 '{"answer": "Would you like to create a task for this?"}',
#                 name="example_assistant",
#                 tool_calls=[
#                     {"name": "call_task_tool", "args": {"task_creation": False}, "id": "1"}
#                 ],
#             ),
#             ToolMessage("False", tool_call_id="1"),
#             HumanMessage("Yes", name="example_user"),
#             AIMessage(
#                 '{"answer": "I\'m creating a task for you."}',
#                 name="example_assistant",
#                 tool_calls=[
#                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "2"}
#                 ],
#             ),
#             ToolMessage("True", tool_call_id="2"),
#             # Example 2: User implicitly confirms task creation
#             HumanMessage("Please go ahead and create a task.", name="example_user"),
#             AIMessage(
#                 '{"answer": "I\'m creating a task for you now."}',
#                 name="example_assistant",
#                 tool_calls=[
#                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "3"}
#                 ],
#             ),
#             ToolMessage("True", tool_call_id="3"),
#             # Example 3: User declines task creation
#             HumanMessage("No, I don't want to create a task.", name="example_user"),
#             AIMessage(
#                 '{"answer": "Understood, I won\'t create a task."}',
#                 name="example_assistant",
#                 tool_calls=[
#                     {"name": "call_task_tool", "args": {"task_creation": False}, "id": "4"}
#                 ],
#             ),
#             ToolMessage("False", tool_call_id="4"),
#             # Example 4: General question
#             HumanMessage("What's the weather today?", name="example_user"),
#             AIMessage(
#                 '{"answer": "I don\'t have access to current weather information. Would you like to create a task to check the weather?"}',
#                 name="example_assistant",
#                 tool_calls=[
#                     {"name": "call_task_tool", "args": {"task_creation": False}, "id": "5"}
#                 ],
#             ),
#             ToolMessage("False", tool_call_id="5"),
#             # Example 5: User expresses agreement
#             HumanMessage("That sounds good to me.", name="example_user"),
#             AIMessage(
#                 '{"answer": "Great! I\'ll create that task for you."}',
#                 name="example_assistant",
#                 tool_calls=[
#                     {"name": "call_task_tool", "args": {"task_creation": True}, "id": "6"}
#                 ],
#             ),
#             ToolMessage("True", tool_call_id="6"),
#         ]
        
#         act_prompt = ACT_PROMPT
#         tools = [call_task_tool]
#         self.chat_model_json = self.chat_model_json.bind_tools(tools, tool_choice={"type": "function", "function": {"name": "call_task_tool"}})
        
#         prompt = ChatPromptTemplate.from_messages(
#                         [
#                             (
#                                 "system",
#                                 act_prompt,
#                             ),
#                             *examples,
#                             MessagesPlaceholder(variable_name="chat_history"),
#                             ("human", "{question}"),
#                         ]
#                     )
        
#         rag_chain = prompt | self.chat_model_json
#         chain_with_message_history = RunnableWithMessageHistory(
#                                 rag_chain,
#                                 lambda session_id: chat_history_object,
#                                 input_messages_key="question",
#                                 history_messages_key="chat_history",
#                             )
        
#         import pdb;pdb.set_trace()
#         try:
#             generation = chain_with_message_history.invoke(
#                     {"question": data['user_query'], "context": content},
#                     {"configurable": {"session_id": data['organisation_id']}},
#                 )
            
#             # Extract task_creation value from tool calls
#             task_creation = False
#             if hasattr(generation, 'tool_calls') and generation.tool_calls:
#                 for tool_call in generation.tool_calls:
#                     if tool_call.get('name') == 'call_task_tool':
#                         task_creation = tool_call.get('args', {}).get('task_creation', False)
            
#             # Parse the JSON content if available
#             answer = None
#             try:
#                 if generation.content:
#                     content_json = json.loads(generation.content)
#                     answer = content_json.get('answer')
#             except:
#                 answer = generation.content
            
#             return {
#                 'message': 'Query processed successfully', 
#                 'status': 200, 
#                 'question': data["user_query"], 
#                 'answer': answer, 
#                 'task_creation': task_creation
#             }
#         except Exception as e:
#             # Handle any other errors that might occur
#             return {
#                 'message': f'Error occurred: {str(e)}', 
#                 'status': 500, 
#                 'question': data["user_query"], 
#                 'answer': "I'm sorry, I encountered an error processing your request.", 
#                 'task_creation': False
#             }