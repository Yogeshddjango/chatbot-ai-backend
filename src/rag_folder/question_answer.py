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

set_debug(True)
set_verbose(True)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeQuestion(BaseModel):
    """Binary score if question is mathematical equation or not."""

    binary_score: str = Field(description="Question is mathematical equation or not, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


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
                                                                search_kwargs={"id": organisation_id},
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

        structured_llm_document_grader = self.chat_model.with_structured_output(GradeDocuments)
        grade_documents_prompt = GRADE_DOCUMENTS_PROMPT
        grade_documents_prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    grade_documents_prompt,
                                ),
                                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
                            ]
                        )   
                                
        retrieval_grader = grade_documents_prompt | structured_llm_document_grader

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

        structured_llm__hallucination_grader = self.chat_model.with_structured_output(GradeHallucinations)
        hallucination_prompt = HALLUCINATION_PROMPT
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", hallucination_prompt),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        hallucination_grader = hallucination_prompt | structured_llm__hallucination_grader
        
        structured_llm_Answer_grader = self.chat_model.with_structured_output(GradeAnswer)
        answer_prompt = ANSWER_PROMPT
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_prompt),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        answer_grader = answer_prompt | structured_llm_Answer_grader

        chain_with_message_history = RunnableWithMessageHistory(
                                rag_chain,
                                lambda session_id: chat_history_object,
                                input_messages_key="question",
                                history_messages_key="chat_history",
                            )

        def retrieve(state):
            """
            Retrieve documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """

            state_dict = state["keys"]
            question = state_dict["question"]
            documents = retriever.get_relevant_documents(question)
            return {"keys": {"documents": documents, "question": question}}
        def grade_documents(state):
            """
            Determines whether the retrieved documents are relevant to the question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with only filtered relevant documents
            """
            state_dict = state["keys"]
            question = state_dict["question"]
            documents = state_dict["documents"]
            filtered_docs = []
            for d in documents:
                score = retrieval_grader.invoke({"question": question, "document": d.page_content})
                grade = score.binary_score
                if grade == "yes":
                    LOGGER.info("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    LOGGER.info("---GRADE: DOCUMENT NOT RELEVANT---")
                    continue
            return {"keys": {"documents": filtered_docs, "question": question}}
        def generate(state):
            """
            Generate answer

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            
            state_dict = state["keys"]
            question = state_dict["question"]
            documents = state_dict["documents"]
            generation = chain_with_message_history.invoke(
                {"question": question, "context": documents},
                {"configurable": {"session_id": data['organisation_id']}},
            )
            return {"keys": {"documents": documents, "question": question, "generation": generation}}
        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Decision for next node to call
            """
            try:
                state_dict = state["keys"]
                question = state_dict["question"]
                documents = state_dict["documents"]
                generation = state_dict["generation"]
                score = hallucination_grader.invoke({"documents": documents, "generation": generation})
                grade = score.binary_score
                if grade == "yes":
                    score = answer_grader.invoke({"question": question,"generation": generation})
                    grade = score.binary_score
                    if grade == "yes":
                        return "useful"
                    else:
                        return "not useful"
                else:
                    return "not supported"
            except Exception:
                return "not supported"

        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": END,
                "useful": END,
                "not useful": END,
            },
        )
        
        app = workflow.compile()
        inputs = {
            "keys": {
                "question": data["user_query"],
            }
        }
        
        for output in app.stream(inputs):
            for key, value in output.items():
                LOGGER.info(f"Node '{key}':")
            LOGGER.info("\n---\n")
        
        generation = value["keys"]["generation"]
        return {'message': 'Query processed successfully', 'status': 200, 'question': data["user_query"], 'answer': generation.get('answer')}
