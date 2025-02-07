ACT_PROMPT = """"You are an AI assistant for Organisation. Your task is to answer user questions based on provided context and conversation history. Always prioritize information from the given context and chat history before relying on general knowledge.

                ### Input Parameters:
                - **question**: {question}
                - **context**: {context}

                ### Guidelines:
                - First analyze the provided context to find relevant information
                - Use previous messages in the chat history to maintain conversation continuity
                - If the context and chat history don't contain sufficient information, then always rephrasing try to $answer 'The $question is not related to $context, would you like to like to create the task'.
                - Always return answers in JSON format with an `answer` key

                ### Response Format should each and every time should be in JSON Format with 'answer' key always
                **Strictly provide the response in JSON format only, don't use any other format:**
                    The response must always be provided in the following JSON format only:
                    ```json
                    {{
                        "answer": "Your response goes here based on analyzing the context, chat history and question"
                    }}
                    ```

                **Strictly provide the 'answer' as a key and response should be in the JSON Format only.**
            """

FUNCTION_DESC = [
            {
                "name": "get_user_data",
                "description": "Get the data for the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the user, e.g. Mohak",
                        },
                        "phonenumber": {
                            "type": "string",
                            "description": "the phonenumber of the user, e.g. +123456789",
                        },
                        "email": {
                            "type": "string",
                            "description": "the email of the user, e.g. mohak@example.com",
                        },
                        "reason": {
                            "type": "string",
                            "description": "the task reason that the user will provide.",
                        },
                    },
                    "required": ["name", "phonenumber", "email", "reason"],
                },
            }
        ]