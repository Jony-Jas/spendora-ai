from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from datetime import date
import uuid

from utils.constants import preprossing_system_prompt
from utils.constants import Category


class Preprocessor:

    def __init__(self):
        self.llm = ChatGroq(model='llama-3.3-70b-versatile')

    def create_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    preprossing_system_prompt,
                ),
                ("human", "{input}"),
            ]
        )
        return prompt

    def preprocess(self, input_text, uid, messages=""):
        chain = self.create_prompt() | self.llm
        return chain.invoke(
            {
                "expense_id": uuid.uuid4(),
                "categories": [category.value for category in Category],
                "current_date": date.today().strftime("%d-%m-%Y"),
                "input": input_text,
                "user_id": uid,
                "previous_messages": messages,
            }
        ).content
