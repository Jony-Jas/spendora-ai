from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import ValidationError
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from utils.constants import ExpenseSchema, json_conversion_prompt_template


class JsonConverter:

    def __init__(self):
        """"""
        self.llm = ChatGroq(model='llama-3.3-70b-versatile')
        self.output_parser = JsonOutputParser(pydantic_object=ExpenseSchema)

    def convert(self, input):
        """
        Converts extracted user intent and data to a structured JSON format.

        Args:
            input_text (str): The processed user input text to convert to JSON.

        Returns:
            dict: A JSON object containing categorized data.
        """

        prompt_template = json_conversion_prompt_template

        messages = prompt_template.format_messages(
            text=input, schema=ExpenseSchema.model_json_schema())

        response = self.llm.invoke(messages)

        try:
            json_result = self.output_parser.parse(response.content)
        except ValidationError as e:
            print("ValidationError:", e)
        except Exception as e:
            print("Other Error:", e)

        return json_result
