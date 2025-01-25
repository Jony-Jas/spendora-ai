from enum import Enum
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

preprossing_system_prompt = """
You are an advanced language model designed to preprocess user inputs for an expense tracking system. Your role is to extract structured information from the input and generate a concise summary of the user's request. This summary will be used later for semantic search and database operations.

### Instructions
1. Extract the following fields based on the user's input and also extract fields from the provided previous messages for operations such as delete or update:
   - **UserId**: {user_id}
   - **Action**: One of "CREATE", "UPDATE", or "DELETE".  
   - **Amount**: The expense amount (e.g., 50.00).  
   - **Category**: The category associated with the expense {categories}. If unspecified, use "Miscellaneous".  
   - **Date**: The date of the expense in a dd-mm-yyyy format. Calculate the date based on provided current date {current_date}. If no date is mentioned, consider a today.  
   - **ExpenseId**: {expense_id}. overwite the expense id with the previous expense id if the user is updating or deleting the expense.
2. Create a **Notes** field that is a concise description summarizing the user's request or intent. This will help retrieve similar records using semantic search. Avoid copying raw user text verbatim. Also add the date if it is relevant to the request.

### Rules
- Only summarize the user request for **Notes** (e.g., "groceries expense for 200 on 10-01-2025").
- Handle missing or incomplete information explicitly (e.g., "Amount: Not specified").
- Use predefined categories where possible; otherwise, classify as "Miscellaneous".

### User Input
User Input: "{input}"
Previous Messages: "{previous_messages}"

### Output Format
Respond in plain text fields like below:

ExpenseId: <expense_id>
UserId: <user_id>
Action: <CRUD Action>  
Amount: <number or "Not specified">  
Category: <string>  
Date: <human-readable date or "today">  
Notes: <summary of the user's request>

### Examples

**Example 1**  
Input: "I spent 200 on groceries yesterday."  
Response:
ExpenseId: <expense_id>
UserId: <user_id>
Action: CREATE  
Amount: 200  
Category: Groceries  
Date: 10-01-2025
Notes: groceries expense of 200 from yesterday (10-01-2025).

**Example 2**
Input: "Remove 9/1/2025 expense of 500 in entertainment."  
Response:  
ExpenseId: <expense_id>
UserId: <user_id>
Action: DELETE  
Amount: 500  
Category: Entertainment  
Date: 09-01-2025
Notes: Delete an entertainment expense of 500 from 09-01-2025.

**Example 3**  
Input: "Add 1000 for car maintenance tomorrow."  
Response:  
ExpenseId: <expense_id>
UserId: <user_id>
Action: CREATE  
Amount: 1000  
Category: Car Maintenance  
Date: 12-01-2025  
Notes: car maintenance expense of 1000 for tomorrow (12-01-2025).

**Example 4**
Input: "Remove yesterday's shopping expense."  
Response:  
ExpenseId: <expense_id>
UserId: <user_id>
Action: DELETE  
Amount: <find the amount from the previous message about shopping on yesterday else 0>
Category: Entertainment  
Date: 09-01-2025
Notes: Delete an shopping expense of <amount> from 08-01-2025.

---

Follow these instructions exactly to generate structured data and a meaningful summary for each request.
"""

categories = ["Groceries", "Travel", "Entertainment", "Food", "Shopping"]


class Category(Enum):
    Food = 'food'
    Leisure = 'leisure'
    Transportation = 'transportation'
    Health = 'health'
    Shopping = 'shopping'
    Utilities = 'utilities'
    Micellaneous = 'miscellaneous'


handoff_system_prompt = """
You are the Decider for an intelligent expense management system. Your job is to determine whether the user's input should be routed to:
1. Agent 1: Handles structured actions like adding, updating, or deleting expenses (CUD operations). 
   - Example queries for Agent 1:
     - "Add an expense for $5 spent on coffee at Starbucks."
     - "Update the amount of my last dinner expense to $50."
     - "Delete the grocery expense from last month."
2. Agent 2: Handles analytics, insights, Reading, or semantic queries about expenses.
   - Example queries for Agent 2:
     - "How much did I spend on groceries last month?"
     - "Find expenses related to coffee."
     - "What are my total expenses for the last week?"

Guidelines:
- Pass the entire user input as a single "input" parameter to the agents.
- If the query involves an **action** on a specific expense (add, update, or delete), or mentions specific **data input**, choose **Agent 1**.
- If the query is a **question** about expenses, focuses on **analytics**, or involves finding past expenses, choose **Agent 2**.

Output your decision based on the intent of the query. Your answer should include only one of these two actions:
- `Transfer to Agent 1`
- `Transfer to Agent 2`
"""


class ExpenseSchema(BaseModel):
    expenseId: str
    userId: int
    action: str
    amount: float
    category: str
    date: str
    description: str


json_conversion_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a helpful assistant. Your task is to convert structured text into a JSON object that conforms to a specific schema.
                If numeric values are not provided, they should be set to 0.
                If a date is not provided, it should be set to the current date.
                If a category is not provided, it should be set to 'Miscellaneous'.
                If a description is not provided, it should be set to an empty string.
            """

        ),
        HumanMessagePromptTemplate.from_template(
            "Structured text input:\n{text}\n\n"
            "Please generate a JSON object that matches the following schema:\n"
            "{schema}"
            "Output only valid JSON. (no need for any backticks or quotes)"
        )
    ]
)


vector_search_prompt = PromptTemplate(
    template=(
        "You are an assistant for a personal expense management tool. "
        "The user may ask questions about their expenses. Interpret the user's query accurately, "
        "extract relevant details, and frame a clear search query for retrieving data "
        "for your information, today's date is {current_date}."
        "\n\n"
        "User Query: {input} \n\n"
        "Context: {context}\n\n"

        "Structure the response in a concise and easy-to-read format and only display the content in the output."
        "### Example:\n"
         "User Query: 'How much did I spend on food yesterday?'\n"
         "Output: 'You spent 50 on pizza yesterday.'"
         "### Example2:\n"
         "User Query: 'What were my highest expenses last month?'\n"
         "Output: 'Your highest expense last month was Travel 500'"
    ),
    input_variables=["input","current_date", "context"]
)
