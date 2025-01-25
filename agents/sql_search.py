from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.tools import tool
from typing import Annotated
from langgraph.prebuilt import InjectedState


def sql_search_agent_executor():
    """Search the SQL database for the user's input."""
    llm = ChatGroq(model='llama-3.3-70b-versatile')
    db = SQLDatabase.from_uri(
        'postgresql://postgres:admin@localhost:5432/postgres')

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    system_message = prompt_template.format(dialect="postgresql", top_k=5)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_react_agent(
        llm, toolkit.get_tools(), state_modifier=system_message
    )

    print("************* Call came to sql search **********")

    # res = agent_executor.invoke({"messages": [("user", input)]})[
    #     "messages"][-1].content
    # res = agent_executor.stream({"messages": [("user", input.content)]})
    # return agent_executor
    return agent_executor
