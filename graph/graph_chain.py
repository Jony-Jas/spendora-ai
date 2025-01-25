import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage


from utils.constants import handoff_system_prompt
from agents.input_preprocessing import Preprocessor
from agents.json_converter import JsonConverter
from agents.db_saver import DbSaver
from agents.vector_search import VectorSearch
from agents.sql_search import sql_search_agent_executor

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.tools import tool
from typing import Annotated
from langgraph.prebuilt import InjectedState


load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


def semantic_vector_search(state: Annotated[dict, InjectedState]) -> Command[Literal["agent2"]]:
    """
    Perform a semantic search on the received messages and return the result.
    Routes the message to 'agent2' after processing.
    """
    print("Call came to semantic_vector_search ***********************",
          state["messages"][0].content)

    # Perform semantic search
    search_result = VectorSearch().search(state["messages"][0].content)

    # Validate the search result
    if not search_result:
        print("Semantic vector search returned no results.")
        search_result = "No relevant results found."

    return Command(
        update={
            "messages": [
                HumanMessage(content=search_result,
                             name="semantic_vector_search")
            ]
        },
        goto="agent2",
    )


def sql_search(state: Annotated[dict, InjectedState]) -> Command[Literal["agent2"]]:
    """
    Perform a SQL search on the received messages and return the result.
    Routes the message to 'agent2' after processing.
    """
    print("Call came to sql_search ***********************",
          state["messages"][0].content)

    # Perform SQL search
    sql_result = sql_search_agent_executor().invoke({
        "messages": [("user", state["messages"][0].content)]
    })

    # Validate the SQL search result
    if not sql_result or "messages" not in sql_result or not sql_result["messages"]:
        print("SQL search returned no results.")
        sql_result = {"messages": [HumanMessage(
            content="No SQL results found.")]}

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=sql_result["messages"][-1].content, name="sql_search")
            ]
        },
        goto="agent2",
    )


def search_aggregator(state: Annotated[dict, InjectedState]) -> Command[Literal["agent2"]]:
    """
    Aggregate results from previous searches and prepare the final response.
    Routes the aggregated result to 'agent2'.
    """
    print("Call came to search_aggregator ***********************")

    # Fetch current messages and append aggregation result
    messages = state["messages"]
    messages.append(
        AIMessage(content="Aggregated results from semantic and SQL searches.",
                  name="search_aggregator")
    )

    return Command(
        update={"messages": messages},
        goto="agent2",
    )


@tool("process_messages", return_direct=True)
def process_messages(input, state: Annotated[dict, InjectedState]) -> str:
    """Process the received messages and return the last message."""
    print("************* Call came to process_messages **********")

    preprocessor = Preprocessor()
    res = preprocessor.preprocess(input, 27456, state["messages"])

    return res


@tool("json_converter", return_direct=True)
def json_converter(input, state: Annotated[dict, InjectedState]) -> dict:
    """Convert the extracted user intent and data to a structured JSON format."""

    converter = JsonConverter()
    return converter.convert(input)


@tool("db_saver", return_direct=True)
def db_saver(input, state: Annotated[dict, InjectedState]) -> str:
    """Save the structured JSON data to a database."""

    saver = DbSaver()
    return saver.save(input)


class GraphChain:
    def __init__(self):
        self.llm = ChatGroq(model='llama-3.3-70b-specdec')
        self.memory = MemorySaver()
        self.graph = None

    def make_handoff_tool(self, *, agent_name: str):
        """Create a tool that can return handoff via a Command"""
        tool_name = f"transfer_to_{agent_name}"

        @tool(tool_name)
        def handoff_to_agent(
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            """Ask another agent for help."""
            tool_message = {
                "role": "tool",
                "content": f"Successfully transferred to {agent_name}",
                "name": tool_name,
                "tool_call_id": tool_call_id,
            }

            return Command(
                goto=agent_name,
                graph=Command.PARENT,
                update={"messages": state["messages"] + [tool_message]},
            )

        return handoff_to_agent

    def agent2_node(obj, llm: BaseChatModel, members: list[str]) -> str:
        """
        Routes the conversation sequentially to specified workers or terminates with 'FINISH' after all workers are called.
        """
        print(obj, llm, members)

        # Define routing options: 'FINISH' or the list of member workers
        options = ["FINISH"] + members

        # Define the system prompt for the LLM
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. if no results found try with another worker."
            " even if no results found, respond with FINISH."
            " When finished, combine their response and summarize the results. If no more workers then"
            " respond with FINISH."
        )

        # Router structure to interpret LLM responses
        class Router(TypedDict):
            """Structure for routing to the next worker or finishing."""
            next: Literal[*options]  # Possible values include 'FINISH' and the members

        def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
            """
            LLM-based router that determines the next worker to act on the user request or finishes the process.
            """
            # Prepare the conversation context
            messages = [
                {"role": "system", "content": system_prompt},
            ] + state["messages"]

            # Use LLM to process the messages and get structured output
            response = llm.with_structured_output(Router).invoke(messages)

            # Determine the next step
            goto = response["next"]  # Get the worker to route or 'FINISH'
            return Command(goto=END if goto == "FINISH" else goto)

        return supervisor_node

    def create_handoff(self):
        self.agent1 = create_react_agent(
            self.llm,
            [process_messages]
        )

        self.agent2_node__ = self.agent2_node(
            llm=self.llm, members=["semantic_vector_search", "sql_search", "search_aggregator"])

        self.decider = create_react_agent(
            self.llm,
            [self.make_handoff_tool(agent_name="agent1"),
             self.make_handoff_tool(agent_name="agent2")],
            state_modifier=handoff_system_prompt
        )
        self.json_converter = create_react_agent(
            self.llm,
            [json_converter],
        )
        self.db_saver = create_react_agent(
            self.llm,
            [db_saver],
        )

    def build_graph(self):
        self.create_handoff()
        builder = StateGraph(MessagesState)

        builder.add_node("agent1", self.agent1)
        builder.add_node("decider", self.decider)
        builder.add_node("json_converter", self.json_converter)
        builder.add_node("db_saver", self.db_saver)

        builder.add_node("agent2", self.agent2_node__)
        builder.add_node("semantic_vector_search", semantic_vector_search)
        builder.add_node("sql_search", sql_search)
        builder.add_node("search_aggregator", search_aggregator)

        builder.add_edge(START, "decider")
        builder.add_edge("agent1", "json_converter")
        builder.add_edge("json_converter", "db_saver")
        builder.add_edge("db_saver", END)

        self.graph = builder.compile(checkpointer=self.memory)
