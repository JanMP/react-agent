"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Define the function that calls the model


async def reasoner(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def final_response(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Generate a comprehensive final answer based on collected information.

    This node creates a new response that synthesizes all information gathered during
    the conversation into a clear final answer.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model without tool binding for the final response
    model = load_chat_model(configuration.model)

    # Create system message for final response generation
    system_message = (
        configuration.system_prompt +
        "\n\nBased on all information gathered so far, provide a comprehensive, "
        "clear, and direct final answer to the user's original question. "
        "Don't mention your research process or tool usage in your answer."
    ).format(system_time=datetime.now(tz=timezone.utc).isoformat())

    # Generate the final response using the full conversation history
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages],
            config
        ),
    )

    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(reasoner)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node(final_response)

# Set the entrypoint as `reasoner`
# This means that this node is the first one called
builder.add_edge("__start__", "reasoner")


def route_model_output(state: State) -> Literal["tools", "__end__"]:
    """Determine the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there are no tool calls, default to ending
    if not last_message.tool_calls:
        return "__end__"

    # Always go to tools if there are tool calls
    return "tools"


def route_after_tools(state: State) -> Literal["reasoner", "final_response"]:
    """Determine where to go after executing tools."""
    # Option 1: Check if we've reached a maximum number of tool calls
    tool_calls_count = sum(
        1 for msg in state.messages
        if isinstance(msg, AIMessage) and msg.tool_calls
    )
    if tool_calls_count >= 3:  # Set a reasonable limit
        return "final_response"

    # Option 2: Check if the most recent tool result contains conclusive information
    # For example, you might look for specific patterns in the results
    last_message = state.messages[-1]
    if isinstance(last_message, HumanMessage):
        content_str = str(last_message.content)
        if "founder" in content_str.lower() and "Harrison Chase" in content_str:
            return "final_response"

    # By default, continue the reasoning process
    return "reasoner"


# Add a conditional edge to determine the next step after `reasoner`
builder.add_conditional_edges(
    "reasoner",
    # After reasoner finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

def create_react_agent():
    workflow = StateGraph(State, input=InputState, config_schema=Configuration)

    # Add nodes
    workflow.add_node("reasoner", reasoner)  # Using the existing reasoner function in your code
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("final_response", final_response)

    # Create edges
    workflow.add_edge("__start__", "reasoner")
    workflow.add_conditional_edges("reasoner", route_model_output)
    workflow.add_conditional_edges("tools", route_after_tools)
    workflow.add_edge("final_response", "__end__")

    # Compile the workflow - Only stream the final_response node
    graph = workflow.compile(
        interrupt_before=[],
        interrupt_after=["final_response"],  # Only stream the final response
    )
    graph.name = "ReAct Agent"

    return graph

# Create the agent graph
graph = create_react_agent()