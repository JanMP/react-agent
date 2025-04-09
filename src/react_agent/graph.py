"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


async def reasoner(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our research agent with enforced tool usage."""
    configuration = Configuration.from_runnable_config(config)

    # Get the provider from the model name
    # provider = configuration.model.split('/')[0].lower() if '/' in configuration.model else ""

    # Initialize the model with tool binding
    model = load_chat_model(configuration.model).bind_tools(TOOLS, tool_choice="auto")

    # Format the system prompt
    system_message = configuration.reasoner_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the last step case
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="I've reached the maximum number of steps. Here's what I've found so far."
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def final_response(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Generate a comprehensive final answer based on collected information."""
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model without tool binding for the final response
    model = load_chat_model(configuration.model)

    # Use the final response specific prompt
    system_message = configuration.final_response_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Generate the final response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages],
            config
        ),
    )

    return {"messages": [response]}


async def reasoner_talkback(state: State, config: RunnableConfig) -> Dict[str, List]:
    """Add a reminder to use tools."""
    retry_message = HumanMessage(content="""
        This is a message from the system:
        You have not used any tools in your previous message, so we assume you have been talking to yourself to plan and reason.
        You should use a tool in your answer to this message to either perform the required actions or
        use the finish tool to hand over to the Agent that communicates the results
        """)

    # Return the updated messages
    return {"messages": [retry_message]}



def route_after_reasoner(state: State) -> Literal["tools", "reasoner_talkback", "final_response"]:
    """Determine the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
)

    # If this is the last step and there are no tool calls, go straight to final_response
    if state.is_last_step and not last_message.tool_calls:
        return "final_response"

    # If there are no tool calls but not the last step, go to reasoner_talkback
    if not last_message.tool_calls:
        return "reasoner_talkback"

    # Otherwise, use tools
    return "tools"


def route_after_tools(state: State) -> Literal["reasoner", "final_response"]:
    """Determine where to go after executing tools."""

    last_message = state.messages[-1]

    # If the last message is a tool result, and the tool name is "finish" go to 'final_response
    if isinstance(last_message, ToolMessage) and last_message.name == "finish":
        return "final_response"

    # By default, continue the reasoning process
    return "reasoner"


def create_react_agent():
    workflow = StateGraph(State, input=InputState, config_schema=Configuration)

    # Add nodes
    workflow.add_node("reasoner", reasoner)  # Using the existing reasoner function in your code
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("final_response", final_response)
    workflow.add_node("reasoner_talkback", reasoner_talkback)

    # Create edges
    workflow.add_edge("__start__", "reasoner")
    workflow.add_conditional_edges("reasoner", route_after_reasoner)
    workflow.add_conditional_edges("tools", route_after_tools)
    workflow.add_edge("reasoner_talkback", "reasoner")
    workflow.add_edge("final_response", "__end__")

    # Compile the workflow - Only stream the final_response node
    graph = workflow.compile()
    graph.name = "ReAct Agent"

    return graph

# Create the agent graph
graph = create_react_agent()