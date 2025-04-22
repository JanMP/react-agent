"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration

from utilities.meteor_client_connection import MeteorClientConnection
# from utilities.greeting import greeting
# greeting("Jane Doe")
from react_agent.meteor_tools import create_tools

tool_client = MeteorClientConnection(Configuration.meteor_prefix)

meteor_tools = tool_client.create_tools(
    [
        {
            "method_name": "TestCall",
            "json_schema": {
                "type": "object",
                "properties": {
                    "testString": {
                        "type": "string",
                        "description": "A test string to send to the server",
                    },
                    "testNumber": {
                        "type": "number",
                        "description": "A test number to send to the server",
                    },
                },
                "required": ["testString", "testNumber"],
                "description": "A test method to send a string and a number to the server and receive the same parameters as a response",
            },
        }
    ]
)


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def finish(
    *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Use this tool when you've gathered all necessary information and are ready to provide your final answer.

    This tool doesn't require any parameters - just call it when you're ready to conclude.
    """
    return "Switching to final answer generation mode."
TOOLS: List[Callable[..., Any]] = [search, finish, *meteor_tools]

import asyncio

if __name__ == "__main__":
    print(Configuration.meteor_prefix)
    print(tool_client.url)
    print(tool_client.username)
    async def main():
        for tool in meteor_tools:
            print(f"Tool name: {tool.__name__}")
            print(f"Tool docstring: {tool.__doc__}")
            print("-" * 50)
            result = await tool('{"testString": "Hello", "testNumber": 42}')
            print(f"Result: {result}")

    asyncio.run(main())