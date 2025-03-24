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

from react_agent.meteor_tools import create_tools

meteor_tools = create_tools(
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


TOOLS: List[Callable[..., Any]] = [search, *meteor_tools]
