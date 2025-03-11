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

from MeteorClient import MeteorClient
import asyncio

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

async def test_call(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Make a test call to the Meteor Server"""
    try:
        # Get configuration (you may want to add meteor_url to your Configuration class)
        configuration = Configuration.from_runnable_config(config)
        meteor_url = getattr(configuration, 'meteor_url', 'ws://plaiground.coding-pioneers.com:3000/websocket')
        
        # Create a client and connect
        client = MeteorClient(meteor_url)
        
        # Create a future that will be resolved by the callback
        future = asyncio.Future()
        
        def callback(error, result):
            if result:
                future.set_result(f"Success: {result}")
        
        # Connect to the server
        client.connect()
        
        # Make the call with the callback
        client.call('testCall', [query], callback)
        
        # Wait for the callback to resolve the future
        try:
            return await future
        finally:
            # Disconnect when done
            client.disconnect()
            
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS: List[Callable[..., Any]] = [search, test_call]
