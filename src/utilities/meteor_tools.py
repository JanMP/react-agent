# src/react_agent/meteor_tools.py
"""
Meteor tools for LangGraph.

This module provides factory functions for creating LangGraph tools
that interact with Meteor methods.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Union, Optional
from typing_extensions import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from MeteorClient import MeteorClient

from react_agent.configuration import Configuration


class MeteorClientConnection:
    """
    A singleton manager for maintaining a persistent Meteor client connection.
    """
    _instance = None
    _client = None
    _endpoint = None  # Add this to store the current endpoint URL
    _is_connected = False
    _is_logged_in = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MeteorClientManager, cls).__new__(cls)
        return cls._instance
    
    def get_client(self, meteor_url: str, username: Optional[str] = None, password: Optional[str] = None) -> MeteorClient:
        """
        Get the shared MeteorClient instance, connecting and logging in if necessary.
        """
        # Create client if it doesn't exist or URL has changed
        if self._client is None or self._endpoint != meteor_url:
            self._client = MeteorClient(meteor_url)
            self._endpoint = meteor_url  # Store the endpoint URL
            self._is_connected = False
            self._is_logged_in = False
        
        # Rest of the method stays the same
        if not self._is_connected:
            self._client.connect()
            self._is_connected = True
        
        if username and password and not self._is_logged_in:
            password_bytes = password.encode('utf-8') if isinstance(password, str) else password
            self._client.login(username, password_bytes)
            self._is_logged_in = True
        
        return self._client
    
    async def call_method(self, method_name: str, params: Any, 
                         meteor_url: str, 
                         username: Optional[str] = None, 
                         password: Optional[str] = None) -> Any:
        """
        Call a Meteor method using the shared client.
        """
        client = self.get_client(meteor_url, username, password)
        
        # Create a future that will be resolved by the callback
        future = asyncio.Future()
        
        def callback(error, result):
            if error:
                future.set_exception(Exception(str(error)))
            else:
                future.set_result(result)
        
        # Make the call with the callback
        client.call(method_name, [params], callback)
        
        # Wait for the callback to resolve the future
        return await future


def create_tool(
    method_name: str,
    json_schema: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
) -> Callable:
    """
    Factory function that creates a tool function for calling a Meteor method.
    
    Args:
        method_name: The name of the Meteor method to call
        json_schema: Optional JSON schema describing the expected input format
        description: Optional description of what the tool does
        
    Returns:
        A callable function that can be used as a LangGraph tool
    """
    # Default description if none provided
    if description is None:
        description = f"Call the Meteor method '{method_name}' with the provided query."
        
        if json_schema:
            # Add schema description if available
            schema_desc = json_schema.get("description", "")
            if schema_desc:
                description += f"\n\n{schema_desc}"
    
    # Include the full JSON schema in the tool description
    if json_schema:
        description += f"\n\nJSON Schema: {json.dumps(json_schema, indent=4)}"
        description += "\n\nWhen calling this tool, ensure your input is a valid JSON object that conforms to this schema."
    
    async def meteor_tool(
        query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
    ) -> Any:
        """Call a Meteor method with the given query."""
        try:
            # Get configuration
            configuration = Configuration.from_runnable_config(config)
            meteor_url = configuration.meteor_url
            username = configuration.meteor_user_name
            password = configuration.meteor_user_password
            
            # Parse as JSON if JSON schema is provided
            if json_schema:
                try:
                    params = json.loads(query)
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON in query: {query}"
            else:
                params = {"query": query}
            
            # Use the shared client manager to make the call
            client_manager = MeteorClientManager()
            result = await client_manager.call_method(
                method_name, 
                params, 
                meteor_url,
                username,
                password
            )
            
            return result
                
        except Exception as e:
            return f"Error calling Meteor method '{method_name}': {str(e)}"
    
    # Set metadata for the tool function
    meteor_tool.__name__ = f"meteor_{method_name.lower()}"
    meteor_tool.__doc__ = description
    
    return meteor_tool


def create_tools(
    method_specs: List[Dict[str, Any]]
) -> List[Callable]:
    """
    Create multiple Meteor tool functions based on specifications.
    
    Args:
        method_specs: List of dictionaries with:
            - 'method_name': Name of the Meteor method
            - 'json_schema' (optional): JSON schema for the method parameters
            - 'description' (optional): Description of the tool
    
    Returns:
        List of tool functions
    """
    tools = []
    for spec in method_specs:
        method_name = spec["method_name"]
        json_schema = spec.get("json_schema")
        description = spec.get("description")
        
        tool = create_tool(method_name, json_schema, description)
        tools.append(tool)
    
    return tools


# Create example tools
if __name__ == "__main__":
    METEOR_TOOLS = create_tools([
        {
            "method_name": "TestCall",
            "json_schema": {
                "type": "object",
                "properties": {
                    "testString": {
                        "type": "string",
                        "description": "A test string to send to the server"
                    },
                    "testNumber": {
                        "type": "number",
                        "description": "A test number to send to the server"
                    }
                },
                "required": ["testString", "testNumber"],
                "description": "A test method to send a string and a number to the server and receive the same parameters as a response"
            }
        }
    ])