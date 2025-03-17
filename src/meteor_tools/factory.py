# src/meteor_tools/factory.py
"""
Factory functions for creating LangGraph tools that interact with Meteor methods.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Union, Optional
from typing_extensions import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from MeteorClient import MeteorClient

from react_agent.configuration import Configuration


def create_meteor_tool(
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
        schema_note = " (See schema for parameter details)" if json_schema else ""
        description = f"Call the Meteor method '{method_name}' with the provided query.{schema_note}"
        
        if json_schema:
            # Add schema details to the description
            schema_desc = json_schema.get("description", "")
            if schema_desc:
                description += f"\n\n{schema_desc}"
            
            # Add properties information
            props = json_schema.get("properties", {})
            if props:
                description += "\n\nParameters:"
                for prop_name, prop_details in props.items():
                    prop_desc = prop_details.get("description", "No description")
                    description += f"\n- {prop_name}: {prop_desc}"
    
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
            
            # Create a client and connect
            client = MeteorClient(meteor_url)
            
            # Create a future that will be resolved by the callback
            future = asyncio.Future()
            
            def callback(error, result):
                if error:
                    future.set_exception(Exception(str(error)))
                else:
                    future.set_result(result)
            
            # Connect to the server
            client.connect()
            
            # Login if credentials are provided
            if username and password:
                password_bytes = password.encode('utf-8') if isinstance(password, str) else password
                client.login(username, password_bytes)
            
            # Make the call with the callback
            client.call(method_name, [params], callback)
            
            # Wait for the callback to resolve the future
            try:
                result = await future
                return result
            finally:
                # Disconnect when done
                client.disconnect()
                
        except Exception as e:
            return f"Error calling Meteor method '{method_name}': {str(e)}"
    
    # Set metadata for the tool function
    meteor_tool.__name__ = f"meteor_{method_name.lower()}"
    meteor_tool.__doc__ = description
    
    return meteor_tool


def create_meteor_tools(
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
        
        tool = create_meteor_tool(method_name, json_schema, description)
        tools.append(tool)
    
    return tools