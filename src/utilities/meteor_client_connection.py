import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Union, Optional
from typing_extensions import Annotated

# from langchain_core.runnables import RunnableConfig
# from langchain_core.tools import InjectedToolArg
from MeteorClient import MeteorClient
import socket

class MeteorClientConnection:

    def __init__(self, meteor_prefix: str):
        self.url = "ws://127.0.0.1:3000/websocket" # os.environ.get(f"{meteor_prefix}_METEOR_URL", "fnord")
        self.username = "LangGraphAgent" # os.environ.get(f"{meteor_prefix}_METEOR_USERNAME", "snafu")
        self.password = "reasonablySafePassword1723" # os.environ.get(f"{meteor_prefix}_METEOR_PASSWORD", None)
        if self.url is None:
            raise ValueError(f"{meteor_prefix}_METEOR_URL not set")
        if self.username is None:
            raise ValueError(f"{meteor_prefix}_METEOR_USERNAME not set")
        if self.password is None:
            raise ValueError(f"{meteor_prefix}_METEOR_PASSWORD not set")
        self.client = MeteorClient(self.url)
        self.is_connected = False
        self.is_logged_in = False
        print('MeteorClientConnection', self.url, self.username, self.password)

    def ensure_connection(self) -> None:
        """
        Get the shared MeteorClient instance, connecting and logging in if necessary.
        """
        if self.client is None:
            self.client = MeteorClient(self.url)
            self.is_connected = False
            self.is_logged_in = False
        if not self.is_connected:
            try:
                self.client.connect()
                self.is_connected = True
            except socket.error as e:
                raise ConnectionError(f"Failed to connect to Meteor server at {self.url}: {e}")
        if not self.is_logged_in:
            try:
                self.client.login(self.username, self.password.encode('utf-8'))
                self.is_logged_in = True
            except socket.error as e:
                raise ConnectionError(f"Failed to login to Meteor server at {self.url}: {e}")
    
    async def call(self, method_name: str, params: List[Any]) -> Any:
        """
        Async wrapper for calling a Meteor method using the shared client connection.
        """
        self.ensure_connection()
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def callback(error, result):
            if error:
                loop.call_soon_threadsafe(future.set_exception, Exception(error))
            else:
                loop.call_soon_threadsafe(future.set_result, result)

        self.client.call(method_name, params, callback)
        return await future

    def create_tool(
        self,
        method_name: str,
        json_schema: Dict[str, Any],
        instruction: Optional[str] = None,) -> Callable:
        """
        Factory function that creates a tool function for calling a Meteor method.

        Args:
            method_name (str): The name of the Meteor method to call.
            json_schema (Dict[str, Any]): The JSON schema for the method's parameters.
            instruction (Optional[str]): Optional instruction for the tool using llm.

        Returns:
            Callable: A callable function with metadata that can be used as a LangGraph tool.
        """

        doc = f"""
            {json_schema.get("description", "")}
            {instruction or ""}
            Call with a dictionary of parameters that match the JSON schema:
            {json.dumps(json_schema, indent=4)}
         """
        
        async def tool_function(params_json: str) -> Any:
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"
            try:
                result = await self.call(method_name, [params])
            except Exception as e:
                return f"Error calling {method_name}: {e}"
            return result

        tool_function.__name__ = method_name
        tool_function.__doc__ = doc
        return tool_function

    def create_tools(
        self,
        tool_definitions: List[Dict[str, Any]]
    ) -> List[Callable]:
        return [
            self.create_tool(
                method_name=tool["method_name"],
                json_schema=tool.get("json_schema", {}),
                instruction=tool.get("instruction")
            ) for tool in tool_definitions
        ]
