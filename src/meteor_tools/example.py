# src/meteor_tools/example.py
"""
Example usage of the Meteor tools factory.
"""

from meteor_tools.factory import create_meteor_tool, create_meteor_tools


# Create multiple tools using specifications
tools = create_meteor_tools([
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

# Export the tools for use with LangGraph
METEOR_TOOLS = tools