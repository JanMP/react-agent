from MeteorClient import MeteorClient
import asyncio
import json

async def main():
    # Create a client and connect
    client = MeteorClient('ws://127.0.0.1:3000/websocket')
    client.connect()

    # Encode the password as bytes before passing it to login
    client.login('LangGraphAgent', 'reasonablySafePassword1723'.encode('utf-8'))

    # Create a future that will be resolved by the callback
    future = asyncio.Future()

    def callback(error, result):
        if error:
            print(f"Error: {error}")
            future.set_exception(Exception(str(error)))
        else:
            print(f"Success: {result}")
            future.set_result(result)

    # Make the call with the callback
    client.call('TestCall', [{'query': 'Hello, world!'}], callback)

    # Wait for the callback to resolve the future
    try:
        result = await future
        return result
    finally:
        client.disconnect()

if __name__ == "__main__":
    result = asyncio.run(main())
print(f"Final result: {json.dumps(result, indent=2)}")