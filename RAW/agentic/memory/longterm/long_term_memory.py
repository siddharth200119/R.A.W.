from typing import List
from RAW.models import Message, Tool, ToolParam
from RAW.utils import Logger
from RAW.agentic.memory import SQLiteStorage
from RAW.agentic.llms import Ollama
import json

class LongTermMemory:
    def __init__(self, storage: SQLiteStorage, logger: Logger = Logger()):
        self.storage = storage
        self.logger = logger

        # Create Ollama LLM instance
        self.llm = Ollama(
            capabilities=['tools', 'vision'],
            logger=logger,
            timeout=5000,
            model='qwen3:0.6b'
        )

    async def __call__(self, user_id: str, new_message: Message) -> List[Message]:
        print("Long term memory called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.storage.store_message(user_id, new_message)

        all_messages = self.storage.get_all_messages(user_id)

        existing_summary = self.storage.get_summary(user_id) or ""
        # print("existing_summary: ", existing_summary)
        system_message = Message(
            role="system",
            content=f"Long-term memory summary for this user:\n{existing_summary}"
        )

        return [system_message] + all_messages

    def create_update_memory_tool(self,user_id:str) -> Tool:
        async def update_memory(action: str, messages: list = None):
            print("update_memory called: ", action, user_id, messages)
            try:
                if action == "delete":
                    self.storage.delete_summary(user_id)
                    yield f"Summary deleted for user {user_id}"
                    return

                elif action in ["add", "update"]:
                    if not messages:
                        yield "No messages provided for summary creation"
                        return

                    if not messages or len(messages) < len(self.storage.get_all_messages(user_id)):
                        messages = [msg.content for msg in self.storage.get_all_messages(user_id)]
                    print("messages: ", messages)
                    messages_json = json.dumps(messages, indent=2)
                    prompt = f"Create a comprehensive long-term summary of the following:\n\n{messages_json}"

                    response = await self.llm.generate(prompt=prompt)
                    summary_text = response.content.strip()

                    self.storage.store_summary(user_id, summary_text)
                    yield f"Summary {action} successfully for user {user_id}"
                    return

                else:
                    yield "Invalid action. Use 'add', 'update', or 'delete'."
                    return

            except Exception as e:
                yield f"Error occurred: {str(e)}"
                return

        return Tool(
            name="update_memory_tool",
            description="To Add, update, or delete a user's long-term memory summary.",
            parameters=[
                ToolParam(name="action", type="string", description="Action to perform: add, update, or delete", required=True),
                # ToolParam(name="user_id", type="string", description="Unique ID of the user", required=False),
                ToolParam(name="messages", type="array", description="List of messages", required=False),
            ],
            function=update_memory
        )
    

        

