from RAW.agentic.chatbot import ChatBot
from RAW.agentic.llms import Ollama
from RAW.agentic.memory import ShortTermMemory
from RAW.agentic.memory import LongTermMemory
from RAW.agentic.memory import SQLiteStorage
from RAW.models import Message
from RAW.utils import Logger

import asyncio
import json

logger = Logger()

llm = Ollama(
        capabilities=['tools', 'vision'],
        logger=logger,
        timeout=5000,
        model='qwen3:0.6b'
    )

short_term_memory = ShortTermMemory(logger=logger, actionable_unit='length', actionable_quantity=3, summarize_to_system_prompt=True, llm=llm)

storage = SQLiteStorage(db_path="longterm_memory.db", logger=logger)


long_term_memory = LongTermMemory(storage=storage, logger=logger)


chatbot = ChatBot(
    llm=llm,
    user_id='4',
system_prompt = """
You are a helpful assistant. When the user requests to add, update, or delete long-term memory, respond by calling the appropriate tool `update_memory_tool` with correct parameters.  
For example:  
User: "Add a summary of our conversation"  
You: call tool "update_memory_tool" with action "add" and messages containing the summary text.  
If the user just asks general questions, answer normally without calling tools.
""",
    logger=logger,
    short_term_memory=short_term_memory,
    long_term_memory=long_term_memory   
)

message = Message(
    role='user',
    content='who are you'
)

async def main():
    print("Welcome to your friendly assistant! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        message = Message(role='user', content=user_input)
        response = await chatbot(message)

        print(f"Bot: {response.content}\n")

    logger.debug(json.dumps(chatbot.messages, indent=2, default=str))
    logger.stop()

if __name__ == '__main__':
    asyncio.run(main())