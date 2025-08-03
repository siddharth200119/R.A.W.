from RAW.agentic.chatbot import ChatBot
from RAW.agentic.llms import Ollama
from RAW.agentic.memory import ShortTermMemory
from RAW.models import Message
from RAW.utils import Logger

import asyncio
import json

logger = Logger()

memory = ShortTermMemory(logger=logger, actionable_unit='length', actionable_quantity=5)

chatbot = ChatBot(
    llm=Ollama(
        capabilities=['tools', 'vision'],
        logger=logger,
        timeout=5000,
        model='MrScarySpaceCat/gemma3-tools:4b'
    ),
    system_prompt='You are a friendly neighbourhood assistant',
    logger=logger,
    short_term_memory=memory
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