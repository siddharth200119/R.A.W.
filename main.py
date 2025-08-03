from RAW.agentic.chatbot import ChatBot
from RAW.agentic.llms import Ollama
from RAW.models import Message
from RAW.utils import Logger

import asyncio
import json

logger = Logger()

chatbot = ChatBot(
    llm=Ollama(
        capabilities=['tools', 'vision'],
        logger=logger,
        timeout=5000,
        model='MrScarySpaceCat/gemma3-tools:4b'
    ),
    system_prompt='You are a friendly neighbourhood assistant',
    logger=logger
)

message = Message(
    role='user',
    content='who are you'
)

async def main():
    _message = await chatbot(message)
    logger.debug(json.dumps(chatbot.messages, indent=2, default=str))
    logger.stop()

if __name__ == '__main__':
    asyncio.run(main())