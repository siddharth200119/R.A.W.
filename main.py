from RAW.utils import Logger
from pathlib import Path
from RAW.agentic.llms import Ollama
from RAW.models import Image, Message
import uuid
import asyncio
from pathlib import Path

async def main():
    logger = Logger(log_file=Path("app.log"))
    logger.info(content="app started")
    # image = Image.from_path(Path('/home/siddharth/Downloads/556-200x300.jpg'))
    llm = Ollama(model='gemma3:4b', logger=logger, capabilities=['vision'])
    # async for chunk in llm.generate(prompt="whats in the image", images=[image], stream=True):
    #     logger.info(f"{chunk.thought}{chunk.content}")

    async for chunk in llm.chat(messages=[
        Message(
            content='what is in the image',
            images=[],
            role='user'
        )
    ], tools=[], stream=True):
        print(chunk)

    logger.stop()

if __name__ == '__main__':
    asyncio.run(main=main())