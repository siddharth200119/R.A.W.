import asyncio
from RAW import OllamaLLM, OllamaOptions
from RAW.models import Message

async def main():
    llm = OllamaLLM(
        model="qwen3:0.6b"
    )
    async for item in llm.chat(message=Message(
        role="user",
        content="Hello"
    ), think=True, stream=True):
        pass

    # res = await llm.chat(message=Message(
    #     role="user",
    #     content="Hello"
    # ), think=True)

    # print(res.content)

if __name__ == "__main__":
    asyncio.run(main())
