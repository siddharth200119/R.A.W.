import asyncio
from RAW import OllamaLLM, OllamaOptions
from RAW.models import Message

async def test_generate_no_stream(llm):
    """Test generate method without streaming"""
    print("\n" + "="*50)
    print("GENERATE - NO STREAMING")
    print("="*50)
    
    try:
        result = await llm.generate(
            prompt="Write a short story about a robot learning to paint.",
            think=True,
            format="json"
        )
        
        print(f"Generated Content: {result.content}")
        if result.thought:
            print(f"Thought Process: {result.thought}")
            
    except Exception as e:
        print(f"Error in generate (no stream): {e}")

async def test_generate_with_stream(llm):
    """Test generate method with streaming"""
    print("\n" + "="*50)
    print("GENERATE - WITH STREAMING")
    print("="*50)
    
    full_content = ""
    full_thought = ""
    buffer = ""  # Buffer to accumulate partial words
    
    try:
        print("Generated Response: ", end="", flush=True)
        
        async for chunk in llm.generate(
            prompt="Explain quantum computing in simple terms.",
            think=True,
            stream=True
        ):
            if chunk.content:
                buffer += chunk.content
                full_content += chunk.content
                
                # Split buffer into words and keep the last incomplete word in buffer
                words = buffer.split(' ')
                if len(words) > 1:
                    # Print all complete words except the last one
                    for word in words[:-1]:
                        if word.strip():  # Only print non-empty words
                            print(word + " ", end="", flush=True)
                            await asyncio.sleep(0.1)  # Small delay for typewriter effect
                    
                    # Keep the last word in buffer (might be incomplete)
                    buffer = words[-1]
            
            if chunk.thought:
                full_thought += chunk.thought
        
        # Print any remaining content in buffer
        if buffer.strip():
            print(buffer, end="", flush=True)
        
        print(f"\n\nFull Generated Content: {full_content}")
        if full_thought:
            print(f"Full Thought Process: {full_thought}")
            
    except Exception as e:
        print(f"Error in generate (stream): {e}")

async def test_chat_no_stream(llm):
    """Test chat method without streaming"""
    print("\n" + "="*50)
    print("CHAT - NO STREAMING")
    print("="*50)
    
    try:
        # First message
        response1 = await llm.chat(
            message=Message(
                role="user",
                content="Hello! Can you help me understand machine learning?"
            ),
            think=True
        )
        
        print(f"Assistant Response 1: {response1.content.content}")
        if response1.content.thought:
            print(f"Assistant Thought 1: {response1.content.thought}")
        
        # Follow-up message (conversation continues)
        response2 = await llm.chat(
            message=Message(
                role="user", 
                content="What are the main types of machine learning?"
            ),
            think=True
        )
        
        print(f"Assistant Response 2: {response2.content.content}")
        if response2.content.thought:
            print(f"Assistant Thought 2: {response2.content.thought}")
            
    except Exception as e:
        print(f"Error in chat (no stream): {e}")

async def test_chat_with_stream(llm):
    """Test chat method with streaming"""
    print("\n" + "="*50)
    print("CHAT - WITH STREAMING")
    print("="*50)
    
    full_content = ""
    full_thought = ""
    buffer = ""  # Buffer to accumulate partial words
    
    try:
        print("User: What are the benefits of renewable energy?")
        print("Assistant: ", end="", flush=True)
        
        async for message_chunk in llm.chat(
            message=Message(
                role="user",
                content="What are the benefits of renewable energy?"
            ),
            think=True,
            stream=True
        ):
            if message_chunk.content.content:
                buffer += message_chunk.content.content
                full_content += message_chunk.content.content
                
                # Split buffer into words and keep the last incomplete word in buffer
                words = buffer.split(' ')
                if len(words) > 1:
                    # Print all complete words except the last one
                    for word in words[:-1]:
                        if word.strip():  # Only print non-empty words
                            print(word + " ", end="", flush=True)
                            await asyncio.sleep(0.1)  # Small delay for typewriter effect
                    
                    # Keep the last word in buffer (might be incomplete)
                    buffer = words[-1]
            
            if message_chunk.content.thought:
                full_thought += message_chunk.content.thought
        
        # Print any remaining content in buffer
        if buffer.strip():
            print(buffer, end="", flush=True)
        
        print(f"\n\nFull Chat Response: {full_content}")
        if full_thought:
            print(f"Full Assistant Thought: {full_thought}")
            
    except Exception as e:
        print(f"Error in chat (stream): {e}")

async def test_advanced_features(llm):
    """Test advanced features like different formats and options"""
    print("\n" + "="*50)
    print("ADVANCED FEATURES")
    print("="*50)
    
    try:
        # Test with JSON format
        print("\n--- JSON Format Generate ---")
        json_result = await llm.generate(
            prompt="Create a JSON object describing a fictional character with name, age, occupation, and hobbies.",
            format="json",
            think=False
        )
        print(f"JSON Result: {json_result.content}")
        
        # Test with custom options
        print("\n--- Custom Options ---")
        custom_llm = OllamaLLM(
            model="qwen3:0.6b",
            options=OllamaOptions(
                temperature=0.8,
                top_p=0.9,
                num_predict=100
            )
        )
        
        custom_result = await custom_llm.generate(
            prompt="Tell me a creative joke.",
            think=False
        )
        print(f"Custom Options Result: {custom_result.content}")
        
        # Test conversation with multiple exchanges
        print("\n--- Multi-turn Conversation ---")
        conv_llm = OllamaLLM(model="qwen3:0.6b")
        
        # Message 1
        resp1 = await conv_llm.chat(
            message=Message(role="user", content="I'm learning Python programming."),
            think=False
        )
        print(f"Turn 1 - Assistant: {resp1.content.content}")
        
        # Message 2
        resp2 = await conv_llm.chat(
            message=Message(role="user", content="Can you give me a simple code example?"),
            think=False
        )
        print(f"Turn 2 - Assistant: {resp2.content.content}")
        
        # Message 3
        resp3 = await conv_llm.chat(
            message=Message(role="user", content="Explain what that code does."),
            think=False
        )
        print(f"Turn 3 - Assistant: {resp3.content.content}")
        
    except Exception as e:
        print(f"Error in advanced features: {e}")

async def main():
    """Main function demonstrating all OllamaLLM capabilities"""
    print("Starting OllamaLLM Comprehensive Test")
    print("Model: qwen3:0.6b")
    
    try:
        # Initialize the LLM
        llm = OllamaLLM(
            model="qwen3:0.6b",
            options=OllamaOptions(
                temperature=0.7,
                num_predict=200
            )
        )
        
        # Test all methods
        await test_generate_no_stream(llm)
        await test_generate_with_stream(llm)
        await test_chat_no_stream(llm)
        await test_chat_with_stream(llm)
        await test_advanced_features(llm)
        
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())