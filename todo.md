#TODO

1. LLMs
    - [ ] Gemini class
    - [ ] Openrouter class
    - [ ] Groq class
    - [ ] Embedding support

2. Utils
    - [ ] pub sub state manager instead of globals
    - [ ] file viewers / loaders
    - [ ] RAG
        - [ ] Normal RAG
            - [ ] Chunking
                - [ ] Fixed size chinking
                - [ ] Overlap chinking (Fixed size with overlap might combine the 2 with a parameter)
                - [ ] Sentence based chunking
                - [ ] Paragraph based chunking
                - [ ] Semantic Chunking
                - [ ] Markdown
                - [ ] Text Splitter
            - [ ] Retrieval
                - [ ] Dense Retrieval (cosine similarity)
                - [ ] Keyword-Based 
                - [ ] Hybrid Retrieval
                - [ ] Multi-Vector Retrieval
                - [ ] Hierarchical Retrieval
                - [ ] Metadata based
            - [ ] Vector Stores
                - [ ] chroma DB
                - [ ] Qdrant
        - [ ] Graph RAG

3. Agentic
    - [ ] Chatbot class
    - [ ] basic tools
        - [ ] date time
        - [ ] file handler
        - [ ] web search
    - [ ] MCP client implementation
    - [ ] N8N style agents support
    - [ ] Sandboxing

4. Tests