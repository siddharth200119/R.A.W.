#TODO

1. LLMs
    - [ ] Gemini class
    - [ ] Openrouter class
    - [ ] Groq class
    - [x] Embedding support

2. Utils
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

3. Chatbot
    - [x] Chatbot class
    - [ ] ChatBot Memory
        - [ ] Shortterm memory
            - [x] simple length based compression (length of messages is limited)
                - [x] length based limit
                - [x] context based limit
            - [ ] remove Tool messages
                - [ ] just plain remove them
                - [ ] add context in assistant messages to show they were removed but this is the summary of what he has done
            - [ ] add summary to system prompt for the lost messages to keep the context
        - [ ] Longterm memory in DB
            - [ ] DB support for
                - [ ] SQLite
                - [ ] Postgres
                - [ ] MongoDB
            - [ ] Store every message ever made
            - [ ] store user specific summaries to be added to the system prompt
            - [ ] RAG to retrieve most relevant context from the stored messages
    - [ ] basic tools
        - [ ] date time
        - [ ] file handler
        - [ ] web search
    - [ ] MCP client implementation

4. Tests