# Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant Session as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Claude as Claude API<br/>(Anthropic)
    participant Tools as Tool Manager<br/>(search_tools.py)
    participant Vector as Vector Store<br/>(vector_store.py)
    participant DB as ChromaDB<br/>(chroma_db/)

    User->>Frontend: Types query & clicks send
    Frontend->>Frontend: Disable input, show loading
    Frontend->>API: POST /api/query<br/>{query, session_id}
    
    API->>API: Validate QueryRequest
    API->>RAG: rag_system.query(query, session_id)
    
    RAG->>Session: get_conversation_history(session_id)
    Session-->>RAG: conversation_history
    
    RAG->>AI: generate_response(query, history, tools, tool_manager)
    
    AI->>Claude: API call with system prompt<br/>& tool definitions
    
    Note over Claude: Claude decides:<br/>General knowledge or<br/>Course-specific query?
    
    alt Course-specific query
        Claude->>Tools: search_course_content(query)
        Tools->>Vector: semantic_search(query)
        Vector->>DB: Query embeddings
        DB-->>Vector: Matching chunks + metadata
        Vector-->>Tools: SearchResults
        Tools-->>Claude: Course content + sources
    else General knowledge
        Note over Claude: Answer from training data
    end
    
    Claude-->>AI: Generated response
    AI-->>RAG: response
    
    RAG->>Tools: get_last_sources()
    Tools-->>RAG: sources[]
    
    RAG->>Session: add_exchange(session_id, query, response)
    RAG-->>API: (response, sources)
    
    API->>API: Create QueryResponse
    API-->>Frontend: {answer, sources, session_id}
    
    Frontend->>Frontend: Remove loading message
    Frontend->>Frontend: addMessage(answer, sources)
    Frontend->>Frontend: Convert markdown to HTML
    Frontend->>Frontend: Re-enable input
    Frontend-->>User: Display response with sources
```

## Component Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[HTML/CSS/JS Interface]
        Script[script.js - Event Handlers]
    end
    
    subgraph "API Layer"
        FastAPI[FastAPI App<br/>app.py]
        Routes["/api/query<br/>/api/courses"]
    end
    
    subgraph "RAG System Core"
        RAGSys[RAG System<br/>rag_system.py]
        Session[Session Manager<br/>session_manager.py]
    end
    
    subgraph "AI Processing"
        AIGen[AI Generator<br/>ai_generator.py]
        Claude[Claude API<br/>Anthropic]
    end
    
    subgraph "Search & Tools"
        Tools[Tool Manager<br/>search_tools.py]
        SearchTool[Course Search Tool]
    end
    
    subgraph "Data Layer"
        Vector[Vector Store<br/>vector_store.py]
        ChromaDB[(ChromaDB<br/>Embeddings)]
        Docs[Course Documents<br/>docs/ folder]
    end
    
    subgraph "Processing"
        DocProc[Document Processor<br/>document_processor.py]
        Models[Data Models<br/>models.py]
    end

    UI --> Script
    Script -->|POST /api/query| FastAPI
    FastAPI --> RAGSys
    RAGSys --> Session
    RAGSys --> AIGen
    AIGen --> Claude
    Claude -->|Tool calls| Tools
    Tools --> SearchTool
    SearchTool --> Vector
    Vector --> ChromaDB
    
    Docs --> DocProc
    DocProc --> Models
    Models --> Vector
    
    Style UI fill:#e1f5fe
    Style FastAPI fill:#f3e5f5
    Style RAGSys fill:#e8f5e8
    Style Claude fill:#fff3e0
    Style ChromaDB fill:#fce4ec
```

## Data Flow Summary

1. **User Input** → Frontend captures and validates
2. **API Request** → FastAPI receives and routes
3. **RAG Processing** → Orchestrates components
4. **AI Decision** → Claude determines search necessity
5. **Vector Search** → ChromaDB semantic lookup (if needed)
6. **Response Generation** → Claude synthesizes answer
7. **Session Update** → Conversation history stored
8. **Frontend Display** → Markdown rendered with sources