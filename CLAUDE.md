# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Installation and Setup:**
```bash
# Install Python dependencies
uv sync

# Install development dependencies (includes code quality tools)
uv sync --group dev

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

**Running the Application:**
```bash
# Quick start using the shell script
chmod +x run.sh
./run.sh

# Manual start (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Code Quality:**
```bash
# Format code with black and isort
bash scripts/format.sh

# Run linting checks (flake8, isort, black)
bash scripts/lint.sh

# Run all quality checks (tests + linting)
bash scripts/quality.sh

# Individual tools
uv run black backend/ main.py           # Format code
uv run isort backend/ main.py            # Sort imports
uv run flake8 backend/ main.py           # Lint code
```

**API Endpoints:**
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Query API: `POST /api/query`
- Course Stats: `GET /api/courses`

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials using Anthropic's Claude API with intelligent tool-based search.

### Core Architecture Pattern

The system uses a **tool-based RAG approach** where Claude decides whether to search course content or answer from general knowledge:

1. **RAG System** (`rag_system.py`) - Main orchestrator
2. **AI Generator** (`ai_generator.py`) - Claude API integration with tool calling
3. **Tool Manager** (`search_tools.py`) - Manages search tool execution
4. **Vector Store** (`vector_store.py`) - ChromaDB semantic search
5. **Session Manager** (`session_manager.py`) - Conversation history

### Query Flow Architecture

```
User Query → FastAPI → RAG System → AI Generator → Claude API
                                        ↓
                                  Tool Manager → Vector Store → ChromaDB
```

**Key Decision Point:** Claude determines whether to use the search tool based on query type:
- **Course-specific questions** → Triggers `search_course_content` tool
- **General knowledge** → Answers from training data

### Document Processing Pipeline

Course documents in `docs/` folder follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional url]
[lesson content...]

Lesson 1: Next Topic
[lesson content...]
```

**Processing Steps:**
1. **Parse metadata** - Extract course title, link, instructor
2. **Identify lessons** - Find "Lesson X:" markers
3. **Chunk content** - Sentence-based chunking with overlap
4. **Add context** - Prefix chunks with course/lesson context
5. **Store vectors** - ChromaDB embeddings for semantic search

### Component Responsibilities

**Frontend** (`frontend/`):
- Single-page web interface with chat functionality
- Handles session management and source display
- Converts markdown responses to HTML

**Backend Core**:
- `app.py` - FastAPI endpoints and static file serving
- `rag_system.py` - Orchestrates all components
- `config.py` - Configuration via environment variables

**AI & Search**:
- `ai_generator.py` - Claude API with system prompt for tool usage
- `search_tools.py` - Tool definitions and execution logic
- `vector_store.py` - ChromaDB operations and semantic search

**Data Processing**:
- `document_processor.py` - Parses course documents into structured data
- `models.py` - Pydantic models for Course, Lesson, CourseChunk
- `session_manager.py` - In-memory conversation history

### Configuration

**Environment Variables** (`.env`):
- `ANTHROPIC_API_KEY` - Required for Claude API access

**System Settings** (`config.py`):
- `ANTHROPIC_MODEL` - Currently "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL` - "all-MiniLM-L6-v2" for semantic search
- `CHUNK_SIZE` - 800 characters per text chunk
- `CHUNK_OVERLAP` - 100 characters overlap between chunks
- `MAX_RESULTS` - 5 search results maximum
- `MAX_HISTORY` - 2 conversation exchanges remembered

### Data Storage

**ChromaDB** (`backend/chroma_db/`):
- Persistent vector storage for course content
- Two collections: course metadata and course content chunks
- Automatic embedding generation using sentence-transformers

**Course Documents** (`docs/`):
- Text files with structured format
- Processed on startup via `app.py` startup event
- Added to vector store for semantic search

### Key Design Decisions

1. **Tool-based RAG** - Claude decides when to search, reducing unnecessary vector queries
2. **Context-enhanced chunks** - Each chunk includes course/lesson metadata for better retrieval
3. **Session-based conversations** - Maintains context across multiple queries
4. **Sentence-based chunking** - Preserves semantic coherence over fixed-size chunking
5. **Static frontend serving** - FastAPI serves HTML/CSS/JS directly for simplicity

### Adding New Course Content

1. Place structured text files in `docs/` folder
2. Restart the application to trigger document processing
3. Files are automatically parsed and added to vector store
4. Existing courses are skipped to avoid duplicates