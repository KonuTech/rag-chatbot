import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add parent directory to path for backend module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "test_chroma"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test-key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    MAX_HISTORY = 2


@pytest.fixture
def mock_config():
    """Provide mock configuration for tests"""
    return MockConfig()


@pytest.fixture
def mock_rag_system():
    """Create a fully mocked RAG system"""
    with patch('rag_system.DocumentProcessor'), \
         patch('rag_system.VectorStore'), \
         patch('rag_system.AIGenerator'), \
         patch('rag_system.SessionManager'), \
         patch('rag_system.ToolManager'), \
         patch('rag_system.CourseSearchTool'), \
         patch('rag_system.CourseOutlineTool'):
        
        from rag_system import RAGSystem
        rag_system = RAGSystem(MockConfig())
        
        # Configure default mock behaviors
        rag_system.ai_generator.generate_response.return_value = "Mock AI response"
        rag_system.tool_manager.get_last_sources.return_value = []
        rag_system.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search course content"}
        ]
        rag_system.session_manager.get_conversation_history.return_value = None
        rag_system.session_manager.create_session.return_value = "test_session_123"
        
        return rag_system


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Import response models
    from app import QueryRequest, QueryResponse, SourceItem, CourseStats
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Define API endpoints inline to avoid static file issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            # Process query using RAG system
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            # Convert sources to SourceItem objects
            source_items = []
            for source in sources:
                if isinstance(source, dict):
                    source_items.append(SourceItem(
                        text=source.get('text', str(source)),
                        url=source.get('url')
                    ))
                else:
                    source_items.append(SourceItem(text=str(source)))
            
            return QueryResponse(
                answer=answer,
                sources=source_items,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint for health check"""
        return {"message": "Course Materials RAG System API"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client for API testing"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is Python programming?",
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data"""
    return {
        "answer": "Python is a high-level programming language.",
        "sources": [
            {"text": "Python Fundamentals - Lesson 1", "url": None},
            {"text": "Python Basics Course", "url": "https://example.com/python-course"}
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def sample_course_stats():
    """Sample course statistics data"""
    return {
        "total_courses": 3,
        "course_titles": ["Python Fundamentals", "Web Development", "Data Science"]
    }


@pytest.fixture
def mock_ai_generator_response():
    """Mock AI generator response"""
    return "Python is a versatile programming language used for web development, data analysis, and automation."


@pytest.fixture
def mock_sources():
    """Mock source items"""
    return [
        {"text": "Python Programming Course - Introduction", "url": None},
        {"text": "Python Fundamentals - Variables and Data Types", "url": "https://example.com/lesson1"}
    ]


@pytest.fixture
def mock_course_analytics():
    """Mock course analytics data"""
    return {
        "total_courses": 5,
        "course_titles": [
            "Introduction to Python",
            "Web Development with Django", 
            "Data Science Fundamentals",
            "Machine Learning Basics",
            "API Development"
        ]
    }


@pytest.fixture(autouse=True)
def cleanup_patches():
    """Automatically clean up patches after each test"""
    yield
    # Cleanup happens automatically with context managers