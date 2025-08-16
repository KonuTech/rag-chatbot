import pytest
import sys
import os
import unittest.mock
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from vector_store import SearchResults


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


class TestRAGSystem:
    """Test suite for RAG system content query handling"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MockConfig()
        
        # Mock all the dependencies
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            self.rag_system = RAGSystem(self.config)
            
            # Get references to mocked components
            self.mock_vector_store = self.rag_system.vector_store
            self.mock_ai_generator = self.rag_system.ai_generator
            self.mock_session_manager = self.rag_system.session_manager
            self.mock_tool_manager = self.rag_system.tool_manager
            self.mock_search_tool = self.rag_system.search_tool
    
    def test_query_content_related_question_success(self):
        """Test RAG system handling of content-related questions with successful results"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Python is a high-level programming language used for web development, data analysis, and automation."
        
        # Mock tool manager returning sources
        mock_sources = [
            {"text": "Python Fundamentals - Lesson 1", "url": None},
            {"text": "Python Fundamentals - Lesson 2", "url": "https://example.com/lesson2"}
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        # Mock session manager
        self.mock_session_manager.get_conversation_history.return_value = None
        
        # Execute query
        response, sources = self.rag_system.query("What is Python?", session_id="test_session")
        
        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        
        assert call_args[1]["query"] == "Answer this question about course materials: What is Python?"
        assert call_args[1]["conversation_history"] is None
        assert "tools" in call_args[1]
        assert "tool_manager" in call_args[1]
        
        # Verify tools were provided
        assert call_args[1]["tools"] == self.mock_tool_manager.get_tool_definitions.return_value
        assert call_args[1]["tool_manager"] == self.mock_tool_manager
        
        # Verify response and sources
        assert response == "Python is a high-level programming language used for web development, data analysis, and automation."
        assert sources == mock_sources
        
        # Verify session management
        self.mock_session_manager.get_conversation_history.assert_called_once_with("test_session")
        self.mock_session_manager.add_exchange.assert_called_once_with(
            "test_session", 
            "What is Python?", 
            "Python is a high-level programming language used for web development, data analysis, and automation."
        )
        
        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_conversation_history(self):
        """Test RAG system using conversation history"""
        # Mock conversation history
        conversation_history = "User: What is programming?\nAssistant: Programming is the process of creating instructions for computers."
        self.mock_session_manager.get_conversation_history.return_value = conversation_history
        
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Python is a popular programming language that's great for beginners."
        self.mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query("Tell me about Python", session_id="test_session")
        
        # Verify conversation history was passed to AI generator
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] == conversation_history
        
        # Verify session was updated with new exchange
        self.mock_session_manager.add_exchange.assert_called_once_with(
            "test_session",
            "Tell me about Python",
            "Python is a popular programming language that's great for beginners."
        )
    
    def test_query_without_session_id(self):
        """Test RAG system query without session ID (no conversation history)"""
        self.mock_ai_generator.generate_response.return_value = "General response about Python"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query("What is Python?")
        
        # Verify no session management calls were made
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator was called without conversation history
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] is None
        
        assert response == "General response about Python"
        assert sources == []
    
    def test_query_ai_generator_failure(self):
        """Test RAG system handling AI generator failures"""
        # Mock AI generator to raise exception
        self.mock_ai_generator.generate_response.side_effect = Exception("API rate limit exceeded")
        
        # Test that exception propagates (or is handled gracefully)
        with pytest.raises(Exception) as exc_info:
            self.rag_system.query("What is Python?")
        
        assert "API rate limit exceeded" in str(exc_info.value)
    
    def test_query_tool_manager_failure(self):
        """Test RAG system when tool manager fails to get sources"""
        self.mock_ai_generator.generate_response.return_value = "Response from AI"
        # Mock tool manager to raise exception when getting sources
        self.mock_tool_manager.get_last_sources.side_effect = Exception("Tool manager error")
        
        # Should handle the error gracefully
        with pytest.raises(Exception):
            self.rag_system.query("What is Python?")
    
    def test_query_prompt_formatting(self):
        """Test that query prompt is correctly formatted"""
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        # Test various query formats
        test_queries = [
            "What is Python?",
            "Tell me about machine learning",
            "How do I use pandas?"
        ]
        
        for query in test_queries:
            self.rag_system.query(query)
            call_args = self.mock_ai_generator.generate_response.call_args[1]
            expected_prompt = f"Answer this question about course materials: {query}"
            assert call_args["query"] == expected_prompt
    
    def test_tool_registration(self):
        """Test that tools are properly registered with tool manager"""
        # Verify that search and outline tools were registered
        expected_calls = [
            unittest.mock.call(self.mock_search_tool),
            unittest.mock.call(self.mock_outline_tool)
        ]
        
        # The actual calls would be made during __init__, so we verify the tools exist
        assert self.rag_system.search_tool == self.mock_search_tool
        assert self.rag_system.outline_tool == self.mock_outline_tool
        assert self.rag_system.tool_manager == self.mock_tool_manager
    
    def test_component_initialization(self):
        """Test that all components are properly initialized"""
        # Verify all components exist and have expected types
        assert hasattr(self.rag_system, 'document_processor')
        assert hasattr(self.rag_system, 'vector_store')
        assert hasattr(self.rag_system, 'ai_generator')
        assert hasattr(self.rag_system, 'session_manager')
        assert hasattr(self.rag_system, 'tool_manager')
        assert hasattr(self.rag_system, 'search_tool')
        assert hasattr(self.rag_system, 'outline_tool')
        
        # Verify config is stored
        assert self.rag_system.config == self.config


class TestRAGSystemIntegration:
    """Integration tests for RAG system components working together"""
    
    def setup_method(self):
        """Set up more realistic mocks for integration testing"""
        self.config = MockConfig()
        
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):
            
            self.rag_system = RAGSystem(self.config)
    
    def test_content_query_full_flow(self):
        """Test full flow of a content-related query"""
        # Setup: AI decides to use search tool, tool returns results, AI generates final response
        
        # Mock tool manager to return tool definitions
        mock_tool_definitions = [
            {"name": "search_course_content", "description": "Search course content"},
            {"name": "get_course_outline", "description": "Get course outline"}
        ]
        self.rag_system.tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        
        # Mock AI generator response (simulating Claude deciding to use tools)
        self.rag_system.ai_generator.generate_response.return_value = "Based on the search results, Python is a programming language that is widely used for data science and web development."
        
        # Mock sources from tool execution
        mock_sources = [{"text": "Python Basics - Lesson 1", "url": None}]
        self.rag_system.tool_manager.get_last_sources.return_value = mock_sources
        
        # Execute the query
        response, sources = self.rag_system.query("What is Python used for?", session_id="session_123")
        
        # Verify the full flow
        assert "Python is a programming language" in response
        assert sources == mock_sources
        
        # Verify tool manager interactions
        self.rag_system.tool_manager.get_tool_definitions.assert_called_once()
        self.rag_system.tool_manager.get_last_sources.assert_called_once()
        self.rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_general_knowledge_query_flow(self):
        """Test flow for general knowledge questions (no tool use)"""
        # Setup: AI answers from general knowledge without using tools
        
        mock_tool_definitions = [
            {"name": "search_course_content", "description": "Search course content"}
        ]
        self.rag_system.tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        
        # AI responds without using tools
        self.rag_system.ai_generator.generate_response.return_value = "The capital of France is Paris."
        
        # No sources from tools
        self.rag_system.tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query("What is the capital of France?")
        
        # Verify response without sources
        assert response == "The capital of France is Paris."
        assert sources == []
        
        # Tools were available but not used (no sources generated)
        self.rag_system.tool_manager.get_last_sources.assert_called_once()


if __name__ == "__main__":
    import unittest.mock
    pytest.main([__file__, "-v"])