import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import Config


class TestRealSystemIntegration:
    """Integration tests using real system components to identify actual failures"""
    
    def setup_method(self):
        """Set up real system with temporary data"""
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test config that uses our temp directory
        self.config = Config()
        self.config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma")
        self.config.ANTHROPIC_API_KEY = "test-key-will-be-mocked"
        
        # Create test course document
        self.test_doc_path = os.path.join(self.temp_dir, "test_course.txt")
        with open(self.test_doc_path, 'w') as f:
            f.write("""Course Title: Python Fundamentals
Course Link: https://example.com/python-course
Course Instructor: Dr. Smith

Lesson 0: Introduction to Python
Lesson Link: https://example.com/lesson0
Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.

Lesson 1: Variables and Data Types
Lesson Link: https://example.com/lesson1
In Python, variables are created when you assign a value to them. Python has several data types including integers, floats, strings, and booleans.

Lesson 2: Control Structures
Python provides several control structures including if statements, for loops, and while loops. These allow you to control the flow of your program execution.
""")
        
        # Initialize RAG system (but don't initialize AI generator yet)
        self.rag_system = RAGSystem(self.config)
    
    def teardown_method(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_document_processing_and_vector_storage(self):
        """Test that documents are properly processed and stored in vector store"""
        # Add the test document
        course, chunks_count = self.rag_system.add_course_document(self.test_doc_path)
        
        # Verify course was processed
        assert course is not None
        assert course.title == "Python Fundamentals"
        assert course.instructor == "Dr. Smith"
        assert course.course_link == "https://example.com/python-course"
        assert len(course.lessons) == 3
        assert chunks_count > 0
        
        print(f"Course processed: {course.title} with {chunks_count} chunks")
        
        # Verify data was stored in vector store
        course_count = self.rag_system.vector_store.get_course_count()
        assert course_count == 1
        
        existing_titles = self.rag_system.vector_store.get_existing_course_titles()
        assert "Python Fundamentals" in existing_titles
    
    def test_vector_store_search_functionality(self):
        """Test that vector store search is working"""
        # Add test document first
        self.rag_system.add_course_document(self.test_doc_path)
        
        # Test direct vector store search
        results = self.rag_system.vector_store.search("Python programming language")
        
        print(f"Search results: {len(results.documents)} documents found")
        print(f"Error: {results.error}")
        print(f"Documents: {results.documents}")
        print(f"Metadata: {results.metadata}")
        
        # Should find relevant content
        if results.error:
            pytest.fail(f"Vector store search failed with error: {results.error}")
        
        assert not results.is_empty(), "Vector store search returned no results"
        assert len(results.documents) > 0
        
        # Check that results contain relevant content
        found_python_content = any("python" in doc.lower() for doc in results.documents)
        assert found_python_content, "Search results don't contain Python-related content"
    
    def test_course_search_tool_execution(self):
        """Test CourseSearchTool execution with real data"""
        # Add test document
        self.rag_system.add_course_document(self.test_doc_path)
        
        # Test search tool directly
        search_tool = self.rag_system.search_tool
        
        # Test basic search
        result = search_tool.execute("Python programming")
        print(f"Search tool result: {result}")
        
        assert "error" not in result.lower(), f"Search tool returned error: {result}"
        assert "python" in result.lower(), "Search tool didn't return Python-related content"
        assert "[Python Fundamentals" in result, "Search tool didn't format results with course context"
        
        # Test search with course filter
        result_filtered = search_tool.execute("variables", course_name="Python Fundamentals")
        print(f"Filtered search result: {result_filtered}")
        
        assert "variables" in result_filtered.lower(), "Filtered search didn't return relevant content"
    
    def test_tool_manager_functionality(self):
        """Test that tool manager correctly executes tools"""
        # Add test document
        self.rag_system.add_course_document(self.test_doc_path)
        
        # Get tool definitions
        tool_definitions = self.rag_system.tool_manager.get_tool_definitions()
        print(f"Available tools: {[tool['name'] for tool in tool_definitions]}")
        
        assert len(tool_definitions) >= 1, "No tools available in tool manager"
        
        # Find search tool
        search_tool_def = next((tool for tool in tool_definitions if tool['name'] == 'search_course_content'), None)
        assert search_tool_def is not None, "search_course_content tool not found"
        
        # Execute tool via tool manager
        result = self.rag_system.tool_manager.execute_tool(
            "search_course_content", 
            query="Python variables"
        )
        
        print(f"Tool manager execution result: {result}")
        assert "error" not in result.lower(), f"Tool manager execution failed: {result}"
        assert "python" in result.lower(), "Tool manager didn't return relevant content"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_ai_generator_with_mocked_api(self, mock_anthropic_class):
        """Test AI generator with mocked Anthropic API"""
        # Add test document
        self.rag_system.add_course_document(self.test_doc_path)
        
        # Mock Anthropic client
        mock_client = mock_anthropic_class.return_value
        mock_response = type('MockResponse', (), {
            'content': [type('MockContent', (), {'text': 'Python is a programming language used for various applications.'})()],
            'stop_reason': 'end_turn'
        })()
        mock_client.messages.create.return_value = mock_response
        
        # Test AI generator directly
        response = self.rag_system.ai_generator.generate_response(
            "What is Python?",
            tools=self.rag_system.tool_manager.get_tool_definitions(),
            tool_manager=self.rag_system.tool_manager
        )
        
        print(f"AI generator response: {response}")
        assert response == "Python is a programming language used for various applications."
        
        # Verify API was called
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == self.config.ANTHROPIC_MODEL
        assert "tools" in call_args
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_full_rag_query_flow(self, mock_anthropic_class):
        """Test the complete RAG query flow with mocked AI API"""
        # Add test document
        self.rag_system.add_course_document(self.test_doc_path)
        
        # Mock Anthropic client to simulate tool use
        mock_client = mock_anthropic_class.return_value
        
        # First response: AI decides to use search tool
        mock_tool_content = type('MockToolContent', (), {
            'type': 'tool_use',
            'name': 'search_course_content',
            'id': 'tool_123',
            'input': {'query': 'Python variables'}
        })()
        
        mock_initial_response = type('MockResponse', (), {
            'content': [mock_tool_content],
            'stop_reason': 'tool_use'
        })()
        
        # Final response: AI uses tool results
        mock_final_response = type('MockResponse', (), {
            'content': [type('MockContent', (), {
                'text': 'Based on the course content, Python variables are created when you assign a value to them. Python supports several data types including integers, floats, strings, and booleans.'
            })()],
            'stop_reason': 'end_turn'
        })()
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Execute full RAG query
        response, sources = self.rag_system.query("Tell me about Python variables")
        
        print(f"RAG system response: {response}")
        print(f"RAG system sources: {sources}")
        
        # Verify response
        assert "variables" in response.lower(), "Response doesn't mention variables"
        assert "python" in response.lower(), "Response doesn't mention Python"
        
        # Verify sources were provided
        assert len(sources) > 0, "No sources returned from RAG query"
        assert any("Python Fundamentals" in str(source) for source in sources), "Sources don't contain course information"
        
        # Verify AI API was called twice (initial + tool execution)
        assert mock_client.messages.create.call_count == 2
    
    def test_error_scenarios(self):
        """Test various error scenarios"""
        # Test query without any documents
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = mock_anthropic.return_value
            mock_response = type('MockResponse', (), {
                'content': [type('MockContent', (), {'text': 'I don\'t have access to course materials about that topic.'})()],
                'stop_reason': 'end_turn'
            })()
            mock_client.messages.create.return_value = mock_response
            
            response, sources = self.rag_system.query("What is machine learning?")
            
            print(f"Response with no documents: {response}")
            print(f"Sources with no documents: {sources}")
            
            # Should still get a response (even if no course content available)
            assert response is not None
            assert len(response) > 0
    
    def test_vector_store_error_propagation(self):
        """Test that vector store errors are properly handled"""
        # Add document first
        self.rag_system.add_course_document(self.test_doc_path)
        
        # Simulate vector store failure by corrupting the database
        # This will help us see if errors are properly propagated
        original_search = self.rag_system.vector_store.search
        
        def failing_search(*args, **kwargs):
            from vector_store import SearchResults
            return SearchResults.empty("Simulated database failure")
        
        self.rag_system.vector_store.search = failing_search
        
        # Test search tool with failing vector store
        result = self.rag_system.search_tool.execute("test query")
        print(f"Search tool result with DB failure: {result}")
        
        assert "simulated database failure" in result.lower(), "Vector store error not properly propagated"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])