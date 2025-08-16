import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_with_successful_search_results(self):
        """Test execute method with successful search results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is lesson content about Python basics", "Another lesson about variables"],
            metadata=[
                {"course_title": "Python Fundamentals", "lesson_number": 1},
                {"course_title": "Python Fundamentals", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("Python basics")
        
        # Verify the vector store was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )
        
        # Verify formatted results
        assert "[Python Fundamentals - Lesson 1]" in result
        assert "[Python Fundamentals - Lesson 2]" in result
        assert "This is lesson content about Python basics" in result
        assert "Another lesson about variables" in result
        
        # Verify sources were stored
        assert len(self.search_tool.last_sources) == 2
        assert self.search_tool.last_sources[0]["text"] == "Python Fundamentals - Lesson 1"
        assert self.search_tool.last_sources[1]["text"] == "Python Fundamentals - Lesson 2"
    
    def test_execute_with_course_name_filter(self):
        """Test execute method with course name filtering"""
        mock_results = SearchResults(
            documents=["Course content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("functions", course_name="Advanced Python")
        
        # Verify the vector store was called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name="Advanced Python",
            lesson_number=None
        )
        
        assert "Advanced Python" in result
    
    def test_execute_with_lesson_number_filter(self):
        """Test execute method with lesson number filtering"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("loops", lesson_number=5)
        
        # Verify the vector store was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="loops",
            course_name=None,
            lesson_number=5
        )
        
        assert "Lesson 5" in result
    
    def test_execute_with_both_filters(self):
        """Test execute method with both course name and lesson number filters"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("pandas", course_name="Data Science", lesson_number=2)
        
        # Verify the vector store was called with both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="pandas",
            course_name="Data Science",
            lesson_number=2
        )
        
        assert "Data Science" in result
        assert "Lesson 2" in result
    
    def test_execute_with_empty_results(self):
        """Test execute method when no results are found"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        assert "No relevant content found" in result
    
    def test_execute_with_empty_results_and_filters(self):
        """Test execute method when no results are found with filters"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent", course_name="Unknown Course", lesson_number=99)
        
        assert "No relevant content found in course 'Unknown Course' in lesson 99" in result
    
    def test_execute_with_search_error(self):
        """Test execute method when vector store returns an error"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        assert result == "Database connection failed"
    
    def test_execute_with_missing_metadata(self):
        """Test execute method with incomplete metadata"""
        mock_results = SearchResults(
            documents=["Content without full metadata"],
            metadata=[{"course_title": "Incomplete Course"}],  # Missing lesson_number
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test")
        
        # Should handle missing lesson_number gracefully
        assert "[Incomplete Course]" in result
        assert "Content without full metadata" in result
        
        # Source should not include lesson number
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0]["text"] == "Incomplete Course"
    
    def test_get_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        definition = self.search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        
        # Check required parameters
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
        
        # Check that only query is required
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_vector_store_integration_error(self):
        """Test handling of vector store exceptions"""
        # Mock vector store to raise exception
        self.mock_vector_store.search.side_effect = Exception("ChromaDB connection failed")
        
        # This should be handled by the vector store's search method,
        # but if it bubbles up, we should test the behavior
        try:
            result = self.search_tool.execute("test query")
            # If we get here, the exception was caught and handled
            assert "error" in result.lower() or "failed" in result.lower()
        except Exception as e:
            # If exception bubbles up, that's also a valid test result
            # indicating the search tool doesn't handle vector store exceptions
            assert "ChromaDB connection failed" in str(e)
    
    def test_sources_reset_between_searches(self):
        """Test that sources are properly updated between searches"""
        # First search
        mock_results1 = SearchResults(
            documents=["First content"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results1
        
        self.search_tool.execute("first query")
        first_sources = self.search_tool.last_sources.copy()
        
        # Second search
        mock_results2 = SearchResults(
            documents=["Second content", "Third content"],
            metadata=[
                {"course_title": "Course 2", "lesson_number": 2},
                {"course_title": "Course 2", "lesson_number": 3}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results2
        
        self.search_tool.execute("second query")
        second_sources = self.search_tool.last_sources
        
        # Verify sources were updated, not appended
        assert len(first_sources) == 1
        assert len(second_sources) == 2
        assert second_sources != first_sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])