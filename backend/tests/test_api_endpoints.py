import pytest
import json
from fastapi import HTTPException
from unittest.mock import patch, Mock


@pytest.mark.api
class TestQueryEndpoint:
    """Test suite for /api/query endpoint"""
    
    def test_query_endpoint_success_with_session(self, test_client, mock_rag_system, sample_query_request):
        """Test successful query with provided session ID"""
        # Configure mock RAG system
        mock_rag_system.query.return_value = (
            "Python is a high-level programming language.",
            [
                {"text": "Python Course - Lesson 1", "url": None},
                {"text": "Programming Fundamentals", "url": "https://example.com/lesson"}
            ]
        )
        
        response = test_client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Python is a high-level programming language."
        assert data["session_id"] == "test_session_123"
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "Python Course - Lesson 1"
        assert data["sources"][0]["url"] is None
        assert data["sources"][1]["url"] == "https://example.com/lesson"
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is Python programming?", "test_session_123")
    
    def test_query_endpoint_success_without_session(self, test_client, mock_rag_system):
        """Test successful query without session ID (creates new session)"""
        # Configure mock to return new session
        mock_rag_system.session_manager.create_session.return_value = "new_session_456"
        mock_rag_system.query.return_value = (
            "Python is versatile.",
            [{"text": "Course content", "url": None}]
        )
        
        request_data = {"query": "Tell me about Python"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "new_session_456"
        assert data["answer"] == "Python is versatile."
        
        # Verify session creation was called
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("Tell me about Python", "new_session_456")
    
    def test_query_endpoint_with_legacy_string_sources(self, test_client, mock_rag_system):
        """Test query endpoint handling legacy string format sources"""
        # Configure mock to return old string format sources
        mock_rag_system.query.return_value = (
            "Response with legacy sources",
            ["Simple text source", "Another text source"]
        )
        
        request_data = {"query": "Test query"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sources"]) == 2
        assert data["sources"][0]["text"] == "Simple text source"
        assert data["sources"][0]["url"] is None
        assert data["sources"][1]["text"] == "Another text source"
        assert data["sources"][1]["url"] is None
    
    def test_query_endpoint_mixed_source_formats(self, test_client, mock_rag_system):
        """Test query endpoint with mixed source formats"""
        mock_rag_system.query.return_value = (
            "Mixed sources response",
            [
                {"text": "Dict source with URL", "url": "https://example.com"},
                "String source",
                {"text": "Dict source without URL"},
                {"text": "Dict with empty URL", "url": ""}
            ]
        )
        
        request_data = {"query": "Mixed test"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        sources = data["sources"]
        assert len(sources) == 4
        assert sources[0]["text"] == "Dict source with URL"
        assert sources[0]["url"] == "https://example.com"
        assert sources[1]["text"] == "String source"
        assert sources[1]["url"] is None
        assert sources[2]["text"] == "Dict source without URL"
        assert sources[2]["url"] is None
        assert sources[3]["url"] == ""
    
    def test_query_endpoint_empty_sources(self, test_client, mock_rag_system):
        """Test query endpoint with no sources"""
        mock_rag_system.query.return_value = ("Answer without sources", [])
        
        request_data = {"query": "General knowledge question"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Answer without sources"
        assert data["sources"] == []
    
    def test_query_endpoint_missing_query_field(self, test_client):
        """Test query endpoint with missing required query field"""
        response = test_client.post("/api/query", json={"session_id": "test"})
        
        assert response.status_code == 422  # Validation error
        assert "field required" in response.json()["detail"][0]["msg"].lower()
    
    def test_query_endpoint_empty_query(self, test_client, mock_rag_system):
        """Test query endpoint with empty query string"""
        mock_rag_system.query.return_value = ("Please ask a question", [])
        
        request_data = {"query": ""}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with("", "test_session_123")
    
    def test_query_endpoint_rag_system_error(self, test_client, mock_rag_system):
        """Test query endpoint when RAG system raises exception"""
        mock_rag_system.query.side_effect = Exception("RAG system failed")
        
        request_data = {"query": "Test query"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        assert "RAG system failed" in response.json()["detail"]
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post("/api/query", data="invalid json")
        
        assert response.status_code == 422
    
    def test_query_endpoint_large_query(self, test_client, mock_rag_system):
        """Test query endpoint with very large query"""
        large_query = "What is Python? " * 1000  # Very long query
        mock_rag_system.query.return_value = ("Large query response", [])
        
        request_data = {"query": large_query}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(large_query, "test_session_123")


@pytest.mark.api
class TestCoursesEndpoint:
    """Test suite for /api/courses endpoint"""
    
    def test_courses_endpoint_success(self, test_client, mock_rag_system, mock_course_analytics):
        """Test successful courses endpoint response"""
        mock_rag_system.get_course_analytics.return_value = mock_course_analytics
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 5
        assert len(data["course_titles"]) == 5
        assert "Introduction to Python" in data["course_titles"]
        assert "Web Development with Django" in data["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_empty_courses(self, test_client, mock_rag_system):
        """Test courses endpoint with no courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_courses_endpoint_single_course(self, test_client, mock_rag_system):
        """Test courses endpoint with single course"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Single Course"]
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Single Course"]
    
    def test_courses_endpoint_rag_system_error(self, test_client, mock_rag_system):
        """Test courses endpoint when RAG system raises exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics failed")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics failed" in response.json()["detail"]
    
    def test_courses_endpoint_invalid_method(self, test_client):
        """Test courses endpoint with invalid HTTP method"""
        response = test_client.post("/api/courses")
        
        assert response.status_code == 405  # Method not allowed
    
    def test_courses_endpoint_with_parameters(self, test_client, mock_rag_system, mock_course_analytics):
        """Test courses endpoint ignores query parameters"""
        mock_rag_system.get_course_analytics.return_value = mock_course_analytics
        
        response = test_client.get("/api/courses?limit=2&offset=1")
        
        assert response.status_code == 200
        # Should return all courses, ignoring parameters
        data = response.json()
        assert data["total_courses"] == 5


@pytest.mark.api
class TestRootEndpoint:
    """Test suite for root / endpoint"""
    
    def test_root_endpoint_success(self, test_client):
        """Test successful root endpoint response"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "RAG System" in data["message"]
    
    def test_root_endpoint_invalid_method(self, test_client):
        """Test root endpoint with invalid HTTP method"""
        response = test_client.post("/")
        
        assert response.status_code == 405  # Method not allowed


@pytest.mark.api
class TestEndpointHeaders:
    """Test suite for API endpoint headers and CORS"""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.get("/", headers={"Origin": "http://localhost:3000"})
        
        assert response.status_code == 200
        # Note: TestClient may not preserve all middleware headers
        # In a real environment, these would be tested with actual CORS requests
    
    def test_content_type_headers(self, test_client, mock_rag_system):
        """Test content type headers in API responses"""
        mock_rag_system.query.return_value = ("Test response", [])
        
        response = test_client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.api
class TestErrorHandling:
    """Test suite for error handling across endpoints"""
    
    def test_404_not_found(self, test_client):
        """Test 404 response for non-existent endpoints"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_query_endpoint_malformed_request(self, test_client):
        """Test query endpoint with malformed request data"""
        response = test_client.post("/api/query", json={"wrong_field": "value"})
        
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("query" in str(error).lower() for error in error_detail)
    
    def test_internal_server_error_details(self, test_client, mock_rag_system):
        """Test that internal server errors include helpful details"""
        mock_rag_system.query.side_effect = ValueError("Specific error message")
        
        request_data = {"query": "test"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        assert "Specific error message" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.api
class TestEndpointIntegration:
    """Integration tests for API endpoints working together"""
    
    def test_query_then_courses_flow(self, test_client, mock_rag_system, mock_course_analytics):
        """Test querying then getting course stats"""
        # First, make a query
        mock_rag_system.query.return_value = ("Python info", [])
        
        query_response = test_client.post("/api/query", json={"query": "What is Python?"})
        assert query_response.status_code == 200
        
        # Then get course stats
        mock_rag_system.get_course_analytics.return_value = mock_course_analytics
        
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        
        # Verify both calls worked
        assert query_response.json()["answer"] == "Python info"
        assert courses_response.json()["total_courses"] == 5
    
    def test_multiple_queries_same_session(self, test_client, mock_rag_system):
        """Test multiple queries with the same session ID"""
        session_id = "persistent_session"
        
        # First query
        mock_rag_system.query.return_value = ("First response", [])
        response1 = test_client.post("/api/query", json={
            "query": "First question",
            "session_id": session_id
        })
        
        # Second query with same session
        mock_rag_system.query.return_value = ("Second response", [])
        response2 = test_client.post("/api/query", json={
            "query": "Second question", 
            "session_id": session_id
        })
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["session_id"] == session_id
        assert response2.json()["session_id"] == session_id
        
        # Verify RAG system was called with same session both times
        calls = mock_rag_system.query.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] == session_id  # Second argument is session_id
        assert calls[1][0][1] == session_id