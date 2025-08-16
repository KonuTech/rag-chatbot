#!/usr/bin/env python3
"""
Debug script to test the actual running system and identify the "query failed" issue
"""
import sys
import os
import requests
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag_system import RAGSystem


def test_config_and_api_key():
    """Test configuration and API key availability"""
    print("=== Configuration Test ===")
    print(f"ANTHROPIC_API_KEY present: {bool(config.ANTHROPIC_API_KEY)}")
    print(f"ANTHROPIC_MODEL: {config.ANTHROPIC_MODEL}")
    print(f"CHROMA_PATH: {config.CHROMA_PATH}")
    print(f"API key length: {len(config.ANTHROPIC_API_KEY) if config.ANTHROPIC_API_KEY else 0}")
    
    if not config.ANTHROPIC_API_KEY:
        print("❌ ERROR: No ANTHROPIC_API_KEY found in environment!")
        return False
    
    if not config.ANTHROPIC_API_KEY.startswith('sk-ant-'):
        print("❌ ERROR: ANTHROPIC_API_KEY doesn't appear to be valid (should start with 'sk-ant-')")
        return False
    
    print("✅ Configuration looks good")
    return True


def test_rag_system_initialization():
    """Test RAG system initialization"""
    print("\n=== RAG System Initialization Test ===")
    try:
        rag_system = RAGSystem(config)
        print("✅ RAG system initialized successfully")
        
        # Test course analytics
        analytics = rag_system.get_course_analytics()
        print(f"Total courses: {analytics['total_courses']}")
        print(f"Course titles: {analytics['course_titles']}")
        
        if analytics['total_courses'] == 0:
            print("⚠️  WARNING: No courses loaded in the system")
        
        return rag_system
    except Exception as e:
        print(f"❌ ERROR initializing RAG system: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_ai_generator_directly():
    """Test AI generator with a simple query"""
    print("\n=== AI Generator Direct Test ===")
    try:
        from ai_generator import AIGenerator
        
        ai_gen = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)
        
        # Test simple query without tools
        print("Testing simple query without tools...")
        response = ai_gen.generate_response("What is 2 + 2?")
        print(f"AI response: {response}")
        print("✅ AI Generator works for basic queries")
        return True
        
    except Exception as e:
        print(f"❌ ERROR with AI Generator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_query():
    """Test a full RAG query"""
    print("\n=== Full RAG Query Test ===")
    try:
        rag_system = RAGSystem(config)
        
        # Test with a content-related query
        print("Testing content-related query...")
        response, sources = rag_system.query("What is Python?")
        
        print(f"Response: {response}")
        print(f"Sources: {sources}")
        
        if "error" in response.lower() or "failed" in response.lower():
            print("❌ ERROR: Query returned error message")
            return False
        
        print("✅ RAG query completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ ERROR with RAG query: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoint():
    """Test the actual API endpoint if server is running"""
    print("\n=== API Endpoint Test ===")
    try:
        # Test if server is running
        response = requests.get("http://localhost:8000/api/courses", timeout=5)
        print(f"Courses endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Server is running and courses endpoint works")
            courses_data = response.json()
            print(f"API reports {courses_data['total_courses']} courses")
        
        # Test query endpoint
        query_data = {
            "query": "What is Python?",
            "session_id": "test-session"
        }
        
        response = requests.post(
            "http://localhost:8000/api/query", 
            json=query_data,
            timeout=30
        )
        
        print(f"Query endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {result['answer'][:100]}...")
            print(f"Sources count: {len(result['sources'])}")
            print("✅ API endpoint works correctly")
        else:
            print(f"❌ API endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
        
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running - cannot test API endpoint")
    except Exception as e:
        print(f"❌ ERROR testing API endpoint: {e}")


def main():
    """Run all debug tests"""
    print("Starting system debug tests...\n")
    
    # Test 1: Configuration
    if not test_config_and_api_key():
        print("\n❌ STOPPING: Configuration issues detected")
        return
    
    # Test 2: RAG System
    rag_system = test_rag_system_initialization()
    if not rag_system:
        print("\n❌ STOPPING: RAG system initialization failed")
        return
    
    # Test 3: AI Generator
    if not test_ai_generator_directly():
        print("\n❌ STOPPING: AI Generator failed")
        return
    
    # Test 4: Full RAG Query
    if not test_rag_query():
        print("\n❌ STOPPING: RAG query failed")
        return
    
    # Test 5: API Endpoint (if running)
    test_api_endpoint()
    
    print("\n✅ All tests completed successfully!")
    print("\nIf you're still seeing 'query failed', the issue might be:")
    print("1. Frontend JavaScript error")
    print("2. Network connectivity issue")
    print("3. Browser cache issue")


if __name__ == "__main__":
    main()