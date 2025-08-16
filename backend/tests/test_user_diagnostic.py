#!/usr/bin/env python3
"""
User-friendly diagnostic script to verify the RAG system is working correctly.
Run this script if you experience "query failed" errors.
"""
import os
import sys
import time

import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_system_health():
    """Comprehensive system health check"""
    print("🔍 RAG System Health Check")
    print("=" * 50)

    # Test 1: Environment Variables
    print("\n1. Checking environment configuration...")
    from config import config

    if not config.ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY not found in .env file")
        print("   💡 Solution: Copy .env.example to .env and add your API key")
        return False

    if not config.ANTHROPIC_API_KEY.startswith("sk-ant-"):
        print("❌ Invalid ANTHROPIC_API_KEY format")
        print("   💡 Solution: Ensure your API key starts with 'sk-ant-'")
        return False

    print("✅ Environment configuration OK")

    # Test 2: Server Status
    print("\n2. Checking server status...")
    try:
        response = requests.get("http://localhost:8000/api/courses", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server running - {data['total_courses']} courses loaded")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not running")
        print(
            "   💡 Solution: Start server with: cd backend && uv run uvicorn app:app --reload --port 8000"
        )
        return False
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False

    # Test 3: API Query Test
    print("\n3. Testing API query functionality...")
    try:
        test_query = {"query": "What is Python?", "session_id": "health-check"}

        print("   Sending test query...")
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/api/query", json=test_query, timeout=30
        )
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query successful in {end_time - start_time:.1f}s")
            print(f"   Response length: {len(data['answer'])} characters")
            print(f"   Sources found: {len(data['sources'])}")
        else:
            print(f"❌ Query failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Query timed out (>30s)")
        print("   💡 This suggests an issue with the AI API or network")
        return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

    # Test 4: Frontend Connectivity
    print("\n4. Testing frontend connectivity...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend accessible")
        else:
            print(f"❌ Frontend returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend error: {e}")
        print("   💡 Check if static files are properly served")

    print("\n🎉 All tests passed! Your RAG system is working correctly.")
    print("\nIf you still see 'query failed' in the browser:")
    print("1. Clear browser cache and refresh")
    print("2. Check browser console for JavaScript errors (F12)")
    print("3. Try a different browser")
    print("4. Check network connectivity")

    return True


def main():
    """Run diagnostic tests"""
    if not test_system_health():
        print("\n❌ System health check failed")
        print("Please fix the issues above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
