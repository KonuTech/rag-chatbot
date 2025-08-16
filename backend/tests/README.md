# Testing Framework Enhancement

This document describes the enhanced testing framework for the RAG system.

## New Test Infrastructure

### 1. Pytest Configuration (`pyproject.toml`)
- Added comprehensive pytest configuration with:
  - Test discovery settings (`testpaths`, `python_files`, etc.)
  - Quality of life options (`-v`, `--tb=short`, etc.)
  - Test markers for categorization (unit, integration, api, slow)
  - Added `httpx` dependency for API testing

### 2. Shared Test Fixtures (`conftest.py`)
- **MockConfig**: Standardized configuration for all tests
- **mock_rag_system**: Fully mocked RAG system with default behaviors
- **test_app**: FastAPI test application without static file mounting issues
- **test_client**: TestClient instance for API endpoint testing
- Sample data fixtures for consistent test data

### 3. API Endpoint Tests (`test_api_endpoints.py`)
- **TestQueryEndpoint**: Comprehensive tests for `/api/query`
  - Success scenarios with/without session IDs
  - Error handling (missing fields, RAG system failures)
  - Source format handling (dict vs string, mixed formats)
  - Edge cases (empty queries, large queries)
- **TestCoursesEndpoint**: Tests for `/api/courses`
  - Analytics data retrieval
  - Error scenarios
  - Edge cases (no courses, single course)
- **TestRootEndpoint**: Tests for `/` health check
- **TestErrorHandling**: Cross-endpoint error handling
- **TestEndpointIntegration**: Multi-endpoint integration tests

## Key Features

### Static File Mounting Solution
The test framework solves the FastAPI static file mounting issue by:
- Creating a separate test app without static file dependencies
- Defining API endpoints inline in the test fixtures
- Using comprehensive mocking to avoid filesystem dependencies

### Test Organization
Tests are organized with pytest markers:
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.slow`: Long-running tests

### Fixture Architecture
- **Hierarchical fixtures**: Base fixtures that can be extended
- **Realistic mocking**: Mock behaviors match expected system behavior
- **Reusable components**: Shared fixtures reduce test duplication

## Usage

### Running All Tests
```bash
uv run pytest
```

### Running Specific Test Categories
```bash
# API tests only
uv run pytest -m api

# Integration tests only
uv run pytest -m integration

# Exclude slow tests
uv run pytest -m "not slow"
```

### Running Specific Test Files
```bash
# API endpoint tests only
uv run pytest backend/tests/test_api_endpoints.py

# Verbose output with specific test
uv run pytest backend/tests/test_api_endpoints.py::TestQueryEndpoint::test_query_endpoint_success_with_session -v
```

## Test Coverage

The enhanced framework provides comprehensive coverage for:

1. **API Layer**:
   - Request/response validation
   - Error handling and status codes
   - Session management
   - Source format handling

2. **Integration Points**:
   - RAG system integration
   - Multiple endpoint workflows
   - Session persistence across requests

3. **Edge Cases**:
   - Malformed requests
   - System failures
   - Empty/large data scenarios

## Benefits

1. **Isolated Testing**: Tests run without external dependencies
2. **Fast Execution**: Comprehensive mocking reduces test runtime
3. **Reliable CI/CD**: No static file or external service dependencies
4. **Developer Experience**: Clear error messages and organized test structure
5. **Maintainability**: Shared fixtures reduce code duplication