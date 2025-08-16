"""
Integration test for multi-round sequential tool calling.
Tests the complete flow from user query to final response.
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator_v2 import AIGeneratorV2

# Import components
from component_interfaces import ReasoningConfig, ReasoningSession, TerminationReason
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager


class TestMultiRoundIntegration:
    """Integration tests for multi-round sequential tool calling"""

    def setup_method(self):
        """Set up test environment"""
        # Create mock tool manager similar to the real one
        self.mock_tool_manager = Mock()
        self.mock_tool_manager.tools = {}

        # Configure tool definitions
        self.mock_tool_manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the course content",
                        },
                        "course_name": {
                            "type": "string",
                            "description": "Course title (partial matches work)",
                        },
                        "lesson_number": {
                            "type": "integer",
                            "description": "Specific lesson number to search within",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_course_outline",
                "description": "Get course outline including title, link, and complete lesson list",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "course_name": {
                            "type": "string",
                            "description": "Course title or partial name to get outline for",
                        }
                    },
                    "required": ["course_name"],
                },
            },
        ]

        # Configure tool execution responses
        self.setup_tool_responses()

        # Configure source tracking
        self.mock_tool_manager.get_last_sources.return_value = []
        self.mock_tool_manager.reset_sources.return_value = None

        # Create AI Generator V2
        self.ai_generator = AIGeneratorV2(
            api_key="test-api-key",
            model="claude-sonnet-4-20250514",
            tool_manager=self.mock_tool_manager,
        )

        # Mock the Anthropic client
        self.mock_client = Mock()
        self.ai_generator.client = self.mock_client
        self.ai_generator.reasoning_engine.client = self.mock_client

    def setup_tool_responses(self):
        """Configure realistic tool execution responses"""

        def tool_response_side_effect(tool_name, **kwargs):
            if tool_name == "search_course_content":
                query = kwargs.get("query", "")
                course_name = kwargs.get("course_name", "")

                if "Python" in query or "Python" in course_name:
                    return "[Python Fundamentals - Lesson 1]\nPython is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
                elif "Java" in query or "Java" in course_name:
                    return "[Java Programming - Lesson 1]\nJava is a class-based, object-oriented programming language designed to be platform-independent. It follows the principle of 'write once, run anywhere' (WORA)."
                else:
                    return f"[Course Search Results]\nFound information about {query} in multiple courses."

            elif tool_name == "get_course_outline":
                course_name = kwargs.get("course_name", "")

                if "Python" in course_name:
                    return """Course: Python Fundamentals
Instructor: Dr. Sarah Smith
Course Link: https://example.com/python-fundamentals

Lessons:
  Lesson 1: Introduction to Python
  Lesson 2: Variables and Data Types
  Lesson 3: Control Structures
  Lesson 4: Functions and Modules
  Lesson 5: Object-Oriented Programming"""

                elif "Java" in course_name:
                    return """Course: Java Programming
Instructor: Prof. Michael Johnson
Course Link: https://example.com/java-programming

Lessons:
  Lesson 1: Java Basics
  Lesson 2: Object-Oriented Concepts
  Lesson 3: Inheritance and Polymorphism
  Lesson 4: Exception Handling
  Lesson 5: Collections Framework"""

                else:
                    return f"No course found matching '{course_name}'"

            return "Tool execution failed"

        self.mock_tool_manager.execute_tool.side_effect = tool_response_side_effect

    def test_single_round_query_completion(self):
        """Test query that completes in a single round (no tools needed)"""
        # Configure mock for direct response
        mock_response = Mock()
        mock_response.content = [Mock(text="2 + 2 equals 4")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Test simple math query
        result = self.ai_generator.generate_response(
            query="What is 2 + 2?",
            tools=self.mock_tool_manager.get_tool_definitions(),
            tool_manager=self.mock_tool_manager,
        )

        assert "4" in result
        # Should use simple response path, not multi-round
        self.mock_tool_manager.execute_tool.assert_not_called()

    @patch("asyncio.run")
    def test_single_round_with_tool_use(self, mock_asyncio_run):
        """Test query that uses tools but completes in one round"""
        # Configure mock session for single round with tool use
        mock_session = Mock()
        mock_session.termination_reason = TerminationReason.NATURAL_COMPLETION
        mock_session.rounds = [Mock()]
        mock_asyncio_run.return_value = mock_session

        # Configure response assembler using patch
        with patch.object(
            self.ai_generator.response_assembler, "assemble_final_response"
        ) as mock_assemble:
            mock_assemble.return_value = (
                "Python is a high-level programming language known for its simplicity and readability.",
                [{"text": "Python Fundamentals - Lesson 1", "url": None}],
            )

            result = self.ai_generator.generate_response(
                query="What is Python?",
                tools=self.mock_tool_manager.get_tool_definitions(),
                tool_manager=self.mock_tool_manager,
            )

            assert "Python" in result
            assert "programming language" in result
            mock_asyncio_run.assert_called_once()

    @patch("asyncio.run")
    def test_multi_round_comparison_query(self, mock_asyncio_run):
        """Test query that requires multiple rounds for comparison"""
        # Configure mock session for multi-round execution
        mock_session = Mock()
        mock_session.termination_reason = TerminationReason.MAX_ROUNDS_REACHED
        mock_session.rounds = [Mock(), Mock()]  # Two rounds
        mock_asyncio_run.return_value = mock_session

        # Configure response assembler for comparison result
        comparison_response = """Based on my research of both courses:

Python Fundamentals focuses on simplicity and readability, making it ideal for beginners. Python supports multiple programming paradigms and is known for its clean syntax.

Java Programming emphasizes object-oriented design and platform independence. Java follows the "write once, run anywhere" principle and has a more structured approach to programming.

Key differences:
- Python: More beginner-friendly, flexible syntax
- Java: More structured, enterprise-focused, strongly typed"""

        self.ai_generator.response_assembler.assemble_final_response.return_value = (
            comparison_response,
            [
                {"text": "Python Fundamentals - Lesson 1", "url": None},
                {"text": "Java Programming - Lesson 1", "url": None},
            ],
        )

        result = self.ai_generator.generate_response(
            query="Compare the Python and Java programming courses. What are the key differences in their approaches?",
            tools=self.mock_tool_manager.get_tool_definitions(),
            tool_manager=self.mock_tool_manager,
        )

        assert "Python" in result
        assert "Java" in result
        assert "differences" in result or "comparison" in result.lower()
        mock_asyncio_run.assert_called_once()

    @patch("asyncio.run")
    def test_max_rounds_termination(self, mock_asyncio_run):
        """Test that system properly terminates at max rounds"""
        # Configure session that hits max rounds
        mock_session = Mock()
        mock_session.termination_reason = TerminationReason.MAX_ROUNDS_REACHED
        mock_session.rounds = [Mock(), Mock()]  # Exactly 2 rounds (max)
        mock_asyncio_run.return_value = mock_session

        # Configure partial response
        self.ai_generator.response_assembler.assemble_final_response.return_value = (
            "I've gathered information from multiple courses but need more time to provide a complete comparison.",
            [{"text": "Course Information", "url": None}],
        )

        result = self.ai_generator.generate_response(
            query="Provide a detailed analysis comparing all programming courses, their methodologies, student outcomes, and industry applications.",
            tools=self.mock_tool_manager.get_tool_definitions(),
            tool_manager=self.mock_tool_manager,
        )

        # Should still provide a useful response even with early termination
        assert len(result) > 0
        assert isinstance(result, str)
        mock_asyncio_run.assert_called_once()

    @patch("asyncio.run")
    def test_error_handling_with_fallback(self, mock_asyncio_run):
        """Test error handling and fallback to simple response"""
        # Configure multi-round to fail
        mock_asyncio_run.side_effect = Exception("Multi-round processing failed")

        # Configure fallback simple response
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text="I can help you with course-related questions. Could you please be more specific?"
            )
        ]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        result = self.ai_generator.generate_response(
            query="Tell me about programming courses",
            tools=self.mock_tool_manager.get_tool_definitions(),
            tool_manager=self.mock_tool_manager,
        )

        # Should fall back gracefully
        assert "help you" in result
        assert len(result) > 0

    def test_configuration_customization(self):
        """Test that configuration can be customized"""
        # Test initial configuration
        config = self.ai_generator.get_config()
        assert config.max_rounds == 2  # Default

        # Test configuration update
        self.ai_generator.update_config(max_rounds=3, max_tokens_per_round=1000)

        updated_config = self.ai_generator.get_config()
        assert updated_config.max_rounds == 3
        assert updated_config.max_tokens_per_round == 1000

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked"""
        # Get initial metrics
        session_metrics = self.ai_generator.get_session_metrics()
        tool_metrics = self.ai_generator.get_tool_metrics()

        assert isinstance(session_metrics, dict)
        assert isinstance(tool_metrics, dict)

        # Reset metrics
        self.ai_generator.reset_metrics()

        # Metrics should still be accessible after reset
        session_metrics_after = self.ai_generator.get_session_metrics()
        assert isinstance(session_metrics_after, dict)

    def test_source_tracking_integration(self):
        """Test that sources are properly tracked and returned"""
        # Configure source tracking
        test_sources = [
            {
                "text": "Python Fundamentals - Lesson 1",
                "url": "https://example.com/python/lesson1",
            },
            {
                "text": "Java Programming - Lesson 2",
                "url": "https://example.com/java/lesson2",
            },
        ]

        # Create a mock search tool that tracks sources
        mock_search_tool = Mock()
        mock_search_tool.last_sources = test_sources
        self.mock_tool_manager.tools = {"search_course_content": mock_search_tool}

        # Configure get_last_sources to return our test sources
        self.mock_tool_manager.get_last_sources.return_value = test_sources

        with patch("asyncio.run") as mock_asyncio_run:
            mock_session = Mock()
            mock_session.termination_reason = TerminationReason.NATURAL_COMPLETION
            mock_asyncio_run.return_value = mock_session

            # Configure response assembler to return sources
            self.ai_generator.response_assembler.assemble_final_response.return_value = (
                "Response with sources",
                test_sources,
            )

            result = self.ai_generator.generate_response(
                query="What programming languages are covered?",
                tools=self.mock_tool_manager.get_tool_definitions(),
                tool_manager=self.mock_tool_manager,
            )

            # Verify sources were processed
            self.mock_tool_manager.reset_sources.assert_called()


class TestRealWorldScenarios:
    """Test realistic multi-round scenarios"""

    def setup_method(self):
        """Set up for scenario testing"""
        self.mock_tool_manager = Mock()
        self.mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get course outline"},
        ]
        self.mock_tool_manager.get_last_sources.return_value = []
        self.mock_tool_manager.reset_sources.return_value = None

        self.ai_generator = AIGeneratorV2(
            api_key="test-key", model="test-model", tool_manager=self.mock_tool_manager
        )

    @patch("asyncio.run")
    def test_progressive_refinement_scenario(self, mock_asyncio_run):
        """Test: Find Python concepts, then show examples from specific lessons"""
        mock_session = Mock()
        mock_session.termination_reason = TerminationReason.NATURAL_COMPLETION
        mock_asyncio_run.return_value = mock_session

        # Configure progressive refinement response
        self.ai_generator.response_assembler.assemble_final_response.return_value = (
            "Python covers key concepts like variables, functions, and classes. Here are specific examples: In Lesson 1, you'll learn about variable assignment (x = 5). Lesson 3 introduces function definition (def my_function():). Lesson 5 covers class creation (class MyClass:).",
            [{"text": "Python Fundamentals - Multiple Lessons", "url": None}],
        )

        result = self.ai_generator.generate_response(
            query="What are the main Python concepts covered, and can you show me specific examples from the lessons?",
            tools=self.mock_tool_manager.get_tool_definitions(),
            tool_manager=self.mock_tool_manager,
        )

        assert "concepts" in result.lower()
        assert "examples" in result.lower()
        mock_asyncio_run.assert_called_once()

    @patch("asyncio.run")
    def test_cross_course_analysis_scenario(self, mock_asyncio_run):
        """Test: What are the main themes across all courses, then detail examples from each"""
        mock_session = Mock()
        mock_session.termination_reason = TerminationReason.MAX_ROUNDS_REACHED
        mock_asyncio_run.return_value = mock_session

        # Configure cross-course analysis response
        self.ai_generator.response_assembler.assemble_final_response.return_value = (
            "Common themes across programming courses include: 1) Problem-solving methodologies, 2) Data structure fundamentals, 3) Algorithm design patterns. Python emphasizes readability and rapid prototyping. Java focuses on enterprise-scale development and object-oriented design principles.",
            [
                {"text": "Python Fundamentals", "url": None},
                {"text": "Java Programming", "url": None},
                {"text": "Data Structures Course", "url": None},
            ],
        )

        result = self.ai_generator.generate_response(
            query="What are the main themes across all programming courses? Give me detailed examples from each course showing how they approach these themes.",
            tools=self.mock_tool_manager.get_tool_definitions(),
            tool_manager=self.mock_tool_manager,
        )

        assert "themes" in result.lower()
        assert len(result) > 100  # Should be substantial response
        mock_asyncio_run.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
