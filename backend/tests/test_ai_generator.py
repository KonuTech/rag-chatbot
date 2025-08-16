import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator

try:
    from ai_generator_v2 import AIGeneratorV2

    HAS_V2 = True
except ImportError:
    HAS_V2 = False


class TestAIGenerator:
    """Test suite for AIGenerator tool calling functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create AIGenerator with mock API key
        self.ai_generator = AIGenerator(
            api_key="test-key", model="claude-sonnet-4-20250514"
        )

        # Mock the Anthropic client
        self.mock_client = Mock()
        self.ai_generator.client = self.mock_client

    def test_generate_response_without_tools(self):
        """Test basic response generation without tools"""
        # Mock response from Anthropic
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a general knowledge answer")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        result = self.ai_generator.generate_response("What is Python?")

        # Verify the API was called correctly
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args[1]

        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args

        assert result == "This is a general knowledge answer"

    def test_generate_response_with_tools_no_tool_use(self):
        """Test response generation with tools available but not used"""
        # Mock response from Anthropic (no tool use)
        mock_response = Mock()
        mock_response.content = [
            Mock(text="This is a general answer without using tools")
        ]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Mock tools and tool manager
        mock_tools = [
            {"name": "search_course_content", "description": "Search courses"}
        ]
        mock_tool_manager = Mock()

        result = self.ai_generator.generate_response(
            "What is the capital of France?",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify the API was called with tools
        call_args = self.mock_client.messages.create.call_args[1]
        assert call_args["tools"] == mock_tools
        assert call_args["tool_choice"] == {"type": "auto"}

        # Tool manager should not be called since no tool was used
        mock_tool_manager.execute_tool.assert_not_called()

        assert result == "This is a general answer without using tools"

    def test_generate_response_with_tool_use(self):
        """Test response generation with tool use"""
        # Mock initial response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_call_123"
        mock_tool_content.input = {"query": "Python basics"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(
                text="Based on the course content, Python is a programming language..."
            )
        ]

        # Set up client to return both responses
        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Course content about Python basics from Lesson 1"
        )

        mock_tools = [
            {"name": "search_course_content", "description": "Search courses"}
        ]

        result = self.ai_generator.generate_response(
            "Tell me about Python basics from the course",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Verify two API calls were made
        assert self.mock_client.messages.create.call_count == 2

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify final result
        assert (
            result == "Based on the course content, Python is a programming language..."
        )

    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Answer with context")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        conversation_history = (
            "User: What is Python?\nAssistant: Python is a programming language."
        )

        result = self.ai_generator.generate_response(
            "Can you tell me more?", conversation_history=conversation_history
        )

        # Verify system prompt includes conversation history
        call_args = self.mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert "Previous conversation:" in system_content
        assert conversation_history in system_content

    def test_handle_tool_execution_with_multiple_tools(self):
        """Test handling multiple tool calls in one response"""
        # Create mock tool contents
        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.name = "search_course_content"
        tool1.id = "tool_1"
        tool1.input = {"query": "Python"}

        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.name = "get_course_outline"
        tool2.id = "tool_2"
        tool2.input = {"course_name": "Python Fundamentals"}

        mock_initial_response = Mock()
        mock_initial_response.content = [tool1, tool2]

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Response using both tools")]
        self.mock_client.messages.create.return_value = mock_final_response

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result for Python",
            "Course outline for Python Fundamentals",
        ]

        base_params = {
            "messages": [{"role": "user", "content": "Tell me about Python course"}],
            "system": "System prompt",
        }

        result = self.ai_generator._handle_tool_execution(
            mock_initial_response, base_params, mock_tool_manager
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="Python"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_name="Python Fundamentals"
        )

        # Verify final API call structure
        final_call_args = self.mock_client.messages.create.call_args[1]
        messages = final_call_args["messages"]

        # Should have: original user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Tool results should be in the final user message
        tool_results = messages[2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[0]["content"] == "Search result for Python"
        assert tool_results[1]["type"] == "tool_result"
        assert tool_results[1]["tool_use_id"] == "tool_2"
        assert tool_results[1]["content"] == "Course outline for Python Fundamentals"

    def test_api_error_handling(self):
        """Test handling of API errors"""
        # Mock API to raise an exception
        self.mock_client.messages.create.side_effect = Exception(
            "API rate limit exceeded"
        )

        with pytest.raises(Exception) as exc_info:
            self.ai_generator.generate_response("Test query")

        assert "API rate limit exceeded" in str(exc_info.value)

    def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors"""
        # Mock initial response with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_call_123"
        mock_tool_content.input = {"query": "test"}

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Error handled response")]

        self.mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Mock tool manager to return error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = (
            "Tool error: Database connection failed"
        )

        mock_tools = [{"name": "search_course_content"}]

        result = self.ai_generator.generate_response(
            "Test query", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Should still complete the flow and return a response
        assert result == "Error handled response"

        # Tool error should be passed back to AI
        final_call_args = self.mock_client.messages.create.call_args[1]
        tool_results = final_call_args["messages"][2]["content"]
        assert "Tool error: Database connection failed" in tool_results[0]["content"]

    def test_system_prompt_structure(self):
        """Test that system prompt contains required elements"""
        system_prompt = AIGenerator.SYSTEM_PROMPT

        # Check for key components
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "One tool call per query maximum" in system_prompt
        assert "general knowledge questions" in system_prompt.lower()
        assert "course content questions" in system_prompt.lower()

        # Verify response guidelines
        assert "Brief, Concise and focused" in system_prompt
        assert "Educational" in system_prompt
        assert "Clear" in system_prompt

    def test_base_parameters_configuration(self):
        """Test that base parameters are correctly configured"""
        assert self.ai_generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert self.ai_generator.base_params["temperature"] == 0
        assert self.ai_generator.base_params["max_tokens"] == 800


@pytest.mark.skipif(not HAS_V2, reason="AIGeneratorV2 not available")
class TestAIGeneratorV2Compatibility:
    """Test suite for AIGeneratorV2 backward compatibility"""

    def setup_method(self):
        """Set up test fixtures for V2"""
        # Create mock tool manager
        self.mock_tool_manager = Mock()
        self.mock_tool_manager.tools = {}
        self.mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"}
        ]
        self.mock_tool_manager.execute_tool.return_value = "Mock search result"
        self.mock_tool_manager.get_last_sources.return_value = []
        self.mock_tool_manager.reset_sources.return_value = None

        # Create AIGeneratorV2
        self.ai_generator_v2 = AIGeneratorV2(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            tool_manager=self.mock_tool_manager,
        )

        # Mock the Anthropic client
        self.mock_client = Mock()
        self.ai_generator_v2.client = self.mock_client
        self.ai_generator_v2.reasoning_engine.client = self.mock_client

    def test_backward_compatibility_simple_query(self):
        """Test that V2 maintains backward compatibility for simple queries"""
        # Mock response from Anthropic
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a simple answer")]
        mock_response.stop_reason = "end_turn"
        self.mock_client.messages.create.return_value = mock_response

        # Should work the same as V1 for simple queries
        result = self.ai_generator_v2.generate_response("What is 2+2?")

        assert result == "This is a simple answer"
        self.mock_client.messages.create.assert_called_once()

    @patch("asyncio.run")
    def test_v2_multi_round_capability(self, mock_asyncio_run):
        """Test that V2 uses multi-round reasoning when tools are available"""
        # Configure mock session result
        mock_session = Mock()
        mock_session.termination_reason = Mock()
        mock_asyncio_run.return_value = mock_session

        # Configure mock response assembly
        self.ai_generator_v2.response_assembler.assemble_final_response.return_value = (
            "Multi-round response with sources",
            [{"text": "Source 1", "url": None}],
        )

        tools = [{"name": "search_course_content"}]

        result = self.ai_generator_v2.generate_response(
            query="What is Python programming?",
            tools=tools,
            tool_manager=self.mock_tool_manager,
        )

        # Should use multi-round reasoning
        mock_asyncio_run.assert_called_once()
        assert result == "Multi-round response with sources"

        # Should update tool manager with sources
        self.mock_tool_manager.reset_sources.assert_called()

    def test_v2_configuration_methods(self):
        """Test V2-specific configuration and metrics methods"""
        # Test configuration access
        config = self.ai_generator_v2.get_config()
        assert config is not None
        assert hasattr(config, "max_rounds")

        # Test configuration update
        original_max_rounds = config.max_rounds
        self.ai_generator_v2.update_config(max_rounds=5)
        assert self.ai_generator_v2.config.max_rounds == 5

        # Test metrics methods
        session_metrics = self.ai_generator_v2.get_session_metrics()
        assert isinstance(session_metrics, dict)

        tool_metrics = self.ai_generator_v2.get_tool_metrics()
        assert isinstance(tool_metrics, dict)

    def test_v2_system_prompt_updated(self):
        """Test that V2 has updated system prompt for multi-round reasoning"""
        system_prompt = self.ai_generator_v2.SYSTEM_PROMPT

        # V2 should have multi-round capabilities
        assert "multi-round reasoning" in system_prompt.lower()
        assert "progressive refinement" in system_prompt.lower()

        # Should NOT have the V1 limitation
        assert "One tool call per query maximum" not in system_prompt

    def test_v2_fallback_on_error(self):
        """Test that V2 falls back to simple response on multi-round error"""
        # Configure to simulate multi-round failure
        with patch("asyncio.run", side_effect=Exception("Multi-round failed")):
            # Mock simple response fallback
            mock_response = Mock()
            mock_response.content = [Mock(text="Fallback response")]
            self.mock_client.messages.create.return_value = mock_response

            tools = [{"name": "search_course_content"}]

            result = self.ai_generator_v2.generate_response(
                query="Test query", tools=tools, tool_manager=self.mock_tool_manager
            )

            # Should fall back to simple response
            assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_v2_async_interface(self):
        """Test V2's async interface"""
        # Configure mock for async method
        with patch.object(
            self.ai_generator_v2.reasoning_coordinator, "process_query"
        ) as mock_process:
            mock_session = Mock()
            mock_process.return_value = mock_session

            # Configure response assembler
            with patch.object(
                self.ai_generator_v2.response_assembler, "assemble_final_response"
            ) as mock_assemble:
                mock_assemble.return_value = ("Async response", [])

                response, sources = await self.ai_generator_v2.generate_response_async(
                    query="Async test query",
                    tools=[{"name": "search_course_content"}],
                    tool_manager=self.mock_tool_manager,
                )

                assert response == "Async response"
                assert isinstance(sources, list)
                mock_process.assert_called_once()


class TestMigrationHelper:
    """Helper tests for migrating from V1 to V2"""

    def test_interface_compatibility(self):
        """Test that V2 maintains the same public interface as V1"""
        # Both should have the same core methods
        v1_methods = {
            method for method in dir(AIGenerator) if not method.startswith("_")
        }

        if HAS_V2:
            v2_methods = {
                method for method in dir(AIGeneratorV2) if not method.startswith("_")
            }

            # Core interface methods should be present in V2
            core_methods = {"generate_response", "SYSTEM_PROMPT"}
            assert core_methods.issubset(v1_methods)
            assert core_methods.issubset(v2_methods)

    @pytest.mark.skipif(not HAS_V2, reason="AIGeneratorV2 not available")
    def test_drop_in_replacement(self):
        """Test that V2 can be used as a drop-in replacement for V1"""
        # Create both generators with the same parameters
        v1_generator = AIGenerator("test-key", "test-model")
        v2_generator = AIGeneratorV2("test-key", "test-model")

        # Both should have compatible constructors
        assert v1_generator.model == v2_generator.model
        assert hasattr(v1_generator, "SYSTEM_PROMPT")
        assert hasattr(v2_generator, "SYSTEM_PROMPT")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
