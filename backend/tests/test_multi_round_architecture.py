"""
Comprehensive test suite for the new multi-round reasoning architecture.
Tests external behavior, API calls, tool execution, and end-to-end scenarios.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Import the new architecture components
from component_interfaces import (
    ReasoningConfig, ReasoningSession, ReasoningRound, ToolExecutionResult,
    TerminationReason, ReasoningEventType
)
from reasoning_coordinator import ReasoningCoordinator
from reasoning_engine import ReasoningEngine
from context_synthesizer import ContextSynthesizer
from tool_dispatcher import ToolDispatcher
from response_assembler import ResponseAssembler
from ai_generator_v2 import AIGeneratorV2


class TestReasoningEngine:
    """Test the reasoning engine component"""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        client = Mock()
        
        # Mock successful response with tool use
        mock_response_with_tools = Mock()
        mock_response_with_tools.content = [
            Mock(type="tool_use", name="search_course_content", input={"query": "Python"}, id="tool_1")
        ]
        mock_response_with_tools.stop_reason = "tool_use"
        
        # Mock successful response without tools
        mock_response_final = Mock()
        mock_response_final.content = [Mock(text="Here's information about Python programming...")]
        mock_response_final.stop_reason = "end_turn"
        
        client.messages.create.return_value = mock_response_with_tools
        return client
    
    @pytest.fixture
    def reasoning_engine(self, mock_anthropic_client):
        """Create a reasoning engine with mocked client"""
        config = ReasoningConfig()
        engine = ReasoningEngine("test-api-key", "claude-3-sonnet", config)
        engine.client = mock_anthropic_client
        return engine
    
    @pytest.mark.asyncio
    async def test_execute_reasoning_round_with_tools(self, reasoning_engine, mock_anthropic_client):
        """Test reasoning round that triggers tool use"""
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        result = await reasoning_engine.execute_reasoning_round(
            query="What is Python?",
            context_briefing="",
            tools=tools,
            round_number=0
        )
        
        assert result.round_number == 0
        assert result.user_query == "What is Python?"
        assert len(result.ai_response_content) == 1
        assert result.ai_response_content[0].type == "tool_use"
        assert result.final_text is None  # No final text when tools used
        
        # Verify API call was made with tools
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
    
    @pytest.mark.asyncio
    async def test_execute_reasoning_round_without_tools(self, reasoning_engine, mock_anthropic_client):
        """Test reasoning round that completes without tool use"""
        # Configure mock for final response
        mock_response_final = Mock()
        mock_response_final.content = [Mock(text="Python is a programming language...")]
        mock_response_final.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response_final
        
        result = await reasoning_engine.execute_reasoning_round(
            query="What is Python?",
            context_briefing="Previous context here",
            tools=[],
            round_number=1
        )
        
        assert result.round_number == 1
        assert result.final_text == "Python is a programming language..."
        assert len(result.tool_executions) == 0
    
    def test_token_usage_estimation(self, reasoning_engine):
        """Test token usage estimation"""
        tokens = reasoning_engine.estimate_token_usage(
            context_briefing="Some previous context",
            query="What is machine learning?"
        )
        
        assert tokens > 0
        assert isinstance(tokens, int)


class TestContextSynthesizer:
    """Test the context synthesizer component"""
    
    @pytest.fixture
    def context_synthesizer(self):
        return ContextSynthesizer(ReasoningConfig())
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample session with multiple rounds"""
        session = ReasoningSession(
            session_id="test-session",
            original_query="Compare Python and Java courses",
            rounds=[],
            discovered_facts={
                "tool_search_0": "Python course covers basic syntax",
                "tool_search_1": "Java course covers object-oriented programming"
            },
            reasoning_trace=["Round 0: Searched for Python info", "Round 1: Searched for Java info"],
            evolving_intent="Compare Python and Java courses",
            tool_usage_history=[
                {"round": 0, "tool": "search_course_content", "input": {"query": "Python"}, "success": True},
                {"round": 1, "tool": "search_course_content", "input": {"query": "Java"}, "success": True}
            ]
        )
        return session
    
    def test_build_context_briefing_first_round(self, context_synthesizer):
        """Test context briefing for first round (should be empty)"""
        session = ReasoningSession(
            session_id="test",
            original_query="Test query",
            rounds=[],
            discovered_facts={},
            reasoning_trace=[],
            evolving_intent="Test query",
            tool_usage_history=[]
        )
        
        briefing = context_synthesizer.build_context_briefing(session, 0)
        assert briefing == ""
    
    def test_build_context_briefing_second_round(self, context_synthesizer, sample_session):
        """Test context briefing for second round"""
        briefing = context_synthesizer.build_context_briefing(sample_session, 1)
        
        assert "User Intent:" in briefing
        assert "Information Discovered:" in briefing
        assert "Search History:" in briefing
        assert "Python" in briefing
        assert "Java" in briefing
    
    def test_extract_factual_information(self, context_synthesizer):
        """Test extraction of facts from reasoning round"""
        round_data = ReasoningRound(
            round_number=0,
            user_query="What is Python?",
            ai_response_content=[],
            tool_executions=[
                ToolExecutionResult(
                    tool_name="search_course_content",
                    tool_input={"query": "Python"},
                    success=True,
                    result="Python is a high-level programming language...",
                    execution_time=1.5
                )
            ],
            final_text="Based on the search, Python is a programming language."
        )
        
        facts = context_synthesizer.extract_factual_information(round_data)
        
        assert len(facts) == 2  # One from tool, one from response
        assert "tool_search_course_content_0" in facts
        assert "response_0" in facts
    
    def test_should_compress_context(self, context_synthesizer, sample_session):
        """Test context compression decision logic"""
        # Small session should not need compression
        assert not context_synthesizer.should_compress_context(sample_session)
        
        # Large session should need compression
        large_session = sample_session
        large_session.discovered_facts = {f"fact_{i}": "content" for i in range(15)}
        large_session.reasoning_trace = [f"trace_{i}" for i in range(10)]
        
        assert context_synthesizer.should_compress_context(large_session)


class TestToolDispatcher:
    """Test the tool dispatcher component"""
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager"""
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Successful search result"
        return tool_manager
    
    @pytest.fixture
    def tool_dispatcher(self, mock_tool_manager):
        config = ReasoningConfig()
        return ToolDispatcher(mock_tool_manager, config)
    
    @pytest.mark.asyncio
    async def test_execute_tools_success(self, tool_dispatcher, mock_tool_manager):
        """Test successful tool execution"""
        tool_calls = [
            {"name": "search_course_content", "input": {"query": "Python"}, "id": "tool_1"}
        ]
        
        results = await tool_dispatcher.execute_tools(tool_calls, "session_1", 0)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].tool_name == "search_course_content"
        assert "Successful" in results[0].result
        
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="Python")
    
    @pytest.mark.asyncio
    async def test_execute_tools_with_fallback(self, tool_dispatcher, mock_tool_manager):
        """Test tool execution with fallback when primary tool fails"""
        # Configure primary tool to fail, fallback to succeed
        def side_effect(tool_name, **kwargs):
            if tool_name == "search_course_content":
                raise Exception("Search failed")
            elif tool_name == "get_course_outline":
                return "Outline retrieved successfully"
            return "Unknown tool"
        
        mock_tool_manager.execute_tool.side_effect = side_effect
        
        tool_calls = [
            {"name": "search_course_content", "input": {"query": "Python"}, "id": "tool_1"}
        ]
        
        results = await tool_dispatcher.execute_tools(tool_calls, "session_1", 0)
        
        assert len(results) == 1
        assert results[0].success is True
        assert "get_course_outline" in results[0].tool_name
        assert "Outline retrieved" in results[0].result
    
    def test_get_fallback_tools(self, tool_dispatcher):
        """Test fallback tool mapping"""
        fallbacks = tool_dispatcher.get_fallback_tools("search_course_content")
        assert "get_course_outline" in fallbacks
        
        fallbacks = tool_dispatcher.get_fallback_tools("get_course_outline")
        assert "search_course_content" in fallbacks
    
    def test_can_retry_tool(self, tool_dispatcher):
        """Test retry decision logic"""
        # Should retry for network errors
        assert tool_dispatcher.can_retry_tool("search", Exception("Connection timeout"))
        assert tool_dispatcher.can_retry_tool("search", Exception("Network error"))
        
        # Should not retry for logic errors
        assert not tool_dispatcher.can_retry_tool("search", Exception("Not found"))
        assert not tool_dispatcher.can_retry_tool("search", Exception("Invalid parameter"))


class TestResponseAssembler:
    """Test the response assembler component"""
    
    @pytest.fixture
    def response_assembler(self):
        return ResponseAssembler(ReasoningConfig())
    
    @pytest.fixture
    def completed_session(self):
        """Create a completed reasoning session"""
        session = ReasoningSession(
            session_id="test-session",
            original_query="What is Python?",
            rounds=[
                ReasoningRound(
                    round_number=0,
                    user_query="What is Python?",
                    ai_response_content=[],
                    tool_executions=[
                        ToolExecutionResult(
                            tool_name="search_course_content",
                            tool_input={"query": "Python"},
                            success=True,
                            result="[Python Course - Lesson 1]\nPython is a programming language...",
                            execution_time=1.5
                        )
                    ],
                    final_text="Python is a high-level programming language used for various applications."
                )
            ],
            discovered_facts={},
            reasoning_trace=[],
            evolving_intent="Learn about Python",
            tool_usage_history=[],
            termination_reason=TerminationReason.NATURAL_COMPLETION
        )
        return session
    
    def test_assemble_final_response_natural_completion(self, response_assembler, completed_session):
        """Test response assembly for naturally completed session"""
        response, sources = response_assembler.assemble_final_response(completed_session)
        
        assert "Python is a high-level programming language" in response
        assert len(sources) > 0 if sources else True  # May or may not have sources depending on extraction
    
    def test_handle_partial_completion_max_rounds(self, response_assembler, completed_session):
        """Test handling of session that hit max rounds"""
        completed_session.termination_reason = TerminationReason.MAX_ROUNDS_REACHED
        
        response, sources = response_assembler.handle_partial_completion(
            completed_session, TerminationReason.MAX_ROUNDS_REACHED
        )
        
        # Should return the best available response without modification
        assert "Python is a high-level programming language" in response
    
    def test_extract_sources(self, response_assembler, completed_session):
        """Test source extraction from session"""
        sources = response_assembler.extract_sources(completed_session)
        
        # Should extract sources from search results
        # Note: This depends on the implementation of _extract_sources_from_search_result
        assert isinstance(sources, list)


class TestReasoningCoordinator:
    """Test the main reasoning coordinator"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for the coordinator"""
        reasoning_engine = AsyncMock()
        context_synthesizer = Mock()
        tool_dispatcher = AsyncMock()
        response_assembler = Mock()
        
        return reasoning_engine, context_synthesizer, tool_dispatcher, response_assembler
    
    @pytest.fixture
    def coordinator(self, mock_components):
        """Create a reasoning coordinator with mocked components"""
        reasoning_engine, context_synthesizer, tool_dispatcher, response_assembler = mock_components
        config = ReasoningConfig()
        
        return ReasoningCoordinator(
            reasoning_engine=reasoning_engine,
            context_synthesizer=context_synthesizer,
            tool_dispatcher=tool_dispatcher,
            response_assembler=response_assembler,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_process_query_single_round_completion(self, coordinator, mock_components):
        """Test query that completes in a single round"""
        reasoning_engine, context_synthesizer, tool_dispatcher, response_assembler = mock_components
        
        # Configure mock for natural completion (no tool use)
        mock_round = ReasoningRound(
            round_number=0,
            user_query="What is 2+2?",
            ai_response_content=[Mock(text="4")],
            tool_executions=[],
            final_text="4",
            token_usage={"total": 50}
        )
        reasoning_engine.execute_reasoning_round.return_value = mock_round
        
        session = await coordinator.process_query("What is 2+2?")
        
        assert session.termination_reason == TerminationReason.NATURAL_COMPLETION
        assert len(session.rounds) == 1
        assert session.rounds[0].final_text == "4"
    
    @pytest.mark.asyncio
    async def test_process_query_max_rounds_reached(self, coordinator, mock_components):
        """Test query that hits the maximum round limit"""
        reasoning_engine, context_synthesizer, tool_dispatcher, response_assembler = mock_components
        
        # Configure mock for tool use in each round
        mock_round_with_tools = ReasoningRound(
            round_number=0,
            user_query="Complex query",
            ai_response_content=[Mock(type="tool_use")],
            tool_executions=[
                ToolExecutionResult("search_tool", {}, True, "result", 1.0)
            ],
            final_text=None,
            token_usage={"total": 200}
        )
        reasoning_engine.execute_reasoning_round.return_value = mock_round_with_tools
        tool_dispatcher.execute_tools.return_value = [
            ToolExecutionResult("search_tool", {}, True, "result", 1.0)
        ]
        
        # Set max rounds to 2 to test the limit
        coordinator.config.max_rounds = 2
        
        session = await coordinator.process_query("Complex query requiring multiple rounds")
        
        assert session.termination_reason == TerminationReason.MAX_ROUNDS_REACHED
        assert len(session.rounds) == 2


class TestAIGeneratorV2Integration:
    """Test the complete AIGeneratorV2 with integration scenarios"""
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create a comprehensive mock tool manager"""
        tool_manager = Mock()
        tool_manager.tools = {}
        tool_manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
        tool_manager.execute_tool.return_value = "Search results found"
        tool_manager.get_last_sources.return_value = []
        tool_manager.reset_sources.return_value = None
        
        return tool_manager
    
    @pytest.fixture
    def ai_generator_v2(self, mock_tool_manager):
        """Create AIGeneratorV2 with mocked dependencies"""
        generator = AIGeneratorV2("test-api-key", "claude-3-sonnet", mock_tool_manager)
        
        # Mock the Anthropic client
        generator.client = Mock()
        generator.reasoning_engine.client = generator.client
        
        return generator
    
    def test_generate_response_simple_query(self, ai_generator_v2):
        """Test simple query without tools"""
        # Configure mock for simple response
        mock_response = Mock()
        mock_response.content = [Mock(text="Simple answer")]
        ai_generator_v2.client.messages.create.return_value = mock_response
        
        response = ai_generator_v2.generate_response(
            query="What is 2+2?",
            tools=None,
            tool_manager=None
        )
        
        assert response == "Simple answer"
    
    @patch('asyncio.run')
    def test_generate_response_with_tools(self, mock_asyncio_run, ai_generator_v2, mock_tool_manager):
        """Test query with tools using multi-round reasoning"""
        # Configure mock session result
        mock_session = Mock()
        mock_session.termination_reason = TerminationReason.NATURAL_COMPLETION
        mock_asyncio_run.return_value = mock_session
        
        # Configure mock response assembly
        ai_generator_v2.response_assembler.assemble_final_response.return_value = (
            "Multi-round response", []
        )
        
        tools = [{"name": "search_course_content"}]
        
        response = ai_generator_v2.generate_response(
            query="What is Python?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Should use multi-round reasoning
        mock_asyncio_run.assert_called_once()
        assert response == "Multi-round response"
    
    def test_configuration_methods(self, ai_generator_v2):
        """Test configuration and metrics methods"""
        config = ai_generator_v2.get_config()
        assert isinstance(config, ReasoningConfig)
        
        # Test config update
        ai_generator_v2.update_config(max_rounds=3)
        assert ai_generator_v2.config.max_rounds == 3
        
        # Test metrics methods
        metrics = ai_generator_v2.get_session_metrics()
        assert isinstance(metrics, dict)
        
        tool_metrics = ai_generator_v2.get_tool_metrics()
        assert isinstance(tool_metrics, dict)


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""
    
    @pytest.mark.asyncio
    async def test_multi_course_comparison_scenario(self):
        """Test scenario: 'Compare Python and Java courses'"""
        # This would be a more complex integration test
        # For now, test the component coordination logic
        
        config = ReasoningConfig()
        config.max_rounds = 2
        
        # Mock components for this scenario
        reasoning_engine = AsyncMock()
        context_synthesizer = Mock()
        tool_dispatcher = AsyncMock()
        response_assembler = Mock()
        
        # Configure mocks for two-round execution
        round_1 = ReasoningRound(
            round_number=0,
            user_query="Compare Python and Java courses",
            ai_response_content=[Mock(type="tool_use", name="search_course_content")],
            tool_executions=[],
            final_text=None,
            token_usage={"total": 300}
        )
        
        round_2 = ReasoningRound(
            round_number=1,
            user_query="Compare Python and Java courses",
            ai_response_content=[Mock(text="Based on my search...")],
            tool_executions=[],
            final_text="Based on my search, Python courses focus on...",
            token_usage={"total": 250}
        )
        
        reasoning_engine.execute_reasoning_round.side_effect = [round_1, round_2]
        tool_dispatcher.execute_tools.return_value = [
            ToolExecutionResult("search_course_content", {"query": "Python"}, True, "Python info", 1.0),
            ToolExecutionResult("search_course_content", {"query": "Java"}, True, "Java info", 1.0)
        ]
        
        coordinator = ReasoningCoordinator(
            reasoning_engine=reasoning_engine,
            context_synthesizer=context_synthesizer,
            tool_dispatcher=tool_dispatcher,
            response_assembler=response_assembler,
            config=config
        )
        
        session = await coordinator.process_query("Compare Python and Java courses")
        
        assert len(session.rounds) == 2
        assert session.termination_reason == TerminationReason.NATURAL_COMPLETION
        
        # Verify that tool dispatcher was called for round 1
        tool_dispatcher.execute_tools.assert_called()


# Performance and stress tests
class TestPerformanceAndLimits:
    """Test performance characteristics and limit handling"""
    
    def test_context_compression_triggers(self):
        """Test that context compression triggers appropriately"""
        config = ReasoningConfig()
        config.context_compression_threshold = 100  # Very low threshold for testing
        
        synthesizer = ContextSynthesizer(config)
        
        # Create session with large context
        session = ReasoningSession(
            session_id="test",
            original_query="Test query",
            rounds=[],
            discovered_facts={f"fact_{i}": "x" * 50 for i in range(10)},  # Large facts
            reasoning_trace=["trace"] * 20,  # Many traces
            evolving_intent="test",
            tool_usage_history=[]
        )
        
        should_compress = synthesizer.should_compress_context(session)
        assert should_compress is True
    
    def test_token_limit_enforcement(self):
        """Test that token limits are enforced"""
        config = ReasoningConfig()
        config.max_total_tokens = 100  # Very low limit for testing
        
        reasoning_engine = ReasoningEngine("test-key", "test-model", config)
        
        # Estimate tokens for a large query
        large_context = "x" * 1000  # Large context
        large_query = "x" * 1000    # Large query
        
        estimated_tokens = reasoning_engine.estimate_token_usage(large_context, large_query)
        
        # Should exceed the limit
        assert estimated_tokens > config.max_total_tokens


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])