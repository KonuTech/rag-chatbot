"""
AI Generator V2 - Refactored to use event-driven multi-round orchestration.
This replaces the original ai_generator.py with the new architecture.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from component_interfaces import ReasoningConfig, TerminationReason
from context_synthesizer import ContextSynthesizer
from reasoning_coordinator import ReasoningCoordinator
from reasoning_engine import ReasoningEngine
from response_assembler import ResponseAssembler
from tool_dispatcher import ToolDispatcher


class AIGeneratorV2:
    """
    Refactored AI Generator using event-driven multi-round orchestration.

    This maintains the same external interface as the original AIGenerator
    but uses the new component architecture internally to support
    sequential tool calling across multiple rounds.
    """

    def __init__(self, api_key: str, model: str, tool_manager=None):
        # Create configuration
        self.config = ReasoningConfig()

        # Initialize components
        self.reasoning_engine = ReasoningEngine(api_key, model, self.config)
        self.context_synthesizer = ContextSynthesizer(self.config)
        self.tool_dispatcher = (
            ToolDispatcher(tool_manager, self.config) if tool_manager else None
        )
        self.response_assembler = ResponseAssembler(self.config)

        # Initialize coordinator
        if self.tool_dispatcher:
            self.reasoning_coordinator = ReasoningCoordinator(
                reasoning_engine=self.reasoning_engine,
                context_synthesizer=self.context_synthesizer,
                tool_dispatcher=self.tool_dispatcher,
                response_assembler=self.response_assembler,
                config=self.config,
            )
        else:
            self.reasoning_coordinator = None

        # Store for backward compatibility
        self.api_key = api_key
        self.model = model
        self.client = self.reasoning_engine.client

        # Legacy system prompt access
        self.SYSTEM_PROMPT = self.reasoning_engine.SYSTEM_PROMPT

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        This maintains the same interface as the original generate_response method
        but uses the new multi-round orchestration internally.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context (legacy - ignored in v2)
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # If no tools provided, fall back to simple single-round response
        if not tools or not tool_manager:
            return self._generate_simple_response(query, conversation_history)

        # Use multi-round orchestration for tool-based queries
        try:
            # Update tool dispatcher if needed
            if tool_manager and tool_manager != self.tool_dispatcher.tool_manager:
                self.tool_dispatcher = ToolDispatcher(tool_manager, self.config)
                self.reasoning_coordinator = ReasoningCoordinator(
                    reasoning_engine=self.reasoning_engine,
                    context_synthesizer=self.context_synthesizer,
                    tool_dispatcher=self.tool_dispatcher,
                    response_assembler=self.response_assembler,
                    config=self.config,
                )

            # Process query through multi-round reasoning
            session = asyncio.run(self.reasoning_coordinator.process_query(query))

            # Assemble final response
            response, sources = self.response_assembler.assemble_final_response(session)

            # Store sources for retrieval by the RAG system
            self._store_sources_for_retrieval(sources, tool_manager)

            return response

        except Exception as e:
            # Fallback to simple response on error
            print(f"Multi-round reasoning failed: {e}")
            return self._generate_simple_response(query, conversation_history)

    def _generate_simple_response(
        self, query: str, conversation_history: Optional[str] = None
    ) -> str:
        """
        Generate a simple single-round response (legacy behavior).

        Used when no tools are available or multi-round reasoning fails.
        """
        # Build system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        try:
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            return (
                response.content[0].text
                if response.content
                else "I couldn't generate a response."
            )

        except Exception as e:
            return f"I'm experiencing technical difficulties: {str(e)}"

    def _store_sources_for_retrieval(
        self, sources: List[Dict[str, Any]], tool_manager
    ) -> None:
        """
        Store sources in the tool manager for retrieval by the RAG system.

        This maintains compatibility with the existing source tracking mechanism.
        """
        if hasattr(tool_manager, "reset_sources"):
            tool_manager.reset_sources()

        # Find the search tool and update its sources
        for tool in getattr(tool_manager, "tools", {}).values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = sources
                break

    # Legacy method compatibility
    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Legacy method for backward compatibility.

        This method is no longer used in the new architecture but is kept
        to avoid breaking existing code that might reference it.
        """
        raise NotImplementedError(
            "This method is deprecated in AIGeneratorV2. "
            "Tool execution is now handled by the multi-round orchestration system."
        )

    # Configuration and debugging methods
    def get_config(self) -> ReasoningConfig:
        """Get the current reasoning configuration"""
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update reasoning configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def get_session_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the reasoning system"""
        if self.reasoning_coordinator:
            return self.reasoning_coordinator.get_session_metrics()
        return {}

    def get_tool_metrics(self) -> Dict[str, Any]:
        """Get tool execution metrics"""
        if self.tool_dispatcher:
            return self.tool_dispatcher.get_execution_metrics()
        return {}

    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        if self.tool_dispatcher:
            self.tool_dispatcher.reset_metrics()

    # Async interface for advanced usage
    async def generate_response_async(
        self,
        query: str,
        session_id: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Async version that returns both response and sources.

        This provides access to the full capabilities of the new architecture
        including session management and detailed source information.
        """
        if not tools or not tool_manager or not self.reasoning_coordinator:
            response = self._generate_simple_response(query)
            return response, []

        # Update tool dispatcher if needed
        if tool_manager != self.tool_dispatcher.tool_manager:
            self.tool_dispatcher = ToolDispatcher(tool_manager, self.config)
            self.reasoning_coordinator = ReasoningCoordinator(
                reasoning_engine=self.reasoning_engine,
                context_synthesizer=self.context_synthesizer,
                tool_dispatcher=self.tool_dispatcher,
                response_assembler=self.response_assembler,
                config=self.config,
            )

        # Process query
        session = await self.reasoning_coordinator.process_query(query, session_id)

        # Assemble response
        response, sources = self.response_assembler.assemble_final_response(session)

        return response, sources

    def get_session(self, session_id: str):
        """Get a reasoning session by ID"""
        if self.reasoning_coordinator:
            return self.reasoning_coordinator.get_session(session_id)
        return None

    def terminate_session(
        self,
        session_id: str,
        reason: TerminationReason = TerminationReason.USER_CANCELLATION,
    ):
        """Terminate an active reasoning session"""
        if self.reasoning_coordinator:
            self.reasoning_coordinator.terminate_session(session_id, reason)
