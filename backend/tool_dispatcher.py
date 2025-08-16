"""
Tool Dispatcher - Handles async tool execution with fallback strategies.
Manages tool execution, error recovery, and alternative search strategies.
"""

import asyncio
import time
from typing import Any, Dict, List

from component_interfaces import (
    IToolDispatcher,
    ReasoningConfig,
    ToolExecutionError,
    ToolExecutionResult,
)


class ToolDispatcher(IToolDispatcher):
    """
    Manages async tool execution with sophisticated error recovery.

    Features:
    - Async execution of multiple tools
    - Tool substitution when primary tools fail
    - Retry logic for transient failures
    - Graceful degradation strategies
    - Performance monitoring
    """

    def __init__(self, tool_manager, config: ReasoningConfig):
        self.tool_manager = tool_manager  # Existing ToolManager from search_tools.py
        self.config = config

        # Tool fallback mappings
        self.tool_fallbacks = {
            "search_course_content": ["get_course_outline"],
            "get_course_outline": ["search_course_content"],
        }

        # Retry configuration
        self.max_retries = 2
        self.retry_delay = 0.5

        # Performance tracking
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "fallback_used": 0,
            "retry_attempts": 0,
        }

    async def execute_tools(
        self, tool_calls: List[Dict[str, Any]], session_id: str, round_number: int
    ) -> List[ToolExecutionResult]:
        """
        Execute multiple tool calls and return results.

        Supports:
        - Parallel execution of independent tools
        - Sequential execution when tools depend on each other
        - Automatic fallback to alternative tools
        - Retry logic for transient failures
        """
        if not tool_calls:
            return []

        results = []

        # Execute tools sequentially for now (could be parallelized if independent)
        for tool_call in tool_calls:
            tool_result = await self._execute_single_tool(
                tool_call, session_id, round_number
            )
            results.append(tool_result)

        return results

    async def _execute_single_tool(
        self, tool_call: Dict[str, Any], session_id: str, round_number: int
    ) -> ToolExecutionResult:
        """Execute a single tool with retries and fallbacks"""
        tool_name = tool_call.get("name", "unknown")
        tool_input = tool_call.get("input", {})
        tool_id = tool_call.get("id", "")

        start_time = time.time()

        # Track metrics
        self.execution_metrics["total_executions"] += 1

        # Try primary tool with retries
        result = await self._try_tool_execution(tool_name, tool_input, tool_id)

        # If primary tool failed and fallbacks are enabled, try alternatives
        if not result.success and self.config.enable_tool_fallbacks:
            fallback_tools = self.get_fallback_tools(tool_name)

            for fallback_tool in fallback_tools:
                # Adapt input for fallback tool if needed
                adapted_input = self._adapt_input_for_tool(fallback_tool, tool_input)

                fallback_result = await self._try_tool_execution(
                    fallback_tool, adapted_input, tool_id
                )

                if fallback_result.success:
                    # Use fallback result but note the original tool name
                    result = ToolExecutionResult(
                        tool_name=f"{tool_name} (via {fallback_tool})",
                        tool_input=tool_input,
                        success=True,
                        result=fallback_result.result,
                        execution_time=time.time() - start_time,
                        error=None,
                    )
                    self.execution_metrics["fallback_used"] += 1
                    break

        # Update metrics
        if result.success:
            self.execution_metrics["successful_executions"] += 1
        else:
            self.execution_metrics["failed_executions"] += 1

        return result

    async def _try_tool_execution(
        self, tool_name: str, tool_input: Dict[str, Any], tool_id: str
    ) -> ToolExecutionResult:
        """Try executing a tool with retries"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                # Execute the tool
                result = self.tool_manager.execute_tool(tool_name, **tool_input)

                execution_time = time.time() - start_time

                # Check if result indicates success
                if (
                    result
                    and not result.startswith("Error")
                    and not result.startswith("No")
                ):
                    return ToolExecutionResult(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        success=True,
                        result=result,
                        execution_time=execution_time,
                        error=None,
                    )
                else:
                    # Tool executed but returned no results or error
                    return ToolExecutionResult(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        success=False,
                        result=result or "No results found",
                        execution_time=execution_time,
                        error=None,
                    )

            except Exception as e:
                last_error = e
                execution_time = time.time() - start_time

                # Check if this is retryable
                if attempt < self.max_retries and self.can_retry_tool(tool_name, e):
                    self.execution_metrics["retry_attempts"] += 1
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    # Max retries reached or non-retryable error
                    break

        # All attempts failed
        return ToolExecutionResult(
            tool_name=tool_name,
            tool_input=tool_input,
            success=False,
            result=f"Tool execution failed: {str(last_error)}",
            execution_time=execution_time,
            error=last_error,
        )

    def get_fallback_tools(self, failed_tool: str) -> List[str]:
        """Get alternative tools when primary tool fails"""
        return self.tool_fallbacks.get(failed_tool, [])

    def can_retry_tool(self, tool_name: str, error: Exception) -> bool:
        """Determine if a failed tool execution can be retried"""
        # Retry for transient errors
        error_str = str(error).lower()

        retryable_errors = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "rate limit",
        ]

        for retryable in retryable_errors:
            if retryable in error_str:
                return True

        # Don't retry for logic errors or missing data
        non_retryable_errors = ["not found", "invalid", "permission", "authentication"]

        for non_retryable in non_retryable_errors:
            if non_retryable in error_str:
                return False

        # Default to retry for unknown errors
        return True

    def _adapt_input_for_tool(
        self, tool_name: str, original_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt input parameters when using fallback tools.

        For example, if search_course_content fails, we might use get_course_outline
        with just the course_name parameter.
        """
        adapted_input = original_input.copy()

        if tool_name == "get_course_outline":
            # For outline tool, we only need course_name
            if "course_name" in adapted_input:
                return {"course_name": adapted_input["course_name"]}
            else:
                # Try to extract course name from query
                query = adapted_input.get("query", "")
                course_name = self._extract_course_name_from_query(query)
                if course_name:
                    return {"course_name": course_name}

        elif tool_name == "search_course_content":
            # For search tool, convert outline request to a search
            if "course_name" in adapted_input and "query" not in adapted_input:
                # Convert outline request to search for course overview
                return {
                    "query": f"overview of {adapted_input['course_name']}",
                    "course_name": adapted_input["course_name"],
                }

        return adapted_input

    def _extract_course_name_from_query(self, query: str) -> str:
        """
        Extract potential course name from a search query.

        This is a simple heuristic that looks for capitalized words
        that might be course names.
        """
        if not query:
            return ""

        # Look for patterns like "Course X", "Introduction to Y", etc.
        words = query.split()

        # Find sequences of capitalized words
        course_candidates = []
        current_candidate = []

        for word in words:
            if word and word[0].isupper():
                current_candidate.append(word)
            else:
                if current_candidate:
                    course_candidates.append(" ".join(current_candidate))
                    current_candidate = []

        # Add final candidate if exists
        if current_candidate:
            course_candidates.append(" ".join(current_candidate))

        # Return the longest candidate (most likely to be a course name)
        if course_candidates:
            return max(course_candidates, key=len)

        return ""

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for tool execution"""
        total = self.execution_metrics["total_executions"]

        if total == 0:
            return self.execution_metrics

        return {
            **self.execution_metrics,
            "success_rate": self.execution_metrics["successful_executions"] / total,
            "failure_rate": self.execution_metrics["failed_executions"] / total,
            "fallback_rate": self.execution_metrics["fallback_used"] / total,
            "retry_rate": self.execution_metrics["retry_attempts"] / total,
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "fallback_used": 0,
            "retry_attempts": 0,
        }
