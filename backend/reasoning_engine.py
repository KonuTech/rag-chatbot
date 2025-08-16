"""
Reasoning Engine - Handles individual Claude API calls for multi-round reasoning.
Isolated component that manages single reasoning rounds with tools.
"""

import time
from typing import Any, Dict, List

import anthropic
from component_interfaces import (
    APIError,
    IReasoningEngine,
    ReasoningConfig,
    ReasoningRound,
)


class ReasoningEngine(IReasoningEngine):
    """
    Handles individual Claude API calls within the multi-round reasoning system.

    This component is responsible for:
    - Making single API calls to Claude with appropriate context
    - Managing the updated system prompt for multi-round reasoning
    - Token usage estimation and tracking
    - Basic API error handling and retries
    """

    # Updated system prompt for multi-round reasoning
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Course Content Search**: Use `search_course_content` for questions about specific course content or detailed educational materials
- **Course Outline**: Use `get_course_outline` for questions about course structure, lesson lists, course overview, or table of contents
- **Multi-round reasoning**: You can make multiple tool calls across rounds to gather comprehensive information
- **Progressive refinement**: If you need additional information after seeing tool results, you can make another tool call in the next round
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course content questions**: Use search tool first, then answer
- **Course outline/structure questions**: Use outline tool first, then answer
- **Complex queries**: Break down into multiple tool calls across rounds if needed
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"
 - Do not explain your reasoning process across rounds

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, config: ReasoningConfig):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.config = config

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": config.max_tokens_per_round,
        }

        # Token estimation (rough approximation)
        self.avg_tokens_per_char = 0.25

        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0

    async def execute_reasoning_round(
        self,
        query: str,
        context_briefing: str,
        tools: List[Dict[str, Any]],
        round_number: int,
    ) -> ReasoningRound:
        """
        Execute a single reasoning round with Claude.

        Args:
            query: Original user query
            context_briefing: Synthesized context from previous rounds
            tools: Available tools for this round
            round_number: Current round number (0-based)

        Returns:
            Complete reasoning round data
        """
        round_start = time.time()

        try:
            # Build system content with context
            system_content = self._build_system_content(context_briefing)

            # Build user message
            user_message = self._build_user_message(
                query, round_number, context_briefing
            )

            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": [{"role": "user", "content": user_message}],
                "system": system_content,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call with retries
            response = await self._make_api_call_with_retries(api_params)

            # Extract final text response if no tools used
            final_text = None
            if response.stop_reason != "tool_use":
                final_text = response.content[0].text if response.content else ""

            # Calculate token usage (approximate)
            token_usage = self._calculate_token_usage(api_params, response)

            # Create reasoning round result
            reasoning_round = ReasoningRound(
                round_number=round_number,
                user_query=query,
                ai_response_content=response.content,
                tool_executions=[],  # Will be filled by coordinator
                final_text=final_text,
                round_duration=time.time() - round_start,
                token_usage=token_usage,
            )

            return reasoning_round

        except Exception as e:
            # Create error round
            error_round = ReasoningRound(
                round_number=round_number,
                user_query=query,
                ai_response_content=[],
                tool_executions=[],
                final_text=f"API Error: {str(e)}",
                round_duration=time.time() - round_start,
                token_usage={"total": 0, "input": 0, "output": 0},
            )
            return error_round

    def estimate_token_usage(self, context_briefing: str, query: str) -> int:
        """
        Estimate tokens that would be used for a reasoning round.

        This is a rough approximation used for context overflow prevention.
        """
        system_tokens = len(self.SYSTEM_PROMPT) * self.avg_tokens_per_char
        context_tokens = len(context_briefing) * self.avg_tokens_per_char
        query_tokens = len(query) * self.avg_tokens_per_char

        # Add buffer for tools and response
        buffer_tokens = 200

        return int(system_tokens + context_tokens + query_tokens + buffer_tokens)

    def _build_system_content(self, context_briefing: str) -> str:
        """Build system prompt with context briefing"""
        if context_briefing:
            return f"{self.SYSTEM_PROMPT}\n\nContext from previous reasoning:\n{context_briefing}"
        else:
            return self.SYSTEM_PROMPT

    def _build_user_message(
        self, query: str, round_number: int, context_briefing: str
    ) -> str:
        """Build user message for the current round"""
        if round_number == 0:
            # First round - just the query
            return f"Answer this question about course materials: {query}"
        else:
            # Subsequent rounds - reference the original query and note this is a continuation
            return f"Continue working on this question: {query}"

    async def _make_api_call_with_retries(self, api_params: Dict[str, Any]):
        """Make API call with exponential backoff retries"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Make the API call
                response = self.client.messages.create(**api_params)
                return response

            except anthropic.APIError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Wait before retrying with exponential backoff
                    wait_time = self.retry_delay * (2**attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    break
            except Exception as e:
                # Non-anthropic errors - don't retry
                raise APIError(f"API call failed: {str(e)}") from e

        # All retries failed
        raise APIError(
            f"API call failed after {self.max_retries} attempts: {str(last_exception)}"
        ) from last_exception

    def _calculate_token_usage(
        self, api_params: Dict[str, Any], response
    ) -> Dict[str, int]:
        """
        Calculate approximate token usage for the API call.

        Since we don't have access to exact token counts from Anthropic,
        this provides a rough estimate for tracking purposes.
        """
        # Estimate input tokens
        system_tokens = len(api_params.get("system", "")) * self.avg_tokens_per_char

        message_tokens = 0
        for message in api_params.get("messages", []):
            message_tokens += (
                len(str(message.get("content", ""))) * self.avg_tokens_per_char
            )

        tools_tokens = 0
        if "tools" in api_params:
            tools_tokens = len(str(api_params["tools"])) * self.avg_tokens_per_char

        input_tokens = int(system_tokens + message_tokens + tools_tokens)

        # Estimate output tokens
        output_tokens = 0
        if response.content:
            for content_block in response.content:
                if hasattr(content_block, "text"):
                    output_tokens += len(content_block.text) * self.avg_tokens_per_char
                elif hasattr(content_block, "input"):
                    output_tokens += (
                        len(str(content_block.input)) * self.avg_tokens_per_char
                    )

        output_tokens = int(output_tokens)

        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens,
        }

    def get_system_prompt(self) -> str:
        """Get the current system prompt"""
        return self.SYSTEM_PROMPT

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt (for testing or configuration)"""
        self.SYSTEM_PROMPT = new_prompt
