"""
Context Synthesizer - Builds multi-round context with semantic compression.
Manages the four-layer context system and intelligent summarization.
"""

import json
import re
from typing import Any, Dict, List

from component_interfaces import (
    IContextSynthesizer,
    ReasoningConfig,
    ReasoningRound,
    ReasoningSession,
)


class ContextSynthesizer(IContextSynthesizer):
    """
    Manages multi-layer context building with semantic compression.

    Implements four context layers:
    1. Factual Layer: Concrete information discovered from tool calls
    2. Reasoning Layer: Claude's intermediate thoughts and reasoning patterns
    3. Intent Layer: Evolving understanding of user goals
    4. Tool Usage Layer: History of what searches were attempted and why
    """

    def __init__(self, config: ReasoningConfig):
        self.config = config

        # Context compression thresholds
        self.max_factual_items = 10
        self.max_reasoning_entries = 5
        self.max_tool_history = 8

    def build_context_briefing(
        self, session: ReasoningSession, round_number: int
    ) -> str:
        """
        Build synthesized context briefing for the next reasoning round.

        Creates a structured briefing that gives Claude context about:
        - What the user is trying to accomplish
        - What information has already been discovered
        - What searches have been attempted
        - Key insights from previous reasoning
        """
        if round_number == 0:
            return ""  # No context for first round

        briefing_parts = []

        # Intent layer - what the user wants
        briefing_parts.append(f"User Intent: {session.evolving_intent}")

        # Factual layer - what we've discovered
        if session.discovered_facts:
            facts_summary = self._summarize_facts(session.discovered_facts)
            briefing_parts.append(f"Information Discovered: {facts_summary}")

        # Tool usage layer - what searches we've tried
        if session.tool_usage_history:
            tools_summary = self._summarize_tool_usage(session.tool_usage_history)
            briefing_parts.append(f"Search History: {tools_summary}")

        # Reasoning layer - key insights
        if session.reasoning_trace:
            reasoning_summary = self._summarize_reasoning(session.reasoning_trace)
            briefing_parts.append(f"Previous Reasoning: {reasoning_summary}")

        # Check if compression is needed
        briefing = "\n\n".join(briefing_parts)
        if self.should_compress_context(session):
            briefing = self._compress_briefing(briefing)

        return briefing

    def extract_factual_information(self, round: ReasoningRound) -> Dict[str, Any]:
        """
        Extract key facts discovered in a reasoning round.

        Looks for concrete information in tool results and Claude's responses.
        """
        facts = {}

        # Extract facts from tool executions
        for tool_exec in round.tool_executions:
            if tool_exec.success and tool_exec.result:
                fact_key = f"tool_{tool_exec.tool_name}_{round.round_number}"
                facts[fact_key] = self._extract_key_info(tool_exec.result)

        # Extract facts from Claude's final text
        if round.final_text:
            fact_key = f"response_{round.round_number}"
            facts[fact_key] = self._extract_key_info(round.final_text)

        return facts

    def update_intent_understanding(
        self, session: ReasoningSession, round: ReasoningRound
    ) -> str:
        """
        Update understanding of user's evolving intent based on new round.

        Analyzes what tools Claude chose and what questions were asked
        to refine understanding of what the user really wants.
        """
        current_intent = session.evolving_intent

        # Analyze tool usage patterns
        tool_patterns = self._analyze_tool_patterns(round.tool_executions)

        # Update intent based on tool patterns
        if "search_course_content" in tool_patterns:
            if "specific content" not in current_intent.lower():
                current_intent += " (seeking specific content)"

        if "get_course_outline" in tool_patterns:
            if "course structure" not in current_intent.lower():
                current_intent += " (exploring course structure)"

        # Analyze query refinement from tool inputs
        query_refinements = self._extract_query_refinements(round.tool_executions)
        if query_refinements:
            current_intent = f"{current_intent} -> {query_refinements}"

        return current_intent

    def should_compress_context(self, session: ReasoningSession) -> bool:
        """
        Determine if context compression is needed.

        Based on:
        - Total length of discovered facts
        - Number of reasoning trace entries
        - Tool usage history length
        """
        if not self.config.enable_context_compression:
            return False

        # Check factual layer size
        if len(session.discovered_facts) > self.max_factual_items:
            return True

        # Check reasoning trace size
        if len(session.reasoning_trace) > self.max_reasoning_entries:
            return True

        # Check tool history size
        if len(session.tool_usage_history) > self.max_tool_history:
            return True

        # Check total briefing length (estimated)
        estimated_length = (
            len(str(session.discovered_facts))
            + len(str(session.reasoning_trace))
            + len(str(session.tool_usage_history))
        )

        return estimated_length > self.config.context_compression_threshold

    def _summarize_facts(self, facts: Dict[str, Any]) -> str:
        """Summarize discovered facts into key points"""
        if not facts:
            return "None"

        # Group facts by type
        tool_facts = []
        response_facts = []

        for key, value in facts.items():
            if key.startswith("tool_"):
                tool_facts.append(value)
            elif key.startswith("response_"):
                response_facts.append(value)

        summary_parts = []

        if tool_facts:
            # Take the most recent and relevant tool facts
            recent_tool_facts = tool_facts[-3:]  # Last 3 tool results
            summary_parts.append(f"Found: {'; '.join(recent_tool_facts)}")

        if response_facts:
            # Take the most recent response insights
            recent_response_facts = response_facts[-2:]  # Last 2 responses
            summary_parts.append(f"Insights: {'; '.join(recent_response_facts)}")

        return ". ".join(summary_parts)

    def _summarize_tool_usage(self, tool_history: List[Dict]) -> str:
        """Summarize tool usage patterns"""
        if not tool_history:
            return "None"

        # Group by tool type
        search_attempts = []
        outline_attempts = []

        for tool_use in tool_history[-6:]:  # Last 6 tool uses
            tool_name = tool_use.get("tool", "unknown")
            tool_input = tool_use.get("input", {})
            success = tool_use.get("success", False)

            status = "✓" if success else "✗"

            if tool_name == "search_course_content":
                query = tool_input.get("query", "")
                course = tool_input.get("course_name", "")
                search_attempts.append(
                    f"{status} searched '{query}' {f'in {course}' if course else ''}"
                )
            elif tool_name == "get_course_outline":
                course = tool_input.get("course_name", "")
                outline_attempts.append(f"{status} outlined '{course}'")

        summary_parts = []
        if search_attempts:
            summary_parts.append(f"Searches: {'; '.join(search_attempts)}")
        if outline_attempts:
            summary_parts.append(f"Outlines: {'; '.join(outline_attempts)}")

        return ". ".join(summary_parts) if summary_parts else "No recent searches"

    def _summarize_reasoning(self, reasoning_trace: List[str]) -> str:
        """Summarize key reasoning insights"""
        if not reasoning_trace:
            return "None"

        # Take the most recent reasoning entries
        recent_reasoning = reasoning_trace[-3:]

        # Extract key insights (remove round prefixes and truncate)
        insights = []
        for trace in recent_reasoning:
            # Remove "Round X:" prefix
            clean_trace = re.sub(r"^Round \d+:\s*", "", trace)
            # Take first meaningful part
            if len(clean_trace) > 50:
                clean_trace = clean_trace[:50] + "..."
            insights.append(clean_trace)

        return "; ".join(insights)

    def _extract_key_info(self, text: str, max_length: int = 100) -> str:
        """Extract key information from text"""
        if not text:
            return ""

        # Clean up the text
        clean_text = text.strip()

        # Extract first meaningful sentence or concept
        sentences = clean_text.split(".")
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > max_length:
                return first_sentence[:max_length] + "..."
            return first_sentence

        # Fallback to truncated text
        if len(clean_text) > max_length:
            return clean_text[:max_length] + "..."

        return clean_text

    def _analyze_tool_patterns(self, tool_executions: List) -> List[str]:
        """Analyze what tools were used in this round"""
        return [
            tool_exec.tool_name for tool_exec in tool_executions if tool_exec.success
        ]

    def _extract_query_refinements(self, tool_executions: List) -> str:
        """Extract how queries were refined in tool calls"""
        refinements = []

        for tool_exec in tool_executions:
            if tool_exec.success and tool_exec.tool_input:
                query = tool_exec.tool_input.get("query", "")
                if query and len(query) > 0:
                    # Extract key terms from the refined query
                    key_terms = self._extract_key_terms(query)
                    if key_terms:
                        refinements.append(key_terms)

        return "; ".join(refinements[:2])  # Limit to avoid bloat

    def _extract_key_terms(self, query: str) -> str:
        """Extract key terms from a search query"""
        # Simple extraction of important words (non-stopwords)
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "what",
            "how",
            "when",
            "where",
            "why",
        }

        words = query.lower().split()
        key_words = [word for word in words if word not in stopwords and len(word) > 2]

        return " ".join(key_words[:3])  # Top 3 key terms

    def _compress_briefing(self, briefing: str) -> str:
        """Compress a briefing that's getting too long"""
        # Simple compression: take first part of each section
        sections = briefing.split("\n\n")
        compressed_sections = []

        for section in sections:
            if len(section) > 200:
                # Truncate long sections
                compressed_sections.append(section[:200] + "...")
            else:
                compressed_sections.append(section)

        return "\n\n".join(compressed_sections)
