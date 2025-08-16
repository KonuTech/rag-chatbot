"""
Response Assembler - Constructs final responses from multiple reasoning rounds.
Handles response assembly, source extraction, and partial completion scenarios.
"""

from typing import Any, Dict, List, Tuple

from component_interfaces import (
    IResponseAssembler,
    ReasoningConfig,
    ReasoningSession,
    TerminationReason,
)


class ResponseAssembler(IResponseAssembler):
    """
    Assembles final responses from completed reasoning sessions.

    Handles:
    - Natural completion from Claude's final response
    - Partial completion due to round limits or errors
    - Source extraction and consolidation
    - Response quality optimization
    """

    def __init__(self, config: ReasoningConfig):
        self.config = config

    def assemble_final_response(
        self, session: ReasoningSession
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assemble final response from completed reasoning session.

        For naturally completed sessions, returns Claude's final response.
        For other termination reasons, uses the best available response.
        """
        if session.termination_reason == TerminationReason.NATURAL_COMPLETION:
            return self._handle_natural_completion(session)
        else:
            return self.handle_partial_completion(session, session.termination_reason)

    def handle_partial_completion(
        self, session: ReasoningSession, termination_reason: TerminationReason
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Handle case where session terminated before natural completion.

        Strategies:
        1. Use the best available response from any round
        2. Synthesize information from multiple rounds
        3. Provide helpful error context when appropriate
        """
        if not session.rounds:
            return self._handle_no_rounds(termination_reason)

        # Find the best response from available rounds
        best_response = self._find_best_response(session)
        sources = self.extract_sources(session)

        # Add context about why the session terminated early
        if termination_reason == TerminationReason.MAX_ROUNDS_REACHED:
            if best_response:
                # Don't modify the response - just return what we have
                return best_response, sources
            else:
                response = "I've gathered some information but need more rounds to provide a complete answer."
                return response, sources

        elif termination_reason == TerminationReason.CONTEXT_OVERFLOW:
            if best_response:
                return best_response, sources
            else:
                response = (
                    "The query is quite complex. Based on the information I found: "
                    + self._synthesize_findings(session)
                )
                return response, sources

        elif termination_reason == TerminationReason.TOOL_FAILURE:
            if best_response:
                return best_response, sources
            else:
                response = (
                    "I encountered some technical difficulties while searching, but here's what I can tell you: "
                    + self._synthesize_findings(session)
                )
                return response, sources

        elif termination_reason == TerminationReason.API_ERROR:
            if best_response:
                return best_response, sources
            else:
                response = "I experienced some technical issues, but I was able to gather some information for you."
                return response, sources

        else:
            # Unknown termination reason
            if best_response:
                return best_response, sources
            else:
                response = "I've gathered some information about your question."
                return response, sources

    def extract_sources(self, session: ReasoningSession) -> List[Dict[str, Any]]:
        """Extract all sources found during the reasoning session"""
        sources = []
        seen_sources = set()  # Avoid duplicates

        for round_data in session.rounds:
            for tool_exec in round_data.tool_executions:
                if tool_exec.success and tool_exec.tool_name.startswith(
                    "search_course_content"
                ):
                    # Extract sources from search results
                    round_sources = self._extract_sources_from_search_result(
                        tool_exec.result
                    )
                    for source in round_sources:
                        source_key = f"{source.get('text', '')}{source.get('url', '')}"
                        if source_key not in seen_sources:
                            sources.append(source)
                            seen_sources.add(source_key)

        return sources

    def _handle_natural_completion(
        self, session: ReasoningSession
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Handle naturally completed sessions"""
        if not session.rounds:
            return "I couldn't process your question properly.", []

        # Get the final round's response
        final_round = session.rounds[-1]

        if final_round.final_text:
            response = final_round.final_text
        else:
            # Fallback to best available response
            response = (
                self._find_best_response(session)
                or "I've processed your question but couldn't generate a final response."
            )

        sources = self.extract_sources(session)
        return response, sources

    def _handle_no_rounds(
        self, termination_reason: TerminationReason
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Handle case where no rounds were completed"""
        if termination_reason == TerminationReason.CONTEXT_OVERFLOW:
            response = "Your question is quite complex and would require extensive processing. Could you try breaking it down into smaller, more specific questions?"
        elif termination_reason == TerminationReason.API_ERROR:
            response = "I'm experiencing technical difficulties right now. Please try your question again in a moment."
        elif termination_reason == TerminationReason.TOOL_FAILURE:
            response = "I'm having trouble accessing the course materials right now. Please try again later."
        else:
            response = "I couldn't process your question. Please try rephrasing it or ask something more specific."

        return response, []

    def _find_best_response(self, session: ReasoningSession) -> str:
        """Find the best available response from any round"""
        # Priority order:
        # 1. Final text from the last round
        # 2. Final text from any round (most recent first)
        # 3. Successful tool results (most recent first)

        # Check for final text responses (reverse order - most recent first)
        for round_data in reversed(session.rounds):
            if round_data.final_text and round_data.final_text.strip():
                # Filter out error messages and incomplete responses
                if not self._is_error_response(round_data.final_text):
                    return round_data.final_text

        # Fallback to synthesizing from tool results
        return self._synthesize_findings(session)

    def _synthesize_findings(self, session: ReasoningSession) -> str:
        """Synthesize information from tool results when no final response exists"""
        findings = []

        for round_data in session.rounds:
            for tool_exec in round_data.tool_executions:
                if tool_exec.success and tool_exec.result:
                    # Extract meaningful content from tool results
                    content = self._extract_meaningful_content(tool_exec.result)
                    if content:
                        findings.append(content)

        if findings:
            # Take the most recent and relevant findings
            recent_findings = findings[-2:]  # Last 2 findings
            return " ".join(recent_findings)
        else:
            return "I wasn't able to find specific information about your question in the available course materials."

    def _extract_meaningful_content(
        self, tool_result: str, max_length: int = 200
    ) -> str:
        """Extract meaningful content from tool results"""
        if (
            not tool_result
            or tool_result.startswith("No")
            or tool_result.startswith("Error")
        ):
            return ""

        # Clean up the result
        content = tool_result.strip()

        # Remove course headers like "[Course Name - Lesson X]"
        import re

        content = re.sub(r"^\[.*?\]\s*", "", content)

        # Take first meaningful paragraph
        paragraphs = content.split("\n\n")
        if paragraphs:
            first_paragraph = paragraphs[0].strip()
            if len(first_paragraph) > max_length:
                # Find a good break point
                sentences = first_paragraph.split(".")
                if len(sentences) > 1:
                    # Take first complete sentence(s) that fit
                    result = ""
                    for sentence in sentences:
                        if len(result + sentence + ".") <= max_length:
                            result += sentence + "."
                        else:
                            break
                    return result.strip()
                else:
                    return first_paragraph[:max_length] + "..."
            return first_paragraph

        return ""

    def _is_error_response(self, response: str) -> bool:
        """Check if a response is an error message"""
        error_indicators = [
            "Error",
            "error",
            "failed",
            "couldn't",
            "unable to",
            "technical difficulties",
            "API Error",
        ]

        response_lower = response.lower()
        return any(
            indicator.lower() in response_lower for indicator in error_indicators
        )

    def _extract_sources_from_search_result(
        self, search_result: str
    ) -> List[Dict[str, Any]]:
        """
        Extract source information from search results.

        This is a simplified version - in practice, this would integrate
        with the existing source tracking from the search tools.
        """
        sources = []

        # Look for course/lesson patterns in the search result
        import re

        # Pattern: [Course Name - Lesson X]
        course_lesson_pattern = r"\[(.*?)\s*-\s*Lesson\s+(\d+)\]"
        matches = re.findall(course_lesson_pattern, search_result)

        for course_name, lesson_num in matches:
            sources.append(
                {
                    "text": f"{course_name.strip()} - Lesson {lesson_num}",
                    "url": None,  # Would be populated by the actual search tool
                }
            )

        # Pattern: [Course Name] (without lesson)
        if not sources:
            course_pattern = r"\[(.*?)\]"
            matches = re.findall(course_pattern, search_result)
            for course_name in matches:
                if course_name.strip():
                    sources.append({"text": course_name.strip(), "url": None})

        return sources
