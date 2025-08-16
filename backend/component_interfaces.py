"""
Component interfaces for the event-driven multi-round orchestration architecture.
These define the contracts between components in the new sequential tool calling system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class ReasoningEventType(Enum):
    """Types of reasoning events in the orchestration pipeline"""
    INITIAL_QUERY = "initial_query"
    TOOL_EXECUTION_COMPLETE = "tool_execution_complete"
    REASONING_ROUND_COMPLETE = "reasoning_round_complete"
    SESSION_COMPLETE = "session_complete"
    ERROR_OCCURRED = "error_occurred"


class TerminationReason(Enum):
    """Reasons why a reasoning session terminated"""
    NATURAL_COMPLETION = "natural_completion"  # Claude provided final answer without tools
    MAX_ROUNDS_REACHED = "max_rounds_reached"  # Hit the 2-round limit
    TOOL_FAILURE = "tool_failure"             # Critical tool execution failure
    API_ERROR = "api_error"                   # Anthropic API error
    CONTEXT_OVERFLOW = "context_overflow"     # Context became too large
    USER_CANCELLATION = "user_cancellation"  # User cancelled (future use)


@dataclass
class ReasoningEvent:
    """Event that flows through the orchestration pipeline"""
    event_type: ReasoningEventType
    session_id: str
    round_number: int
    timestamp: float
    data: Dict[str, Any]
    error: Optional[Exception] = None


@dataclass
class ToolExecutionResult:
    """Result from executing one or more tools"""
    tool_name: str
    tool_input: Dict[str, Any]
    success: bool
    result: str
    execution_time: float
    error: Optional[Exception] = None


@dataclass
class ReasoningRound:
    """Complete data for one reasoning round"""
    round_number: int
    user_query: str
    ai_response_content: List[Any]  # Raw Claude response content blocks
    tool_executions: List[ToolExecutionResult]
    final_text: Optional[str] = None
    round_duration: float = 0.0
    token_usage: Dict[str, int] = None


@dataclass
class ReasoningSession:
    """Complete reasoning session with multi-layer context"""
    session_id: str
    original_query: str
    rounds: List[ReasoningRound]
    discovered_facts: Dict[str, Any]  # Factual layer
    reasoning_trace: List[str]        # Reasoning layer
    evolving_intent: str              # Intent layer
    tool_usage_history: List[Dict]    # Tool usage layer
    termination_reason: Optional[TerminationReason] = None
    total_duration: float = 0.0
    total_tokens: int = 0


class IReasoningCoordinator(ABC):
    """Interface for the main query lifecycle coordinator"""
    
    @abstractmethod
    async def process_query(self, query: str, session_id: Optional[str] = None) -> ReasoningSession:
        """
        Process a complete user query through multi-round reasoning.
        
        Args:
            query: User's question
            session_id: Optional session ID for conversation context
            
        Returns:
            Complete reasoning session with results
        """
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[ReasoningSession]:
        """Get an existing reasoning session by ID"""
        pass
    
    @abstractmethod
    def terminate_session(self, session_id: str, reason: TerminationReason) -> None:
        """Forcefully terminate a reasoning session"""
        pass


class IReasoningEngine(ABC):
    """Interface for individual Claude API call management"""
    
    @abstractmethod
    async def execute_reasoning_round(
        self, 
        query: str, 
        context_briefing: str,
        tools: List[Dict[str, Any]],
        round_number: int
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
        pass
    
    @abstractmethod
    def estimate_token_usage(self, context_briefing: str, query: str) -> int:
        """Estimate tokens that would be used for a reasoning round"""
        pass


class IContextSynthesizer(ABC):
    """Interface for multi-round context building with semantic compression"""
    
    @abstractmethod
    def build_context_briefing(self, session: ReasoningSession, round_number: int) -> str:
        """
        Build synthesized context briefing for the next reasoning round.
        
        Args:
            session: Current reasoning session
            round_number: Round number this briefing is for
            
        Returns:
            Compressed context briefing for Claude
        """
        pass
    
    @abstractmethod
    def extract_factual_information(self, round: ReasoningRound) -> Dict[str, Any]:
        """Extract key facts discovered in a reasoning round"""
        pass
    
    @abstractmethod
    def update_intent_understanding(self, session: ReasoningSession, round: ReasoningRound) -> str:
        """Update understanding of user's evolving intent"""
        pass
    
    @abstractmethod
    def should_compress_context(self, session: ReasoningSession) -> bool:
        """Determine if context compression is needed"""
        pass


class IToolDispatcher(ABC):
    """Interface for async tool execution management"""
    
    @abstractmethod
    async def execute_tools(
        self, 
        tool_calls: List[Dict[str, Any]], 
        session_id: str,
        round_number: int
    ) -> List[ToolExecutionResult]:
        """
        Execute multiple tool calls and return results.
        
        Args:
            tool_calls: List of tool call requests from Claude
            session_id: Current session ID
            round_number: Current round number
            
        Returns:
            List of tool execution results
        """
        pass
    
    @abstractmethod
    def get_fallback_tools(self, failed_tool: str) -> List[str]:
        """Get alternative tools when primary tool fails"""
        pass
    
    @abstractmethod
    def can_retry_tool(self, tool_name: str, error: Exception) -> bool:
        """Determine if a failed tool execution can be retried"""
        pass


class IResponseAssembler(ABC):
    """Interface for constructing final responses from multiple rounds"""
    
    @abstractmethod
    def assemble_final_response(self, session: ReasoningSession) -> tuple[str, List[Dict[str, Any]]]:
        """
        Assemble final response from completed reasoning session.
        
        Args:
            session: Completed reasoning session
            
        Returns:
            Tuple of (final_response_text, sources_list)
        """
        pass
    
    @abstractmethod
    def handle_partial_completion(
        self, 
        session: ReasoningSession, 
        termination_reason: TerminationReason
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Handle case where session terminated before natural completion.
        
        Args:
            session: Partially completed session
            termination_reason: Why the session terminated
            
        Returns:
            Tuple of (best_effort_response, available_sources)
        """
        pass
    
    @abstractmethod
    def extract_sources(self, session: ReasoningSession) -> List[Dict[str, Any]]:
        """Extract all sources found during the reasoning session"""
        pass


class EventBus(ABC):
    """Simple event bus for component communication"""
    
    @abstractmethod
    def publish(self, event: ReasoningEvent) -> None:
        """Publish an event to all subscribers"""
        pass
    
    @abstractmethod
    def subscribe(self, event_type: ReasoningEventType, handler) -> None:
        """Subscribe to specific event types"""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: ReasoningEventType, handler) -> None:
        """Unsubscribe from event types"""
        pass


class ReasoningConfig:
    """Configuration for the reasoning system"""
    
    def __init__(self):
        self.max_rounds = 2
        self.max_tokens_per_round = 800
        self.max_total_tokens = 1600
        self.context_compression_threshold = 1200  # tokens
        self.tool_timeout_seconds = 30
        self.reasoning_timeout_seconds = 120
        self.enable_tool_fallbacks = True
        self.enable_context_compression = True
        self.enable_partial_responses = True


# Error classes for the reasoning system
class ReasoningError(Exception):
    """Base exception for reasoning system errors"""
    pass


class ContextOverflowError(ReasoningError):
    """Raised when context becomes too large to process"""
    pass


class ToolExecutionError(ReasoningError):
    """Raised when tool execution fails critically"""
    pass


class APIError(ReasoningError):
    """Raised when Anthropic API calls fail"""
    pass


class RoundLimitError(ReasoningError):
    """Raised when maximum reasoning rounds exceeded"""
    pass