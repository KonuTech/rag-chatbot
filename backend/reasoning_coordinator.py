"""
Reasoning Coordinator - Main orchestrator for multi-round reasoning sessions.
Manages query lifecycle, enforces round limits, and coordinates between components.
"""

import time
import uuid
from typing import Optional, Dict
from component_interfaces import (
    IReasoningCoordinator, IReasoningEngine, IContextSynthesizer, 
    IToolDispatcher, IResponseAssembler, ReasoningSession, ReasoningEvent,
    ReasoningEventType, TerminationReason, ReasoningConfig, ReasoningRound,
    ReasoningError, RoundLimitError, ContextOverflowError
)


class ReasoningCoordinator(IReasoningCoordinator):
    """
    Main coordinator for multi-round reasoning sessions.
    Implements event-driven orchestration with circuit breakers and fallback strategies.
    """
    
    def __init__(
        self,
        reasoning_engine: IReasoningEngine,
        context_synthesizer: IContextSynthesizer,
        tool_dispatcher: IToolDispatcher,
        response_assembler: IResponseAssembler,
        config: ReasoningConfig
    ):
        self.reasoning_engine = reasoning_engine
        self.context_synthesizer = context_synthesizer
        self.tool_dispatcher = tool_dispatcher
        self.response_assembler = response_assembler
        self.config = config
        
        # Session management
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.completed_sessions: Dict[str, ReasoningSession] = {}
        
        # Performance tracking
        self.session_metrics = {
            "total_sessions": 0,
            "successful_completions": 0,
            "early_terminations": 0,
            "error_terminations": 0
        }
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> ReasoningSession:
        """
        Process a complete user query through multi-round reasoning.
        
        This is the main entry point that orchestrates the entire reasoning process:
        1. Initialize or retrieve session
        2. Execute reasoning rounds (up to max_rounds)
        3. Handle tool execution and context building
        4. Terminate with appropriate reason
        5. Assemble final response
        """
        start_time = time.time()
        
        # Create or retrieve session
        if session_id is None:
            session_id = self._generate_session_id()
        
        session = self._initialize_session(session_id, query)
        
        try:
            # Execute reasoning rounds
            while session.termination_reason is None:
                # Check round limits
                if len(session.rounds) >= self.config.max_rounds:
                    session.termination_reason = TerminationReason.MAX_ROUNDS_REACHED
                    break
                
                # Check token limits
                if session.total_tokens > self.config.max_total_tokens:
                    session.termination_reason = TerminationReason.CONTEXT_OVERFLOW
                    break
                
                # Execute one reasoning round
                round_result = await self._execute_reasoning_round(session)
                
                # Add round to session
                session.rounds.append(round_result)
                session.total_tokens += round_result.token_usage.get('total', 0) if round_result.token_usage else 0
                
                # Update multi-layer context
                self._update_session_context(session, round_result)
                
                # Check if Claude naturally completed (no tool use)
                if not round_result.tool_executions:
                    session.termination_reason = TerminationReason.NATURAL_COMPLETION
                    break
            
            # Calculate total duration
            session.total_duration = time.time() - start_time
            
            # Move to completed sessions
            self._complete_session(session)
            
            return session
            
        except Exception as e:
            # Handle any errors during processing
            session.termination_reason = TerminationReason.API_ERROR
            session.total_duration = time.time() - start_time
            self._complete_session(session)
            raise ReasoningError(f"Query processing failed: {str(e)}") from e
    
    def get_session(self, session_id: str) -> Optional[ReasoningSession]:
        """Get an existing reasoning session by ID"""
        return (self.active_sessions.get(session_id) or 
                self.completed_sessions.get(session_id))
    
    def terminate_session(self, session_id: str, reason: TerminationReason) -> None:
        """Forcefully terminate a reasoning session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.termination_reason = reason
            self._complete_session(session)
    
    async def _execute_reasoning_round(self, session: ReasoningSession) -> ReasoningRound:
        """
        Execute a single reasoning round.
        
        This includes:
        1. Building context briefing from previous rounds
        2. Making Claude API call with tools
        3. Executing any requested tools
        4. Assembling the complete round result
        """
        round_number = len(session.rounds)
        round_start = time.time()
        
        try:
            # Build context briefing for this round
            context_briefing = ""
            if round_number > 0:
                context_briefing = self.context_synthesizer.build_context_briefing(session, round_number)
            
            # Get available tools (this would come from the tool manager)
            tools = self._get_available_tools()
            
            # Execute reasoning with Claude
            reasoning_round = await self.reasoning_engine.execute_reasoning_round(
                query=session.original_query,
                context_briefing=context_briefing,
                tools=tools,
                round_number=round_number
            )
            
            # Execute any tools Claude requested
            if self._has_tool_calls(reasoning_round):
                tool_calls = self._extract_tool_calls(reasoning_round)
                tool_results = await self.tool_dispatcher.execute_tools(
                    tool_calls, session.session_id, round_number
                )
                reasoning_round.tool_executions = tool_results
            
            # Calculate round duration
            reasoning_round.round_duration = time.time() - round_start
            
            return reasoning_round
            
        except Exception as e:
            # Create error round
            error_round = ReasoningRound(
                round_number=round_number,
                user_query=session.original_query,
                ai_response_content=[],
                tool_executions=[],
                final_text=f"Error in round {round_number}: {str(e)}",
                round_duration=time.time() - round_start
            )
            return error_round
    
    def _initialize_session(self, session_id: str, query: str) -> ReasoningSession:
        """Initialize a new reasoning session"""
        session = ReasoningSession(
            session_id=session_id,
            original_query=query,
            rounds=[],
            discovered_facts={},
            reasoning_trace=[],
            evolving_intent=query,  # Start with original query as intent
            tool_usage_history=[]
        )
        
        self.active_sessions[session_id] = session
        self.session_metrics["total_sessions"] += 1
        
        return session
    
    def _complete_session(self, session: ReasoningSession) -> None:
        """Move session from active to completed and update metrics"""
        if session.session_id in self.active_sessions:
            del self.active_sessions[session.session_id]
        
        self.completed_sessions[session.session_id] = session
        
        # Update metrics
        if session.termination_reason == TerminationReason.NATURAL_COMPLETION:
            self.session_metrics["successful_completions"] += 1
        elif session.termination_reason in [TerminationReason.MAX_ROUNDS_REACHED, TerminationReason.CONTEXT_OVERFLOW]:
            self.session_metrics["early_terminations"] += 1
        else:
            self.session_metrics["error_terminations"] += 1
    
    def _update_session_context(self, session: ReasoningSession, round_result: ReasoningRound) -> None:
        """Update the multi-layer context after each round"""
        
        # Update factual layer
        new_facts = self.context_synthesizer.extract_factual_information(round_result)
        session.discovered_facts.update(new_facts)
        
        # Update reasoning trace
        if round_result.final_text:
            session.reasoning_trace.append(f"Round {round_result.round_number}: {round_result.final_text[:100]}...")
        
        # Update evolving intent
        session.evolving_intent = self.context_synthesizer.update_intent_understanding(session, round_result)
        
        # Update tool usage history
        for tool_exec in round_result.tool_executions:
            session.tool_usage_history.append({
                "round": round_result.round_number,
                "tool": tool_exec.tool_name,
                "input": tool_exec.tool_input,
                "success": tool_exec.success,
                "duration": tool_exec.execution_time
            })
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def _get_available_tools(self) -> list:
        """Get available tools - this would be injected or configured"""
        # This would be integrated with the existing ToolManager
        # For now, return a placeholder that matches the existing tool interface
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching and lesson filtering",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for in the course content"},
                        "course_name": {"type": "string", "description": "Course title (partial matches work)"},
                        "lesson_number": {"type": "integer", "description": "Specific lesson number to search within"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_course_outline",
                "description": "Get course outline including title, link, and complete lesson list with lesson numbers and titles",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "course_name": {"type": "string", "description": "Course title or partial name to get outline for"}
                    },
                    "required": ["course_name"]
                }
            }
        ]
    
    def _has_tool_calls(self, reasoning_round: ReasoningRound) -> bool:
        """Check if Claude made any tool calls in this round"""
        if not reasoning_round.ai_response_content:
            return False
        
        # Check for tool_use content blocks
        for content_block in reasoning_round.ai_response_content:
            if hasattr(content_block, 'type') and content_block.type == "tool_use":
                return True
        
        return False
    
    def _extract_tool_calls(self, reasoning_round: ReasoningRound) -> list:
        """Extract tool call information from Claude's response"""
        tool_calls = []
        
        for content_block in reasoning_round.ai_response_content:
            if hasattr(content_block, 'type') and content_block.type == "tool_use":
                tool_calls.append({
                    "name": content_block.name,
                    "input": content_block.input,
                    "id": content_block.id
                })
        
        return tool_calls
    
    def get_session_metrics(self) -> dict:
        """Get performance metrics for the reasoning system"""
        return {
            **self.session_metrics,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.completed_sessions)
        }