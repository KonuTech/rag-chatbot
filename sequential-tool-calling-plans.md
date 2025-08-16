# Sequential Tool Calling Implementation Plans

This document outlines two architectural approaches for implementing sequential tool calling in the RAG system, allowing Claude to make up to 2 tool calls in separate API rounds.

## Current Problem

- Claude makes 1 tool call → tools are removed from API params → final response
- If Claude wants another tool call after seeing results, it can't (gets empty response)
- Need support for complex queries requiring multiple searches, comparisons, or multi-part questions

## Plan A: Round-Based Sequential Tool Calling Architecture

### Core Architecture Changes

**Main Approach**: Transform `generate_response` into a **round-based loop system** where each tool call represents a separate API conversation round, maintaining full tool access throughout.

**Key Structural Changes**:
- Replace the current single API call + tool execution pattern with a `for round in range(max_rounds)` loop
- Each round maintains full conversation context and tool availability
- Implement round state tracking to decide when to terminate early

### Modified `generate_response` Method Structure

```python
generate_response(query, conversation_history, tools, tool_manager):
    1. Initialize round tracking:
       - current_round = 0
       - max_rounds = 2  
       - conversation_context = [base_user_message]
       - round_history = []

    2. Main processing loop:
       for current_round in range(max_rounds):
           a) Build API parameters with tools ALWAYS included
           b) Make Claude API call with full conversation context
           c) Analyze response type:
              - If no tool_use: TERMINATE (natural completion)
              - If tool_use: Execute tools and continue
           d) Execute all tools in current round
           e) Add AI response + tool results to conversation context
           f) Update round history for debugging/logging
           
    3. Post-loop cleanup:
       - Extract final text response
       - Log round statistics
       - Return formatted response
```

### Round and Conversation State Tracking

**Round State Object**:
```python
@dataclass
class RoundState:
    round_number: int
    tools_used: List[str]
    tool_results: List[Dict]
    ai_response_content: List[Any]
    termination_reason: Optional[str] = None
```

**Conversation Context Management**:
- Maintain running `messages` list that grows with each round
- Each round adds: `[assistant_message, user_tool_results_message]`
- Preserve original system prompt and user query throughout
- Track cumulative token usage across rounds

**Early Termination Logic**:
```python
def should_terminate(response, current_round, max_rounds) -> Tuple[bool, str]:
    # Natural completion - no tools requested
    if response.stop_reason != "tool_use":
        return True, "natural_completion"
    
    # Maximum rounds reached
    if current_round >= max_rounds - 1:
        return True, "max_rounds_reached"
    
    # Continue to next round
    return False, None
```

### Tool Execution and Error Handling

**Enhanced Tool Execution Strategy**:
- Execute all tools in current round before proceeding (same as current)
- Implement **graceful degradation** for tool failures:
  ```python
  def execute_tools_with_fallback(tool_calls):
      results = []
      for tool_call in tool_calls:
          try:
              result = tool_manager.execute_tool(...)
              results.append(success_result)
          except Exception as e:
              # Add error context that Claude can understand
              error_msg = f"Tool '{tool_call.name}' failed: {str(e)}"
              results.append(error_result)
      return results
  ```

**Tool Failure Termination**:
- If ALL tools in a round fail → terminate with error response
- If SOME tools fail → continue with partial results and error context
- Log all tool execution outcomes for debugging

### Context Preservation Between API Calls

**Message History Strategy**:
```python
def build_conversation_context(base_messages, round_history):
    # Start with system prompt + original user query
    messages = base_messages.copy()
    
    # Add each completed round's conversation
    for round_state in round_history:
        # Add AI's response (including tool calls)
        messages.append({
            "role": "assistant", 
            "content": round_state.ai_response_content
        })
        
        # Add tool results as user message
        if round_state.tool_results:
            messages.append({
                "role": "user", 
                "content": round_state.tool_results
            })
    
    return messages
```

**System Prompt Updates**:
- Update system prompt to remove "One tool call per query maximum"
- Add guidance: "You can make multiple tool calls across rounds to gather comprehensive information"
- Include round awareness: "If you need additional information after seeing tool results, you can make another tool call in the next round"

### Testing Strategy

**Unit Testing Approach**:
```python
class TestSequentialToolCalling:
    def test_single_round_completion(self):
        # Query that should complete in one round
        
    def test_two_round_comparison_query(self):
        # Query requiring multiple courses comparison
        
    def test_max_rounds_termination(self):
        # Ensure system terminates at max rounds
        
    def test_tool_failure_recovery(self):
        # Verify graceful handling of tool failures
        
    def test_conversation_context_preservation(self):
        # Ensure context maintained across rounds
```

**Benefits**: Minimal architectural changes, backwards compatible, straightforward to implement and test

---

## Plan B: Event-Driven Multi-Round Orchestration

### Core Architectural Philosophy

Instead of extending the current linear API call pattern, Plan B reimagines the system as an **event-driven orchestration pipeline** where each tool call becomes a discrete "reasoning event" that can trigger subsequent events.

### Alternative API Call Loop Structure

**Current Approach:** Single method with embedded tool handling
**Plan B:** Event-driven pipeline with discrete reasoning phases

```
Query Coordinator
    ↓
[ Reasoning Event 1 ] → [ Tool Execution Event ] → [ Synthesis Event ]
    ↓
[ Reasoning Event 2 ] → [ Tool Execution Event ] → [ Final Response Event ]
```

**Key Components:**

- **Query Coordinator**: Manages the overall query lifecycle and enforces round limits
- **Reasoning Engine**: Isolated component that makes individual API calls to Claude
- **Tool Execution Dispatcher**: Handles tool calls asynchronously 
- **Context Synthesizer**: Builds context between rounds using structured memory
- **Response Assembler**: Constructs final responses from multiple reasoning phases

**Implementation Strategy:**
- Each "reasoning round" is a completely independent API call with full context reconstruction
- Use a queue-based system where each round enqueues the next potential round
- Implement circuit breakers at the coordinator level rather than within individual calls

### State Management & Context Preservation

**Multi-Layer Context System:**

1. **Factual Layer**: Concrete information discovered from tool calls
2. **Reasoning Layer**: Claude's intermediate thoughts and reasoning patterns
3. **Intent Layer**: Evolving understanding of user goals
4. **Tool Usage Layer**: History of what searches were attempted and why

**Context Reconstruction Strategy:**
- Instead of passing raw conversation history, reconstruct context as a structured "reasoning briefing"
- Use semantic compression: summarize previous rounds into key insights rather than preserving exact message chains
- Implement "context decay" where older reasoning rounds are summarized more aggressively

**State Persistence Design:**
```python
class ReasoningSession:
    def __init__(self):
        self.rounds = []
        self.discovered_facts = {}
        self.search_coverage = {}
        self.evolving_intent = ""
        self.reasoning_trace = []
```

### Error Handling Strategy

**Error Recovery Hierarchy:**

1. **Tool-Level Recovery**: If a tool call fails, substitute with alternative search strategies
2. **Round-Level Recovery**: If a reasoning round fails, attempt simplified context reconstruction
3. **Session-Level Recovery**: If multiple rounds fail, gracefully degrade to single-round response
4. **Graceful Degradation**: Always maintain ability to provide a response, even if suboptimal

**Specific Strategies:**
- **Tool Substitution**: If `search_course_content` fails, automatically try `get_course_outline` with modified parameters
- **Context Simplification**: If context becomes too large, use AI to summarize previous rounds
- **Partial Response Assembly**: If final round fails, assemble response from successful earlier rounds
- **User Feedback Loop**: Include "uncertainty indicators" in responses when error recovery was used

### Code Structure Organization

**Component Separation:**

```
rag_system.py (Orchestrator only)
    ↓
reasoning_coordinator.py (Query lifecycle management)
    ↓
reasoning_engine.py (Individual Claude API calls)
context_synthesizer.py (Multi-round context building)
tool_dispatcher.py (Async tool execution)
response_assembler.py (Final response construction)
```

**Benefits of this structure:**
- Each component has a single responsibility
- Easy to test individual reasoning phases in isolation
- Can be extended to true microservices later if needed
- Clear separation of concerns for debugging

**Interface Design:**
- All components communicate through well-defined events/messages
- No direct coupling between reasoning engine and tool execution
- Context synthesizer acts as the "memory" that other components query

### Testing Approach

**Test Strategy Levels:**

1. **Reasoning Scenario Tests**: Test complete multi-round reasoning flows with known expected outcomes
2. **Context Evolution Tests**: Verify that context building preserves important information across rounds
3. **Error Recovery Tests**: Simulate various failure modes and verify graceful degradation
4. **Tool Interaction Tests**: Mock sequences of tool calls and verify logical progression
5. **Performance Boundary Tests**: Test behavior at the 2-round limit with complex queries

**Novel Testing Techniques:**

- **Reasoning Replay**: Record actual reasoning sessions and replay them to verify consistency
- **Context Compression Testing**: Verify that summarized context still enables good reasoning
- **Error Injection Testing**: Systematically inject failures at different points in the pipeline
- **Comparative Quality Testing**: Compare single-round vs multi-round responses on the same queries

### Key Differentiators from Linear Approaches

**Event-Driven vs Sequential:**
- No predetermined flow - each reasoning event determines the next based on results
- Can handle unexpected tool results more flexibly
- Easier to add new reasoning patterns without changing core logic

**Semantic Context vs Raw History:**
- Context is actively constructed rather than passively accumulated
- Better handling of complex queries that span multiple topics
- More efficient token usage through intelligent summarization

**Component Isolation vs Monolithic Methods:**
- Each component can be optimized independently
- Easier to debug specific aspects of multi-round reasoning
- Clear upgrade path if any component needs to become more sophisticated

**Benefits**: More sophisticated, better error recovery, cleaner separation of concerns, more scalable

---

## Recommendation

**Plan A** is more **pragmatic and immediate** - builds directly on existing architecture with minimal refactoring.

**Plan B** is more **architected and future-proof** - creates a sophisticated reasoning system that handles complex scenarios better and is easier to extend.

Both plans support complex queries like "Search for a course that discusses the same topic as lesson 4 of course X" but Plan B provides better error handling and extensibility.