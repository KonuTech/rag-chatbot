"""
Simple integration test to verify multi-round sequential tool calling works end-to-end.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator_v2 import AIGeneratorV2


class TestSimpleMultiRound:
    """Simple tests to verify multi-round functionality"""
    
    def setup_method(self):
        """Set up minimal test environment"""
        # Create mock tool manager
        self.mock_tool_manager = Mock()
        self.mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"}
        ]
        self.mock_tool_manager.execute_tool.return_value = "Mock search result"
        self.mock_tool_manager.get_last_sources.return_value = []
        self.mock_tool_manager.reset_sources.return_value = None
        
        # Create AI Generator V2
        self.ai_generator = AIGeneratorV2(
            api_key="test-key",
            model="test-model",
            tool_manager=self.mock_tool_manager
        )
    
    def test_v2_initialization(self):
        """Test that AIGeneratorV2 initializes correctly"""
        assert self.ai_generator is not None
        assert hasattr(self.ai_generator, 'config')
        assert hasattr(self.ai_generator, 'reasoning_coordinator')
        assert hasattr(self.ai_generator, 'reasoning_engine')
        assert hasattr(self.ai_generator, 'context_synthesizer')
        assert hasattr(self.ai_generator, 'tool_dispatcher')
        assert hasattr(self.ai_generator, 'response_assembler')
    
    def test_configuration_access(self):
        """Test configuration access and modification"""
        config = self.ai_generator.get_config()
        assert config.max_rounds == 2  # Default value
        
        # Test configuration update
        self.ai_generator.update_config(max_rounds=3)
        updated_config = self.ai_generator.get_config()
        assert updated_config.max_rounds == 3
    
    def test_system_prompt_updated(self):
        """Test that V2 has updated system prompt"""
        system_prompt = self.ai_generator.SYSTEM_PROMPT
        
        # V2 should support multi-round reasoning
        assert "multi-round reasoning" in system_prompt.lower()
        assert "progressive refinement" in system_prompt.lower()
        
        # Should NOT have the V1 limitation
        assert "One tool call per query maximum" not in system_prompt
    
    def test_simple_query_fallback(self):
        """Test fallback to simple response for queries without tools"""
        # Mock the Anthropic client for simple response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Simple answer")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        self.ai_generator.client = mock_client
        self.ai_generator.reasoning_engine.client = mock_client
        
        # Test simple query without tools
        result = self.ai_generator.generate_response("What is 2+2?")
        
        assert result == "Simple answer"
        mock_client.messages.create.assert_called_once()
    
    @patch('asyncio.run')
    def test_multi_round_coordination(self, mock_asyncio_run):
        """Test that multi-round coordination is invoked when tools are present"""
        # Configure mock session
        mock_session = Mock()
        mock_session.termination_reason = Mock()
        mock_asyncio_run.return_value = mock_session
        
        # Use patch to mock the response assembler method
        with patch.object(self.ai_generator.response_assembler, 'assemble_final_response') as mock_assemble:
            mock_assemble.return_value = ("Multi-round response", [])
            
            result = self.ai_generator.generate_response(
                query="Test query",
                tools=[{"name": "search_course_content"}],
                tool_manager=self.mock_tool_manager
            )
            
            # Should invoke multi-round coordination
            mock_asyncio_run.assert_called_once()
            assert result == "Multi-round response"
    
    def test_error_handling(self):
        """Test error handling and graceful fallback"""
        # Configure multi-round to fail
        with patch('asyncio.run', side_effect=Exception("Coordination failed")):
            # Mock fallback response
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text="Fallback response")]
            mock_client.messages.create.return_value = mock_response
            
            self.ai_generator.client = mock_client
            self.ai_generator.reasoning_engine.client = mock_client
            
            result = self.ai_generator.generate_response(
                query="Test query",
                tools=[{"name": "search_course_content"}],
                tool_manager=self.mock_tool_manager
            )
            
            # Should fall back gracefully
            assert result == "Fallback response"
    
    def test_metrics_availability(self):
        """Test that metrics methods are available and functional"""
        # Test metrics methods exist and return data
        session_metrics = self.ai_generator.get_session_metrics()
        tool_metrics = self.ai_generator.get_tool_metrics()
        
        assert isinstance(session_metrics, dict)
        assert isinstance(tool_metrics, dict)
        
        # Test reset
        self.ai_generator.reset_metrics()
    
    def test_backward_compatibility_interface(self):
        """Test that V2 maintains backward compatibility with V1 interface"""
        # Should have the same core methods as V1
        assert hasattr(self.ai_generator, 'generate_response')
        assert hasattr(self.ai_generator, 'SYSTEM_PROMPT')
        assert hasattr(self.ai_generator, 'client')
        assert hasattr(self.ai_generator, 'model')
        assert hasattr(self.ai_generator, 'api_key')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])