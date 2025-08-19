# tests/test_classifier.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.experiment.classifier import LLMJudge

class TestLLMJudge:
    @pytest.fixture
    def judge(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return LLMJudge("gpt-4o-mini")
    
    @pytest.mark.asyncio
    async def test_classify_with_confidence_float(self, judge):
        """Test float confidence classification"""
        with patch.object(judge.model, 'with_structured_output') as mock:
            mock.return_value.ainvoke = AsyncMock(return_value=Mock(
                classification=True,
                confidence=0.85
            ))
            
            result = await judge.classify_with_confidence(
                "Great product!", "float", "Classify sentiment"
            )
            
            assert result['classification'] == True
            assert 0 <= result['confidence'] <= 1
    
    def test_fallback_on_error(self, judge):
        """Test fallback behavior on API error"""
        with patch.object(judge.model, 'with_structured_output') as mock:
            mock.side_effect = Exception("API Error")
            
            result = judge.classify_with_confidence_sync(
                "Test text", "float"
            )
            
            assert 'classification' in result
            assert 'confidence' in result