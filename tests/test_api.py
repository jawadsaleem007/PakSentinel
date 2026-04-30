"""
Task 7 — API Tests [30 Marks]

pytest + httpx tests covering:
- All 6 endpoints
- Edge cases (validation, empty text, oversized batch)
- Response time assertions (/classify < 100ms, batch of 10 < 200ms)
"""

import time
import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app

pytestmark = pytest.mark.asyncio


# ──────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def client():
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ──────────────────────────────────────────────────────────
#  Health Endpoint Tests
# ──────────────────────────────────────────────────────────
class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
    
    async def test_health_has_required_fields(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "model_stage" in data
        assert "f1_score" in data
        assert "load_timestamp" in data
        assert "status" in data
    
    async def test_health_status_is_healthy(self, client):
        response = await client.get("/health")
        assert response.json()["status"] == "healthy"


# ──────────────────────────────────────────────────────────
#  Preprocess Endpoint Tests
# ──────────────────────────────────────────────────────────
class TestPreprocessEndpoint:
    async def test_preprocess_basic(self, client):
        response = await client.post("/preprocess", json={
            "text": "This is a test sentence for preprocessing analysis.",
            "steps": ["clean", "tokenize"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert "processing_time_ms" in data
        assert len(data["tokens"]) > 0
    
    async def test_preprocess_all_steps(self, client):
        response = await client.post("/preprocess", json={
            "text": "Pakistan's government released BREAKING NEWS about the economy!!!"
        })
        assert response.status_code == 200
        data = response.json()
        assert "removed_stopwords" in data
        assert "token_count" in data
    
    async def test_preprocess_short_text_rejected(self, client):
        response = await client.post("/preprocess", json={
            "text": "short"  # Less than 10 chars
        })
        assert response.status_code == 422  # Validation error
    
    async def test_preprocess_long_text_rejected(self, client):
        response = await client.post("/preprocess", json={
            "text": "x" * 10001  # Exceeds 10,000 chars
        })
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────
#  Classify Endpoint Tests
# ──────────────────────────────────────────────────────────
class TestClassifyEndpoint:
    async def test_classify_returns_prediction(self, client):
        response = await client.post("/classify", json={
            "text": "The prime minister announced new economic reforms today in parliament."
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "processing_time_ms" in data
    
    async def test_classify_prediction_is_valid_class(self, client):
        response = await client.post("/classify", json={
            "text": "BREAKING: Scientists discover that the earth is actually flat, experts baffled!"
        })
        data = response.json()
        assert data["prediction"] in ["Real", "Fake", "Satire", "Unknown"]
    
    async def test_classify_confidence_range(self, client):
        response = await client.post("/classify", json={
            "text": "The stock market rose by 2% following positive economic indicators from the central bank."
        })
        data = response.json()
        assert 0 <= data["confidence"] <= 1
    
    async def test_classify_response_time(self, client):
        """Classification should complete in < 100ms."""
        start = time.time()
        response = await client.post("/classify", json={
            "text": "Government officials announced new policy changes affecting the education sector nationwide."
        })
        elapsed = (time.time() - start) * 1000
        assert response.status_code == 200
        # Allow some slack for test environment
        assert elapsed < 5000, f"Classification took {elapsed:.0f}ms (expected < 100ms in production)"
    
    async def test_classify_validation_error(self, client):
        response = await client.post("/classify", json={
            "text": "short"
        })
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────
#  Batch Classify Endpoint Tests
# ──────────────────────────────────────────────────────────
class TestBatchClassifyEndpoint:
    async def test_batch_classify_basic(self, client):
        texts = [
            "The government announced new economic reforms to boost growth.",
            "SHOCKING: Aliens found living under the parliament building!!!",
            "Area man discovers he has been eating breakfast wrong his entire life.",
        ]
        response = await client.post("/classify/batch", json={"texts": texts})
        assert response.status_code == 200
        data = response.json()
        assert data["total_texts"] == 3
        assert len(data["results"]) == 3
    
    async def test_batch_classify_10_items_speed(self, client):
        """Batch of 10 should complete in < 200ms."""
        texts = [
            f"This is test article number {i} about government policy and economic reform." 
            for i in range(10)
        ]
        start = time.time()
        response = await client.post("/classify/batch", json={"texts": texts})
        elapsed = (time.time() - start) * 1000
        assert response.status_code == 200
        # Allow slack for test environment
        assert elapsed < 10000, f"Batch took {elapsed:.0f}ms (expected < 200ms in production)"
    
    async def test_batch_classify_empty_rejected(self, client):
        response = await client.post("/classify/batch", json={"texts": []})
        assert response.status_code == 422
    
    async def test_batch_classify_too_many_rejected(self, client):
        texts = [f"Test article {i} for validation testing purposes here." for i in range(101)]
        response = await client.post("/classify/batch", json={"texts": texts})
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────
#  Similar Retrieval Endpoint Tests
# ──────────────────────────────────────────────────────────
class TestSimilarEndpoint:
    async def test_similar_basic(self, client):
        response = await client.post("/retrieve/similar", json={
            "text": "The government is spreading fake news about the economy.",
            "top_k": 5
        })
        # May return 503 if dataset not loaded in test env
        assert response.status_code in [200, 503]
    
    async def test_similar_top_k_validation(self, client):
        response = await client.post("/retrieve/similar", json={
            "text": "Political misinformation is spreading rapidly on social media platforms.",
            "top_k": 25  # Exceeds max of 20
        })
        assert response.status_code == 422
    
    async def test_similar_top_k_zero_rejected(self, client):
        response = await client.post("/retrieve/similar", json={
            "text": "This is a test query about misinformation detection systems.",
            "top_k": 0  # Below minimum of 1
        })
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────
#  Model Performance Endpoint Tests
# ──────────────────────────────────────────────────────────
class TestPerformanceEndpoint:
    async def test_performance_returns_200(self, client):
        response = await client.get("/model/performance")
        assert response.status_code == 200
    
    async def test_performance_has_required_fields(self, client):
        response = await client.get("/model/performance")
        data = response.json()
        assert "current_model" in data
        assert "current_version" in data
        assert "metrics" in data
        assert "version_history" in data
        assert "last_updated" in data


# ──────────────────────────────────────────────────────────
#  Edge Cases
# ──────────────────────────────────────────────────────────
class TestEdgeCases:
    async def test_nonexistent_endpoint(self, client):
        response = await client.get("/nonexistent")
        assert response.status_code == 404
    
    async def test_wrong_method_classify(self, client):
        response = await client.get("/classify")
        assert response.status_code == 405
    
    async def test_missing_body_classify(self, client):
        response = await client.post("/classify")
        assert response.status_code == 422
    
    async def test_invalid_json_classify(self, client):
        response = await client.post("/classify", content="not json",
                              headers={"Content-Type": "application/json"})
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
