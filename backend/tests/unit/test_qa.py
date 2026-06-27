from unittest.mock import patch



import pytest



from app.core.config import get_settings

from app.graphs import qa as qa_graph

from app.services import gemini as gemini_service



SAMPLE_CHUNK = {

    "message_id": "msg-1",

    "speaker": "Rahul",

    "timestamp": "2024-03-12T18:22:00",

    "snippet": "Goa trip final kar dete hain March end",

    "score": 0.9,

}





@pytest.fixture(autouse=True)

def clear_settings_cache():

    get_settings.cache_clear()

    yield

    get_settings.cache_clear()





@pytest.fixture

def mock_retrieval(monkeypatch):

    """Stub retrieval so tests only exercise LangChain answer routing."""



    def _rewrite(q):

        return q



    def _rerank(q, chunks, top_k):

        second = {**SAMPLE_CHUNK, "message_id": "msg-2"}

        return [SAMPLE_CHUNK, second]



    monkeypatch.setattr("app.services.rag_chain.rewrite_query", _rewrite)

    monkeypatch.setattr("app.services.rag_chain.rerank_chunks", _rerank)

    monkeypatch.setattr("app.services.rag_chain.embed_service.embed_query", lambda q: [0.1, 0.2])

    monkeypatch.setattr(

        "app.services.rag_chain.vector_service.semantic_search",

        lambda *args, **kwargs: [SAMPLE_CHUNK],

    )

    monkeypatch.setattr("app.services.rag_chain.bm25_service.load_index", lambda ws: None)

@pytest.mark.asyncio

async def test_run_qa_uses_gemini_when_configured(mock_retrieval, monkeypatch):

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    get_settings.cache_clear()

    with patch(

        "app.services.rag_chain.grounded_answer",

        return_value="Goa trip was discussed in March.",

    ) as grounded:

        result = await qa_graph.run_qa("ws-1", "When was Goa trip planned?")



    assert result.status == "answered"

    assert result.answer == "Goa trip was discussed in March."

    grounded.assert_called_once()





@pytest.mark.asyncio

async def test_run_qa_requires_gemini(mock_retrieval, monkeypatch):

    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    get_settings.cache_clear()

    with pytest.raises(gemini_service.GeminiNotConfiguredError):

        await qa_graph.run_qa("ws-1", "When was Goa trip planned?")

