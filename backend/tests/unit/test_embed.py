from unittest.mock import MagicMock, patch

from app.core.config import get_settings
from app.services import embed as embed_service


def test_e5_prefixes_for_query_and_passage(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
    get_settings.cache_clear()

    assert embed_service._prefix_for_e5(["hello"], is_query=True) == ["query: hello"]
    assert embed_service._prefix_for_e5(["doc"], is_query=False) == ["passage: doc"]
    get_settings.cache_clear()


def test_no_prefix_for_non_e5_models(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-m3")
    get_settings.cache_clear()

    assert embed_service._prefix_for_e5(["hello"], is_query=True) == ["hello"]
    get_settings.cache_clear()


def test_resolve_embed_device_cpu_forced(monkeypatch):
    monkeypatch.setenv("EMBED_DEVICE", "cpu")
    get_settings.cache_clear()

    assert embed_service.resolve_embed_device() == "cpu"
    assert embed_service.embed_uses_gpu() is False
    get_settings.cache_clear()


def test_resolve_embed_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("EMBED_DEVICE", "auto")
    monkeypatch.setattr(embed_service, "cuda_available", lambda: (False, "no cuda"))
    get_settings.cache_clear()

    assert embed_service.resolve_embed_device() == "cpu"
    assert embed_service.embed_uses_gpu() is False
    get_settings.cache_clear()


def test_resolve_embed_device_auto_uses_cuda(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setenv("EMBED_DEVICE", "auto")
    monkeypatch.setattr(embed_service, "cuda_available", lambda: (True, None))
    mock_torch = MagicMock()
    mock_torch.cuda.current_device.return_value = 0
    monkeypatch.setitem(__import__("sys").modules, "torch", mock_torch)
    get_settings.cache_clear()

    assert embed_service.resolve_embed_device() == "cuda:0"
    assert embed_service.embed_uses_gpu() is True
    get_settings.cache_clear()


def test_warmup_embed_model_skips_when_env_set(monkeypatch):
    monkeypatch.setenv("CHATMEMORY_SKIP_EMBED_WARMUP", "1")
    assert embed_service.warmup_embed_model() is False


def test_warmup_embed_model_loads_once(monkeypatch):
    monkeypatch.delenv("CHATMEMORY_SKIP_EMBED_WARMUP", raising=False)
    embed_service.unload_embed_model()

    def _fake_ensure() -> None:
        embed_service._local_model = MagicMock()

    with patch.object(embed_service, "_ensure_local_model", side_effect=_fake_ensure) as ensure:
        with patch.object(embed_service, "embed_query", return_value=[0.1, 0.2]) as embed:
            assert embed_service.warmup_embed_model() is True
            ensure.assert_called_once()
            embed.assert_called_once_with("warmup")
            assert embed_service.warmup_embed_model() is True
            ensure.assert_called_once()
    embed_service.unload_embed_model()


def test_embed_ready_reflects_loaded_state():
    embed_service.unload_embed_model()
    assert embed_service.embed_ready() is False
    embed_service._local_model = MagicMock()
    try:
        assert embed_service.embed_ready() is True
    finally:
        embed_service.unload_embed_model()
