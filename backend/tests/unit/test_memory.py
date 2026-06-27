from unittest.mock import MagicMock, patch

from app.core.memory import cleanup_training_artifacts, release_ram


def test_cleanup_training_artifacts_drops_refs():

    obj = MagicMock()

    cleanup_training_artifacts(obj)

    # Should not raise; gc runs after delete.


def test_prepare_vram_unloads_embed(monkeypatch):

    from app.core import memory

    embed = MagicMock()

    monkeypatch.setattr("app.services.embed.unload_embed_model", embed.unload_embed_model)

    with patch.object(memory, "release_all_memory") as release:
        memory.prepare_vram_for_large_model()

        embed.unload_embed_model.assert_called_once()

        release.assert_called_once()

    release_ram()
