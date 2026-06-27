import os

# Avoid loading ~1–2 GB embed model during pytest (integration tests use TestClient lifespan).
os.environ.setdefault("CHATMEMORY_SKIP_EMBED_WARMUP", "1")
