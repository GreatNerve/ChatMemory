from fastapi.testclient import TestClient



from app.main import app



client = TestClient(app)





def test_health():

    r = client.get("/api/v1/health")

    assert r.status_code == 200

    body = r.json()

    assert "status" in body

    assert "geminiConfigured" in body

    assert "embedReady" in body

    assert "dataRootWritable" in body





def test_list_workspaces_empty():

    r = client.get("/api/v1/workspaces")

    assert r.status_code == 200

    assert r.json()["workspaces"] == [] or isinstance(r.json()["workspaces"], list)


