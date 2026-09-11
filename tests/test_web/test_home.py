def test_home_page_renders_nav_links(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Run a Prompt" in response.text
    assert "Run a Benchmark" in response.text
    assert "Manage Guardrails" in response.text


def test_docs_are_disabled(client):
    response = client.get("/docs")
    assert response.status_code == 404
