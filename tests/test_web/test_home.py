def test_home_page_renders_nav_links(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Check Message" in response.text
    assert "Run a benchmark" in response.text
    assert "Manage guardrails" in response.text


def test_docs_are_disabled(client):
    response = client.get("/docs")
    assert response.status_code == 404
