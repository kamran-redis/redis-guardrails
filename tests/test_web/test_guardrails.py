from redis_guardrails.models import Guardrail


def _guardrail_form(**overrides):
    defaults = dict(
        id="g-1", stage="input", category="cat", description="desc",
        examples="hello\nworld", action="BLOCK", match_threshold="0.5",
    )
    defaults.update(overrides)
    return defaults


def test_list_filters_by_stage(client, service):
    service.add_guardrail(Guardrail(id="in-1", stage="input", category="cat", description="d", examples=["a"], action="BLOCK", match_threshold=0.5))
    service.add_guardrail(Guardrail(id="out-1", stage="output", category="cat", description="d", examples=["a"], action="BLOCK", match_threshold=0.5))

    response = client.get("/guardrails", params={"stage": "input"})
    assert "in-1" in response.text
    assert "out-1" not in response.text


def test_new_guardrail_form_renders(client):
    response = client.get("/guardrails/new")
    assert response.status_code == 200
    assert "New Guardrail" in response.text


def test_new_guardrail_form_stage_datalist_is_never_empty(client):
    response = client.get("/guardrails/new")
    assert response.status_code == 200
    assert "input" in response.text
    assert "output" in response.text


def test_create_success_redirects_to_detail(client):
    response = client.post("/guardrails/new", data=_guardrail_form(), follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/guardrails/g-1"

    detail = client.get("/guardrails/g-1")
    assert detail.status_code == 200
    assert "g-1" in detail.text
    assert "hello" in detail.text
    assert "world" in detail.text


def test_create_duplicate_id_shows_inline_error(client):
    client.post("/guardrails/new", data=_guardrail_form())
    response = client.post("/guardrails/new", data=_guardrail_form())
    assert response.status_code == 400
    assert "g-1" in response.text


def test_create_invalid_threshold_shows_inline_error(client):
    response = client.post("/guardrails/new", data=_guardrail_form(match_threshold="0"))
    assert response.status_code == 400


def test_detail_missing_guardrail_returns_404(client):
    response = client.get("/guardrails/does-not-exist")
    assert response.status_code == 404


def test_edit_form_prefills_existing_values(client):
    client.post("/guardrails/new", data=_guardrail_form())
    response = client.get("/guardrails/g-1/edit")
    assert response.status_code == 200
    assert "cat" in response.text


def test_edit_updates_guardrail(client):
    client.post("/guardrails/new", data=_guardrail_form())
    response = client.post(
        "/guardrails/g-1/edit",
        data=_guardrail_form(category="new-cat"),
        follow_redirects=False,
    )
    assert response.status_code == 303
    detail = client.get("/guardrails/g-1")
    assert "new-cat" in detail.text


def test_edit_forces_id_from_url_not_form(client):
    client.post("/guardrails/new", data=_guardrail_form())
    client.post("/guardrails/new", data=_guardrail_form(id="g-2"))
    # attempt to rename g-1 to g-2 via the form's id field
    client.post("/guardrails/g-1/edit", data=_guardrail_form(id="g-2", category="hijacked"))
    g1 = client.get("/guardrails/g-1")
    g2 = client.get("/guardrails/g-2")
    assert "hijacked" in g1.text
    assert "hijacked" not in g2.text


def test_delete_redirects_and_guardrail_is_gone(client):
    client.post("/guardrails/new", data=_guardrail_form())
    response = client.post("/guardrails/g-1/delete", follow_redirects=False)
    assert response.status_code == 303
    assert "flash=deleted" in response.headers["location"]

    detail = client.get("/guardrails/g-1")
    assert detail.status_code == 404


def test_delete_missing_guardrail_is_a_no_op_redirect(client):
    response = client.post("/guardrails/does-not-exist/delete", follow_redirects=False)
    assert response.status_code == 303
    assert "flash=not_found" in response.headers["location"]


def test_new_guardrail_form_suggests_existing_stages(client, store):
    store.add(Guardrail(
        id="g-1", stage="extra-stage", category="cat", description="d",
        examples=["ex"], action="BLOCK", match_threshold=0.5,
    ))
    response = client.get("/guardrails/new")
    assert response.status_code == 200
    assert "extra-stage" in response.text


def test_can_create_guardrail_on_a_novel_stage(client):
    response = client.post("/guardrails/new", data={
        "id": "g-novel", "stage": "extra-stage", "category": "cat", "description": "d",
        "examples": "an example", "action": "BLOCK", "match_threshold": "0.5",
    }, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/guardrails/g-novel"


def test_list_page_shows_a_tab_for_each_stage_actually_present(client, store):
    store.add(Guardrail(
        id="g-1", stage="extra-stage", category="cat", description="d",
        examples=["ex"], action="BLOCK", match_threshold=0.5,
    ))
    response = client.get("/guardrails")
    assert response.status_code == 200
    assert "extra-stage" in response.text
