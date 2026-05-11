from fastapi.testclient import TestClient

from app.main import create_app


app = create_app()
client = TestClient(app)


def test_ping_endpoint() -> None:
    response = client.get("/ping")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service_version"] == app.state.settings.app_version
    assert payload["service_name"] == app.state.settings.app_name


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == app.state.settings.app_version
    assert payload["configuration"]["provider"] == app.state.settings.provider_summary()
    assert payload["configuration"]["settings"]["llm_provider"] == app.state.settings.resolved_provider("llm")
    assert payload["configuration"]["settings"]["embedding_provider"] == app.state.settings.resolved_provider("embedding")
    assert payload["configuration"]["vectorstore"] == app.state.settings.vectorstore_backend
    assert payload["checks"][0]["name"] == "configuration"
    assert payload["providers"][0]["name"] == f"llm:{app.state.settings.resolved_provider('llm')}"
    assert payload["providers"][1]["name"] == f"embedding:{app.state.settings.resolved_provider('embedding')}"


def test_diagnostics_endpoint() -> None:
    response = client.get("/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["providers"][0]["provider"] == f"llm:{app.state.settings.resolved_provider('llm')}"
    assert payload["providers"][1]["provider"] == f"embedding:{app.state.settings.resolved_provider('embedding')}"
    assert payload["vectorstores"][0]["backend"] == app.state.settings.vectorstore_backend
    assert payload["summary"]["total_checks"] >= 3
    assert "bootstrap" in payload["performance_benchmarks"]


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service_name"] == app.state.settings.app_name
    assert payload["service_version"] == app.state.settings.app_version
    assert payload["provider"] == app.state.settings.provider_summary()
    assert payload["vectorstore"] == app.state.settings.vectorstore_backend
    assert payload["collection_period"] == "lifetime"
    assert payload["performance"] == {}
    assert payload["usage"] == {}


def test_debug_config_masks_secrets() -> None:
    response = client.get("/debug/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["app"]["openai_api_key"] != app.state.settings.openai_api_key
    assert payload["app"]["gemini_api_key"] != app.state.settings.gemini_api_key
    assert "*" in payload["app"]["openai_api_key"]
    assert "*" in payload["app"]["gemini_api_key"]
    if app.state.settings.anthropic_api_key is None or app.state.settings.anthropic_api_key == "":
        assert payload["app"]["anthropic_api_key"] in [None, ""]
    else:
        assert payload["app"]["anthropic_api_key"] != app.state.settings.anthropic_api_key
        assert "*" in payload["app"]["anthropic_api_key"]


def test_debug_config_exposes_bootstrap_context() -> None:
    response = client.get("/debug/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["bootstrap"]["service_name"] == app.state.settings.app_name
    assert payload["bootstrap"]["configuration"]["available_providers"] == ["anthropic", "gemini", "mock", "openai"]
    assert payload["app"]["llm_provider"] == app.state.settings.resolved_provider("llm")
    assert payload["app"]["embedding_provider"] == app.state.settings.resolved_provider("embedding")
