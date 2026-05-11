from app.bootstrap import build_service_context
from app.config import Settings


def test_settings_parse_env_values() -> None:
    settings = Settings(
        APP_NAME="custom-app",
        APP_VERSION="2.0.0",
        APP_ENV="test",
        APP_DEBUG=False,
        API_HOST="127.0.0.1",
        API_PORT=9000,
        LOG_LEVEL="DEBUG",
        MOCK_MODE=True,
        PROVIDER_TYPE="claude",
        ANTHROPIC_API_KEY="anthropic-secret-key",
        CLAUDE_MODEL="claude-3-5-sonnet-latest",
        VECTORSTORE_BACKEND="chroma",
        DEFAULT_COLLECTION_NAME="integration-tests",
        ENABLE_CORS=True,
        CORS_ORIGINS="http://localhost:3000,http://localhost:5173",
    )

    assert settings.app_name == "custom-app"
    assert settings.app_version == "2.0.0"
    assert settings.app_env == "test"
    assert settings.mock_mode is True
    assert settings.provider_config() == {
        "api_key": "anthropic-secret-key",
        "model": "claude-3-5-sonnet-latest",
    }
    assert settings.cors_origins == ["http://localhost:3000", "http://localhost:5173"]


def test_two_level_provider_fallback_supports_family_override() -> None:
    settings = Settings(
        PROVIDER="openai",
        LLM_PROVIDER="gemini",
        OPENAI_API_KEY="shared-openai-key",
        OPENAI_MODEL="text-embedding-3-small",
        LLM_GEMINI_API_KEY="gemini-family-key",
        LLM_GEMINI_MODEL="gemini-2.5-flash",
    )

    assert settings.resolved_provider("llm") == "gemini"
    assert settings.resolved_provider("embedding") == "openai"
    assert settings.llm_provider_config() == {
        "api_key": "gemini-family-key",
        "model": "gemini-2.5-flash",
    }
    assert settings.embedding_provider_config() == {
        "api_key": "shared-openai-key",
        "model": "text-embedding-3-small",
    }
    assert settings.provider_setting_sources("llm") == {
        "api_key": "LLM_GEMINI_API_KEY",
        "model": "LLM_GEMINI_MODEL",
    }
    assert settings.provider_setting_sources("embedding") == {
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
    }


def test_mock_provider_config_supports_family_specific_fields() -> None:
    settings = Settings(
        PROVIDER="mock",
        LLM_PROVIDER="mock",
        EMBEDDING_PROVIDER="mock",
        MOCK_MODEL="shared-mock-model",
        MOCK_SEED=11,
        MOCK_LATENCY_MS=0,
        LLM_MOCK_MODEL="mock-llm-template",
        LLM_MOCK_PREFIX="template-",
        EMBEDDING_MOCK_MODEL="mock-embed-template",
        EMBEDDING_MOCK_DIMENSION=64,
    )

    assert settings.llm_provider_config() == {
        "model": "mock-llm-template",
        "seed": 11,
        "latency_ms": 0,
        "prefix": "template-",
    }
    assert settings.embedding_provider_config() == {
        "model": "mock-embed-template",
        "dimension": 64,
        "seed": 11,
        "latency_ms": 0,
    }


def test_build_service_context_accepts_mock_provider_without_api_key() -> None:
    settings = Settings(
        PROVIDER="mock",
        LLM_PROVIDER="mock",
        EMBEDDING_PROVIDER="mock",
        LLM_MOCK_MODEL="mock-llm",
        EMBEDDING_MOCK_MODEL="mock-embed",
        EMBEDDING_MOCK_DIMENSION=32,
    )

    context = build_service_context(settings)

    assert context.provider == "mock"
    assert context.available_providers == ("anthropic", "gemini", "mock", "openai")
    assert context.provider_statuses[0].status == "healthy"
    assert context.provider_statuses[1].status == "healthy"
    assert context.provider_diagnostics[0].provider == "llm:mock"
    assert context.provider_diagnostics[1].provider == "embedding:mock"


def test_cloud_logging_providers_parse_and_normalize() -> None:
    settings = Settings(
        CLOUD_LOGGING_PROVIDERS=" AWS, datadog ,GCP ",
    )

    assert settings.cloud_logging_providers == ["aws", "datadog", "gcp"]


def test_settings_mask_debug_config() -> None:
    settings = Settings(
        OPENAI_API_KEY="abcd1234wxyz",
        GEMINI_API_KEY="gemini-secret",
        ANTHROPIC_API_KEY="anthropic-secret",
    )

    debug_payload = settings.masked_debug_config()

    assert debug_payload["openai_api_key"] == "abcd****wxyz"
    assert debug_payload["gemini_api_key"] == "gemi*****cret"
    assert debug_payload["anthropic_api_key"] == "anth********cret"


def test_build_service_context_wires_operational_fields() -> None:
    settings = Settings(
        APP_NAME="template-service",
        APP_VERSION="1.2.3",
        APP_DEBUG=True,
        MOCK_MODE=False,
        PROVIDER="openai",
        OPENAI_API_KEY="abcd1234wxyz",
        OPENAI_MODEL="text-embedding-3-small",
        VECTORSTORE_BACKEND="chroma",
        DEFAULT_COLLECTION_NAME="documents",
        ENABLE_CORS=True,
    )

    context = build_service_context(settings)

    assert context.service_name == "template-service"
    assert context.service_version == "1.2.3"
    assert context.provider == "openai"
    assert context.available_providers == ("anthropic", "gemini", "mock", "openai")
    assert context.vectorstore == "chroma"
    assert context.available_vectorstores == ("chroma",)
    assert context.masked_secrets["openai_api_key"] == "abcd****wxyz"
    assert len(context.provider_statuses) == 2
    assert context.provider_statuses[0].name == "llm:openai"
    assert context.provider_statuses[1].name == "embedding:openai"
    assert context.vectorstore_statuses[0].name == "chroma"
    assert context.provider_diagnostics[0].provider == "llm:openai"
    assert context.provider_diagnostics[1].provider == "embedding:openai"
    assert context.performance_benchmarks["bootstrap"]["llm_provider_initialized"] is True
