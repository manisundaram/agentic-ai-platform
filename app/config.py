from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ai_service_kit.settings import ServiceSettings, build_two_level_provider_config, parse_csv_list, resolve_provider_setting
from ai_service_kit.utils import mask_secret
from pydantic import AliasChoices, Field, field_validator


class Settings(ServiceSettings):
    app_name: str = Field(default="agentic-ai-platform", alias="APP_NAME")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    file_log_level: str = Field(default="DEBUG", alias="FILE_LOG_LEVEL")
    error_file_level: str = Field(default="ERROR", alias="ERROR_FILE_LEVEL")
    log_structured: bool = Field(default=True, alias="LOG_STRUCTURED")
    log_console: bool = Field(default=True, alias="LOG_CONSOLE")
    log_dir: str = Field(default="./logs", alias="LOG_DIR")
    log_max_file_size: int = Field(default=10485760, alias="LOG_MAX_FILE_SIZE")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")
    cloud_logging_providers: list[str] = Field(default_factory=list, alias="CLOUD_LOGGING_PROVIDERS")

    aws_logging_level: str = Field(default="ERROR", alias="AWS_LOGGING_LEVEL")
    aws_log_group: str = Field(default="/agentic-ai-platform/production", alias="AWS_LOG_GROUP")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    azure_logging_level: str = Field(default="ERROR", alias="AZURE_LOGGING_LEVEL")
    azure_connection_string: str | None = Field(default=None, alias="AZURE_CONNECTION_STRING")

    gcp_logging_level: str = Field(default="ERROR", alias="GCP_LOGGING_LEVEL")
    gcp_project_id: str | None = Field(default=None, alias="GCP_PROJECT_ID")

    datadog_logging_level: str = Field(default="INFO", alias="DATADOG_LOGGING_LEVEL")
    datadog_api_key: str | None = Field(default=None, alias="DATADOG_API_KEY")

    mock_model: str = Field(default="mock-default", alias="MOCK_MODEL")
    mock_seed: int = Field(default=0, alias="MOCK_SEED")
    mock_latency_ms: int = Field(default=0, alias="MOCK_LATENCY_MS")

    llm_mock_model: str | None = Field(default=None, alias="LLM_MOCK_MODEL")
    llm_mock_seed: int | None = Field(default=None, alias="LLM_MOCK_SEED")
    llm_mock_latency_ms: int | None = Field(default=None, alias="LLM_MOCK_LATENCY_MS")
    llm_mock_prefix: str | None = Field(default=None, alias="LLM_MOCK_PREFIX")
    llm_mock_suffix: str | None = Field(default=None, alias="LLM_MOCK_SUFFIX")

    embedding_mock_model: str | None = Field(default=None, alias="EMBEDDING_MOCK_MODEL")
    embedding_mock_seed: int | None = Field(default=None, alias="EMBEDDING_MOCK_SEED")
    embedding_mock_latency_ms: int | None = Field(default=None, alias="EMBEDDING_MOCK_LATENCY_MS")
    embedding_mock_dimension: int | None = Field(default=None, alias="EMBEDDING_MOCK_DIMENSION")

    provider: str = Field(
        default="openai",
        alias="PROVIDER",
        validation_alias=AliasChoices("PROVIDER", "PROVIDER_TYPE"),
    )
    llm_provider: str | None = Field(default=None, alias="LLM_PROVIDER")
    embedding_provider: str | None = Field(default=None, alias="EMBEDDING_PROVIDER")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_timeout: int | None = Field(default=None, alias="OPENAI_TIMEOUT")

    llm_openai_api_key: str | None = Field(default=None, alias="LLM_OPENAI_API_KEY")
    llm_openai_model: str | None = Field(default=None, alias="LLM_OPENAI_MODEL")
    llm_openai_base_url: str | None = Field(default=None, alias="LLM_OPENAI_BASE_URL")
    llm_openai_timeout: int | None = Field(default=None, alias="LLM_OPENAI_TIMEOUT")

    embedding_openai_api_key: str | None = Field(default=None, alias="EMBEDDING_OPENAI_API_KEY")
    embedding_openai_model: str | None = Field(default=None, alias="EMBEDDING_OPENAI_MODEL")
    embedding_openai_base_url: str | None = Field(default=None, alias="EMBEDDING_OPENAI_BASE_URL")
    embedding_openai_timeout: int | None = Field(default=None, alias="EMBEDDING_OPENAI_TIMEOUT")

    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_base_url: str | None = Field(default=None, alias="GEMINI_BASE_URL")
    gemini_timeout: int | None = Field(default=None, alias="GEMINI_TIMEOUT")

    llm_gemini_api_key: str | None = Field(default=None, alias="LLM_GEMINI_API_KEY")
    llm_gemini_model: str | None = Field(default=None, alias="LLM_GEMINI_MODEL")
    llm_gemini_base_url: str | None = Field(default=None, alias="LLM_GEMINI_BASE_URL")
    llm_gemini_timeout: int | None = Field(default=None, alias="LLM_GEMINI_TIMEOUT")

    embedding_gemini_api_key: str | None = Field(default=None, alias="EMBEDDING_GEMINI_API_KEY")
    embedding_gemini_model: str | None = Field(default=None, alias="EMBEDDING_GEMINI_MODEL")
    embedding_gemini_base_url: str | None = Field(default=None, alias="EMBEDDING_GEMINI_BASE_URL")
    embedding_gemini_timeout: int | None = Field(default=None, alias="EMBEDDING_GEMINI_TIMEOUT")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-5-haiku-latest",
        alias="ANTHROPIC_MODEL",
        validation_alias=AliasChoices("ANTHROPIC_MODEL", "CLAUDE_MODEL"),
    )
    anthropic_base_url: str | None = Field(default=None, alias="ANTHROPIC_BASE_URL")
    anthropic_timeout: int | None = Field(default=None, alias="ANTHROPIC_TIMEOUT")

    llm_anthropic_api_key: str | None = Field(default=None, alias="LLM_ANTHROPIC_API_KEY")
    llm_anthropic_model: str | None = Field(
        default=None,
        alias="LLM_ANTHROPIC_MODEL",
        validation_alias=AliasChoices("LLM_ANTHROPIC_MODEL", "LLM_CLAUDE_MODEL"),
    )
    llm_anthropic_base_url: str | None = Field(default=None, alias="LLM_ANTHROPIC_BASE_URL")
    llm_anthropic_timeout: int | None = Field(default=None, alias="LLM_ANTHROPIC_TIMEOUT")

    embedding_anthropic_api_key: str | None = Field(default=None, alias="EMBEDDING_ANTHROPIC_API_KEY")
    embedding_anthropic_model: str | None = Field(
        default=None,
        alias="EMBEDDING_ANTHROPIC_MODEL",
        validation_alias=AliasChoices("EMBEDDING_ANTHROPIC_MODEL", "EMBEDDING_CLAUDE_MODEL"),
    )
    embedding_anthropic_base_url: str | None = Field(default=None, alias="EMBEDDING_ANTHROPIC_BASE_URL")
    embedding_anthropic_timeout: int | None = Field(default=None, alias="EMBEDDING_ANTHROPIC_TIMEOUT")

    vectorstore_backend: str = Field(default="chroma", alias="VECTORSTORE_BACKEND")
    default_collection_name: str = Field(default="default", alias="DEFAULT_COLLECTION_NAME")

    @field_validator("cloud_logging_providers", mode="before")
    @classmethod
    def parse_cloud_providers(cls, value: str | list[str] | None) -> list[str]:
        return [item.lower() for item in parse_csv_list(value)]

    @field_validator("provider", "llm_provider", "embedding_provider", mode="before")
    @classmethod
    def normalize_provider_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized == "claude":
            normalized = "anthropic"
        return normalized or None

    def _settings_values(self) -> dict[str, Any]:
        return self.model_dump(mode="python", by_alias=True)

    def resolved_provider(self, family: str) -> str:
        family_key = family.strip().lower()
        family_provider = getattr(self, f"{family_key}_provider")
        if family_provider:
            return family_provider
        return self.provider

    @property
    def provider_type(self) -> str:
        return self.provider

    def provider_summary(self) -> str:
        llm_provider = self.resolved_provider("llm")
        embedding_provider = self.resolved_provider("embedding")
        if llm_provider == embedding_provider:
            return llm_provider
        return f"llm:{llm_provider},embedding:{embedding_provider}"

    def provider_fields_for(self, family: str) -> list[str]:
        provider_name = self.resolved_provider(family)
        if provider_name == "mock":
            if family == "llm":
                return ["model", "seed", "latency_ms", "prefix", "suffix"]
            return ["model", "dimension", "seed", "latency_ms"]
        return ["api_key", "model", "base_url", "timeout"]

    def provider_config_for(self, family: str) -> dict[str, Any]:
        return build_two_level_provider_config(
            values=self._settings_values(),
            family=family,
            provider_type=self.resolved_provider(family),
            fields=self.provider_fields_for(family),
        )

    def provider_config(self) -> dict[str, Any]:
        return self.provider_config_for("embedding")

    def llm_provider_config(self) -> dict[str, Any]:
        return self.provider_config_for("llm")

    def embedding_provider_config(self) -> dict[str, Any]:
        return self.provider_config_for("embedding")

    def provider_setting_sources(self, family: str) -> dict[str, str]:
        values = self._settings_values()
        sources: dict[str, str] = {}
        provider_name = self.resolved_provider(family)
        for field_name in ("api_key", "model", "base_url", "timeout"):
            _, source = resolve_provider_setting(
                values=values,
                family=family,
                provider=provider_name,
                field=field_name,
            )
            if source is not None:
                sources[field_name] = source
        return sources

    def masked_provider_config(self, family: str) -> dict[str, Any]:
        config = dict(self.provider_config_for(family))
        if "api_key" in config:
            config["api_key"] = mask_secret(config["api_key"])
        return config

    def masked_secret_fields(self) -> dict[str, str | None]:
        return super().masked_secrets(
            [
                "openai_api_key",
                "llm_openai_api_key",
                "embedding_openai_api_key",
                "gemini_api_key",
                "llm_gemini_api_key",
                "embedding_gemini_api_key",
                "anthropic_api_key",
                "llm_anthropic_api_key",
                "embedding_anthropic_api_key",
                "azure_connection_string",
                "datadog_api_key",
            ]
        )

    def operational_settings(self) -> dict[str, Any]:
        settings = dict(super().operational_settings())
        settings.update(
            {
                "provider": self.provider_summary(),
                "shared_provider": self.provider,
                "llm_provider": self.resolved_provider("llm"),
                "embedding_provider": self.resolved_provider("embedding"),
                "mock_mode": self.mock_mode,
                "vectorstore_backend": self.vectorstore_backend,
                "default_collection_name": self.default_collection_name,
                "cloud_logging_providers": list(self.cloud_logging_providers),
            }
        )
        return settings

    def masked_debug_config(self) -> Mapping[str, Any]:
        masked = self.masked_secret_fields()
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_env": self.app_env,
            "app_debug": self.app_debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "file_log_level": self.file_log_level,
            "error_file_level": self.error_file_level,
            "log_structured": self.log_structured,
            "log_console": self.log_console,
            "log_dir": self.log_dir,
            "cloud_logging_providers": self.cloud_logging_providers,
            "aws_logging_level": self.aws_logging_level,
            "aws_log_group": self.aws_log_group,
            "aws_region": self.aws_region,
            "azure_logging_level": self.azure_logging_level,
            "azure_connection_string": masked["azure_connection_string"],
            "gcp_logging_level": self.gcp_logging_level,
            "gcp_project_id": self.gcp_project_id,
            "datadog_logging_level": self.datadog_logging_level,
            "datadog_api_key": masked["datadog_api_key"],
            "mock_model": self.mock_model,
            "mock_seed": self.mock_seed,
            "mock_latency_ms": self.mock_latency_ms,
            "mock_mode": self.mock_mode,
            "provider": self.provider,
            "provider_summary": self.provider_summary(),
            "llm_provider": self.resolved_provider("llm"),
            "embedding_provider": self.resolved_provider("embedding"),
            "llm_provider_config": self.masked_provider_config("llm"),
            "embedding_provider_config": self.masked_provider_config("embedding"),
            "llm_provider_sources": self.provider_setting_sources("llm"),
            "embedding_provider_sources": self.provider_setting_sources("embedding"),
            "openai_api_key": masked["openai_api_key"],
            "llm_openai_api_key": masked["llm_openai_api_key"],
            "embedding_openai_api_key": masked["embedding_openai_api_key"],
            "gemini_api_key": masked["gemini_api_key"],
            "llm_gemini_api_key": masked["llm_gemini_api_key"],
            "embedding_gemini_api_key": masked["embedding_gemini_api_key"],
            "anthropic_api_key": masked["anthropic_api_key"],
            "llm_anthropic_api_key": masked["llm_anthropic_api_key"],
            "embedding_anthropic_api_key": masked["embedding_anthropic_api_key"],
            "vectorstore_backend": self.vectorstore_backend,
            "default_collection_name": self.default_collection_name,
            "enable_cors": self.enable_cors,
            "cors_origins": list(self.cors_origins),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        settings = Settings()
        # Note: Logger may not be configured yet during initial setup
        return settings
    except Exception as e:
        # Use print since Logger might not be configured yet
        print(f"Failed to load settings: {e}")
        raise
