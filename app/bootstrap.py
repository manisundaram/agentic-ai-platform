from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ai_service_kit.health import (
    BaseHealthCheck,
    CheckResult,
    ComponentKind,
    ComponentStatus,
    HealthStatus,
    NoOpMetricsCollector,
    ProviderDiagnosticsResult,
    ServiceContext,
    VectorStoreDiagnosticsResult,
)
from ai_service_kit.providers import LLMProviderFactory, ProviderFactory

from .config import Settings

TEMPLATE_PROVIDER_NAMES = ("anthropic", "gemini", "mock", "openai")
SUPPORTED_VECTORSTORES = ("chroma",)


@dataclass(slots=True)
class ProviderFamilyRuntime:
    family: str
    name: str
    model: str | None
    configured: bool
    available: bool | None
    initialized: bool
    config_sources: dict[str, str]
    registered_providers: tuple[str, ...]


@dataclass(slots=True)
class VectorStoreRuntime:
    backend: str
    default_collection_name: str
    configured: bool
    supported: bool
    initialized: bool
    collections_count: int


class ConfigurationHealthCheck(BaseHealthCheck):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "configuration"

    async def run(self) -> CheckResult:
        llm_provider = self._settings.resolved_provider("llm")
        embedding_provider = self._settings.resolved_provider("embedding")
        vectorstore_backend = self._settings.vectorstore_backend.strip().lower()
        errors: list[str] = []

        if llm_provider != "mock" and not self._settings.llm_provider_config().get("api_key"):
            errors.append(f"Missing API key for llm provider: {llm_provider}")

        if embedding_provider != "mock" and not self._settings.embedding_provider_config().get("api_key"):
            errors.append(f"Missing API key for embedding provider: {embedding_provider}")

        if vectorstore_backend not in SUPPORTED_VECTORSTORES:
            errors.append(f"Unsupported vector store backend: {vectorstore_backend}")

        if not errors:
            status = HealthStatus.HEALTHY
            summary = "Operational configuration is valid"
        elif any(error.startswith("Unsupported") for error in errors):
            status = HealthStatus.CRITICAL
            summary = "Operational configuration contains unsupported components"
        else:
            status = HealthStatus.DEGRADED
            summary = "Operational configuration is incomplete"

        return CheckResult(
            name=self.name,
            status=status,
            summary=summary,
            details={
                "shared_provider": self._settings.provider,
                "llm_provider": llm_provider,
                "embedding_provider": embedding_provider,
                "vectorstore_backend": vectorstore_backend,
                "available_vectorstores": list(SUPPORTED_VECTORSTORES),
            },
            errors=tuple(errors),
        )


def _build_provider_runtime(
    *,
    family: str,
    provider_name: str,
    config: dict[str, Any],
    registered_providers: tuple[str, ...],
    sources: dict[str, str],
) -> ProviderFamilyRuntime:
    is_registered = provider_name in registered_providers if registered_providers else None
    available = True if provider_name in TEMPLATE_PROVIDER_NAMES else is_registered
    configured = provider_name == "mock" or bool(config.get("api_key"))
    initialized = configured and available is not False

    return ProviderFamilyRuntime(
        family=family,
        name=provider_name,
        model=config.get("model"),
        configured=configured,
        available=available,
        initialized=initialized,
        config_sources=sources,
        registered_providers=registered_providers,
    )


def _build_vectorstore_runtime(settings: Settings) -> VectorStoreRuntime:
    backend = settings.vectorstore_backend.strip().lower()
    supported = backend in SUPPORTED_VECTORSTORES
    configured = supported and bool(settings.default_collection_name)
    initialized = configured

    return VectorStoreRuntime(
        backend=backend,
        default_collection_name=settings.default_collection_name,
        configured=configured,
        supported=supported,
        initialized=initialized,
        collections_count=1 if configured else 0,
    )


def _provider_status(provider_runtime: ProviderFamilyRuntime) -> ComponentStatus:
    if provider_runtime.available is False:
        status = HealthStatus.CRITICAL
        error = f"Provider is not registered in the {provider_runtime.family} factory: {provider_runtime.name}"
    elif not provider_runtime.configured:
        status = HealthStatus.DEGRADED
        error = f"Missing credentials for {provider_runtime.family} provider: {provider_runtime.name}"
    else:
        status = HealthStatus.HEALTHY
        error = None

    return ComponentStatus(
        name=f"{provider_runtime.family}:{provider_runtime.name}",
        kind=ComponentKind.PROVIDER,
        status=status,
        configured=provider_runtime.configured,
        available=provider_runtime.available,
        initialized=provider_runtime.initialized,
        details={
            "family": provider_runtime.family,
            "model": provider_runtime.model,
            "sources": provider_runtime.config_sources,
            "registered_providers": list(provider_runtime.registered_providers),
        },
        error=error,
    )


def _vectorstore_status(vectorstore_runtime: VectorStoreRuntime) -> ComponentStatus:
    if not vectorstore_runtime.supported:
        status = HealthStatus.CRITICAL
        error = f"Unsupported vector store backend: {vectorstore_runtime.backend}"
    elif not vectorstore_runtime.configured:
        status = HealthStatus.DEGRADED
        error = f"Vector store backend is missing required configuration: {vectorstore_runtime.backend}"
    else:
        status = HealthStatus.HEALTHY
        error = None

    return ComponentStatus(
        name=vectorstore_runtime.backend,
        kind=ComponentKind.VECTORSTORE,
        status=status,
        configured=vectorstore_runtime.configured,
        available=vectorstore_runtime.supported,
        initialized=vectorstore_runtime.initialized,
        details={"default_collection_name": vectorstore_runtime.default_collection_name},
        error=error,
    )


def _provider_diagnostics(provider_runtime: ProviderFamilyRuntime) -> ProviderDiagnosticsResult:
    provider_status = _provider_status(provider_runtime)
    return ProviderDiagnosticsResult(
        provider=f"{provider_runtime.family}:{provider_runtime.name}",
        status=provider_status.status,
        configured=provider_runtime.configured,
        available=provider_runtime.available,
        initialized=provider_runtime.initialized,
        models_available=(provider_runtime.model,) if provider_runtime.model else (),
        error=provider_status.error,
        details={
            "family": provider_runtime.family,
            "model": provider_runtime.model,
            "sources": provider_runtime.config_sources,
        },
    )


def _vectorstore_diagnostics(vectorstore_runtime: VectorStoreRuntime) -> VectorStoreDiagnosticsResult:
    vectorstore_status = _vectorstore_status(vectorstore_runtime)
    return VectorStoreDiagnosticsResult(
        backend=vectorstore_runtime.backend,
        status=vectorstore_status.status,
        configured=vectorstore_runtime.configured,
        available=vectorstore_runtime.supported,
        initialized=vectorstore_runtime.initialized,
        collections_count=vectorstore_runtime.collections_count,
        default_collection=vectorstore_runtime.default_collection_name,
        error=vectorstore_status.error,
        details={"default_collection_name": vectorstore_runtime.default_collection_name},
    )


def debug_snapshot(context: ServiceContext) -> dict[str, Any]:
    configuration = context.configuration()
    return {
        "service_name": context.service_name,
        "service_version": context.service_version,
        "configuration": asdict(configuration),
        "metrics_collector": type(context.metrics_collector).__name__,
        "health_checks": [getattr(check, "name", type(check).__name__) for check in context.health_checks],
        "diagnostics_checks": [getattr(check, "name", type(check).__name__) for check in context.diagnostics_checks],
    }


def build_service_context(settings: Settings) -> ServiceContext:
    llm_factory = LLMProviderFactory()
    embedding_factory = ProviderFactory()

    llm_available = tuple(llm_factory.get_available_providers())
    embedding_available = tuple(embedding_factory.get_available_providers())

    provider_runtimes = (
        _build_provider_runtime(
            family="llm",
            provider_name=settings.resolved_provider("llm"),
            config=settings.llm_provider_config(),
            registered_providers=llm_available,
            sources=settings.provider_setting_sources("llm"),
        ),
        _build_provider_runtime(
            family="embedding",
            provider_name=settings.resolved_provider("embedding"),
            config=settings.embedding_provider_config(),
            registered_providers=embedding_available,
            sources=settings.provider_setting_sources("embedding"),
        ),
    )
    vectorstore_runtime = _build_vectorstore_runtime(settings)

    return ServiceContext(
        service_name=settings.app_name,
        service_version=settings.app_version,
        provider=settings.provider_summary(),
        available_providers=tuple(sorted(set(TEMPLATE_PROVIDER_NAMES + llm_available + embedding_available))),
        vectorstore=vectorstore_runtime.backend,
        available_vectorstores=SUPPORTED_VECTORSTORES,
        mock_mode=settings.mock_mode,
        debug_mode=settings.app_debug,
        cors_enabled=settings.enable_cors,
        masked_secrets=settings.masked_secret_fields(),
        settings=settings.operational_settings(),
        metrics_collector=NoOpMetricsCollector(),
        health_checks=(ConfigurationHealthCheck(settings),),
        diagnostics_checks=(ConfigurationHealthCheck(settings),),
        provider_statuses=tuple(_provider_status(runtime) for runtime in provider_runtimes),
        vectorstore_statuses=(_vectorstore_status(vectorstore_runtime),),
        provider_diagnostics=tuple(_provider_diagnostics(runtime) for runtime in provider_runtimes),
        vectorstore_diagnostics=(_vectorstore_diagnostics(vectorstore_runtime),),
        performance_benchmarks={
            "bootstrap": {
                "llm_provider_initialized": provider_runtimes[0].initialized,
                "embedding_provider_initialized": provider_runtimes[1].initialized,
                "vectorstore_initialized": vectorstore_runtime.initialized,
            }
        },
    )
