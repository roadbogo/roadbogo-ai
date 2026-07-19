import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from app.core.runtime import ApplicationRuntimeState
from app.inference.model_config_loader import (
    ModelConfigError,
    ModelConfigLoader,
)
from app.inference.model_registry import (
    ModelRegistry,
    ModelRegistryError,
)


logger = logging.getLogger(__name__)

LifespanHandler = Callable[
    [FastAPI],
    AbstractAsyncContextManager[None],
]


def create_lifespan(
    model_config_loader: ModelConfigLoader | None = None,
) -> LifespanHandler:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime_state = ApplicationRuntimeState()
        model_registry = ModelRegistry()

        app.state.runtime_state = runtime_state
        app.state.model_registry = model_registry

        runtime_state.begin_startup()

        loader = (
            model_config_loader
            if model_config_loader is not None
            else ModelConfigLoader()
        )

        try:
            model_config = loader.load()
            model_paths = loader.validate_model_files(model_config)

            model_registry.register(
                config=model_config,
                model_paths=model_paths,
            )

            runtime_state.set_model_config(model_config)
            runtime_state.set_model_paths(model_paths)
            runtime_state.mark_config_validated()

            logger.info(
                "AI serving configuration validation completed: %s models",
                model_registry.model_count,
            )

        except (ModelConfigError, ModelRegistryError) as error:
            model_registry.clear()
            runtime_state.mark_not_ready(error)

            logger.error(
                "AI serving startup validation failed: %s",
                error,
            )

        yield

        model_registry.clear()

    return lifespan