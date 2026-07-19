import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from app.core.runtime import ApplicationRuntimeState
from app.inference.model_config_loader import (
    ModelConfigError,
    ModelConfigLoader,
)
from app.inference.model_loader import (
    ModelLoader,
    ModelLoaderError,
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
    model_loader: ModelLoader | None = None,
) -> LifespanHandler:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime_state = ApplicationRuntimeState()
        model_registry = ModelRegistry()

        app.state.runtime_state = runtime_state
        app.state.model_registry = model_registry

        runtime_state.begin_startup()

        config_loader = (
            model_config_loader
            if model_config_loader is not None
            else ModelConfigLoader()
        )
        inference_loader = (
            model_loader
            if model_loader is not None
            else ModelLoader()
        )

        try:
            model_config = config_loader.load()
            model_paths = config_loader.validate_model_files(
                model_config
            )

            model_registry.register(
                config=model_config,
                model_paths=model_paths,
            )

            runtime_state.set_model_config(model_config)
            runtime_state.set_model_paths(model_paths)
            runtime_state.mark_config_validated()

            logger.info(
                "AI serving configuration validation completed: "
                "%s models",
                model_registry.model_count,
            )

            loaded_models = inference_loader.load(
                config=model_config,
                model_paths=model_paths,
            )

            model_registry.attach_loaded_models(
                loaded_models
            )

            runtime_state.mark_ready()

            logger.info(
                "AI serving inference initialization completed: "
                "%s models",
                model_registry.loaded_model_count,
            )

        except (
            ModelConfigError,
            ModelLoaderError,
            ModelRegistryError,
        ) as error:
            runtime_state.mark_not_ready(error)

            logger.error(
                "AI serving startup initialization failed: %s",
                error,
            )

        yield

        model_registry.clear()

    return lifespan
