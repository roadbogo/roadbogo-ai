from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.domain.model import ModelCode, ModelSpec, ServingModelConfig
from app.inference.model_config_loader import ModelConfigLoader
from app.inference.model_loader import (
    LoadedModel,
    ModelLoader,
    ModelLoaderError,
)


@dataclass
class FakeModel:
    names: dict[int, str]
    predict_calls: list[dict[str, object]] = field(
        default_factory=list
    )

    def predict(self, **kwargs: object) -> list[object]:
        self.predict_calls.append(dict(kwargs))
        return [object()]


def load_model_config() -> ServingModelConfig:
    return ModelConfigLoader().load()


def create_model_paths(
    config: ServingModelConfig,
) -> dict[ModelCode, Path]:
    return {
        model.model_code: Path(model.file_path)
        for model in config.get_enabled_models()
    }


def expected_names(spec: ModelSpec) -> dict[int, str]:
    return {
        class_id: class_spec.raw_name
        for class_id, class_spec in spec.classes.items()
    }


def test_loader_initializes_all_enabled_models() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    specs_by_path = {
        model_paths[spec.model_code]: spec
        for spec in config.get_enabled_models()
    }

    created_models: list[FakeModel] = []
    validated_devices: list[str] = []
    warmup_calls: list[
        tuple[ModelCode, str, bool, int]
    ] = []

    def model_factory(
        model_path: Path,
        task: str,
    ) -> FakeModel:
        spec = specs_by_path[model_path]

        assert task == spec.task

        model = FakeModel(
            names=expected_names(spec)
        )
        created_models.append(model)
        return model

    def device_validator(device: str) -> None:
        validated_devices.append(device)

    def warmup_runner(
        model: FakeModel,
        spec: ModelSpec,
        device: str,
        half_precision: bool,
        batch_size: int,
    ) -> None:
        assert model.names == expected_names(spec)

        warmup_calls.append(
            (
                spec.model_code,
                device,
                half_precision,
                batch_size,
            )
        )

    loader = ModelLoader(
        model_factory=model_factory,
        device_validator=device_validator,
        warmup_runner=warmup_runner,
    )

    loaded = loader.load(
        config=config,
        model_paths=model_paths,
    )

    assert validated_devices == ["cuda:0"]
    assert len(created_models) == 3
    assert len(loaded) == 3

    assert tuple(loaded) == (
        ModelCode.DEBRIS_DETECTOR,
        ModelCode.PROHIBITED_MOBILITY_DETECTOR,
        ModelCode.WILDLIFE_DETECTOR,
    )

    assert all(
        isinstance(model, LoadedModel)
        for model in loaded.values()
    )
    assert all(
        model.device == "cuda:0"
        for model in loaded.values()
    )
    assert all(
        model.half_precision is True
        for model in loaded.values()
    )

    assert warmup_calls == [
        (
            spec.model_code,
            "cuda:0",
            True,
            1,
        )
        for spec in config.get_enabled_models()
    ]


def test_loader_rejects_class_metadata_mismatch() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    loader = ModelLoader(
        model_factory=lambda model_path, task: FakeModel(
            names={0: "wrong-class"}
        ),
        device_validator=lambda device: None,
        warmup_runner=lambda *args: None,
    )

    with pytest.raises(
        ModelLoaderError,
        match="DEBRIS_DETECTOR",
    ):
        loader.load(
            config=config,
            model_paths=model_paths,
        )


def test_loader_rejects_missing_model_path() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    model_paths.pop(
        ModelCode.WILDLIFE_DETECTOR
    )

    loader = ModelLoader(
        model_factory=lambda model_path, task: FakeModel(
            names={}
        ),
        device_validator=lambda device: None,
        warmup_runner=lambda *args: None,
    )

    with pytest.raises(
        ModelLoaderError,
        match="WILDLIFE_DETECTOR",
    ):
        loader.load(
            config=config,
            model_paths=model_paths,
        )
