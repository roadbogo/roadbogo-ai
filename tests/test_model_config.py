from pathlib import Path

import pytest

from app.domain.model import ExecutionMode, ModelCode
from app.inference.model_config_loader import (
    ModelConfigError,
    ModelConfigLoader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


def create_loader(
    project_root: Path = PROJECT_ROOT,
) -> ModelConfigLoader:
    return ModelConfigLoader(
        config_path=CONFIG_PATH,
        project_root=project_root,
    )


def test_models_config_loads() -> None:
    config = create_loader().load()

    assert config.version == 1
    assert config.execution.mode is ExecutionMode.SEQUENTIAL
    assert config.execution.device == "cuda:0"
    assert config.execution.batch_size == 1
    assert config.execution.half_precision is True

    assert [model.model_code for model in config.models] == [
        ModelCode.DEBRIS_DETECTOR,
        ModelCode.PROHIBITED_MOBILITY_DETECTOR,
        ModelCode.WILDLIFE_DETECTOR,
    ]


def test_verified_model_classes() -> None:
    config = create_loader().load()

    debris = config.get_model(ModelCode.DEBRIS_DETECTOR)
    prohibited = config.get_model(
        ModelCode.PROHIBITED_MOBILITY_DETECTOR
    )
    wildlife = config.get_model(ModelCode.WILDLIFE_DETECTOR)

    assert debris.input_size == 832
    assert len(debris.classes) == 4
    assert debris.classes[0].raw_name == "bag"
    assert debris.classes[3].class_code == "TIRE"

    assert prohibited.input_size == 640
    assert len(prohibited.classes) == 10
    assert prohibited.classes[7].raw_name == "car"
    assert prohibited.classes[7].incident_target is False

    assert wildlife.input_size == 640
    assert len(wildlife.classes) == 3
    assert wildlife.classes[1].raw_name == "racoon"
    assert wildlife.classes[1].class_code == "RACCOON"


def test_validate_model_files_with_temporary_files(
    tmp_path: Path,
) -> None:
    config = create_loader().load()
    temporary_models = []

    for index, model in enumerate(config.models):
        relative_path = Path("models") / f"test_model_{index}.pt"
        absolute_path = tmp_path / relative_path

        absolute_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        absolute_path.write_bytes(b"test model file")

        temporary_models.append(
            model.model_copy(
                update={
                    "file_path": relative_path.as_posix(),
                }
            )
        )

    temporary_config = config.model_copy(
        update={
            "models": temporary_models,
        }
    )

    loader = create_loader(project_root=tmp_path)
    model_paths = loader.validate_model_files(temporary_config)

    assert len(model_paths) == 3

    for model_path in model_paths.values():
        assert model_path.is_file()
        assert model_path.suffix == ".pt"


def test_validate_model_files_rejects_missing_file(
    tmp_path: Path,
) -> None:
    config = create_loader().load()
    loader = create_loader(project_root=tmp_path)

    with pytest.raises(
        ModelConfigError,
        match="Model file was not found",
    ):
        loader.validate_model_files(config)
