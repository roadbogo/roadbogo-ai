from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.domain.model import ModelCode, ServingModelConfig
from app.inference.model_config_loader import ModelConfigLoader
from app.inference.model_loader import LoadedModel
from app.inference.model_registry import ModelRegistry
from app.services.detection_service import (
    DetectionService,
    DetectionServiceError,
)


@dataclass
class FakeBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


@dataclass
class FakeResult:
    boxes: FakeBoxes | None


@dataclass
class FakeModel:
    names: dict[int, str]
    results: list[FakeResult]
    predict_calls: list[dict[str, object]] = field(
        default_factory=list
    )

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.predict_calls.append(dict(kwargs))
        return self.results


def load_model_config() -> ServingModelConfig:
    return ModelConfigLoader().load()


def create_registry(
    config: ServingModelConfig,
) -> tuple[
    ModelRegistry,
    dict[ModelCode, FakeModel],
]:
    model_paths = {
        spec.model_code: Path(spec.file_path)
        for spec in config.get_enabled_models()
    }

    fake_models: dict[ModelCode, FakeModel] = {}

    for spec in config.get_enabled_models():
        if (
            spec.model_code
            is ModelCode.DEBRIS_DETECTOR
        ):
            result = FakeResult(
                boxes=FakeBoxes(
                    xyxy=np.array(
                        [[-5.0, 2.0, 30.0, 15.0]]
                    ),
                    conf=np.array([0.91]),
                    cls=np.array([3.0]),
                )
            )
        elif (
            spec.model_code
            is ModelCode.PROHIBITED_MOBILITY_DETECTOR
        ):
            result = FakeResult(
                boxes=FakeBoxes(
                    xyxy=np.array(
                        [[1.0, 1.0, 8.0, 7.0]]
                    ),
                    conf=np.array([0.82]),
                    cls=np.array([7.0]),
                )
            )
        else:
            result = FakeResult(boxes=None)

        fake_models[spec.model_code] = FakeModel(
            names={
                class_id: class_spec.raw_name
                for class_id, class_spec
                in spec.classes.items()
            },
            results=[result],
        )

    loaded_models = {
        spec.model_code: LoadedModel(
            spec=spec,
            file_path=model_paths[spec.model_code],
            model=fake_models[spec.model_code],
            device=config.execution.device,
            half_precision=(
                config.execution.half_precision
            ),
        )
        for spec in config.get_enabled_models()
    }

    registry = ModelRegistry()
    registry.register(
        config=config,
        model_paths=model_paths,
    )
    registry.attach_loaded_models(
        loaded_models
    )

    return registry, fake_models


def create_test_image() -> bytes:
    image = np.zeros(
        (10, 20, 3),
        dtype=np.uint8,
    )

    success, encoded = cv2.imencode(
        ".png",
        image,
    )

    assert success is True

    return encoded.tobytes()


def test_detection_service_runs_all_loaded_models() -> None:
    config = load_model_config()
    registry, fake_models = create_registry(config)

    response = DetectionService().predict(
        create_test_image(),
        registry,
    )

    assert response.image_width == 20
    assert response.image_height == 10
    assert response.model_count == 3
    assert response.total_detection_count == 2
    assert response.incident_detection_count == 1

    debris_result = response.model_results[0]
    debris_detection = debris_result.detections[0]

    assert (
        debris_result.model_code
        is ModelCode.DEBRIS_DETECTOR
    )
    assert debris_detection.class_code == "TIRE"
    assert debris_detection.incident_target is True
    assert debris_detection.confidence == 0.91
    assert debris_detection.bounding_box.x_min == 0.0
    assert debris_detection.bounding_box.x_max == 20.0
    assert debris_detection.bounding_box.y_max == 10.0

    mobility_result = response.model_results[1]
    mobility_detection = mobility_result.detections[0]

    assert mobility_detection.class_code == "CAR"
    assert mobility_detection.incident_target is False

    for spec in config.get_enabled_models():
        fake_model = fake_models[spec.model_code]

        assert len(fake_model.predict_calls) == 1

        options = fake_model.predict_calls[0]

        assert options["imgsz"] == spec.input_size
        assert (
            options["conf"]
            == spec.confidence_threshold
        )
        assert options["iou"] == spec.iou_threshold
        assert options["device"] == "cuda:0"
        assert options["quantize"] == 16


def test_detection_service_rejects_invalid_image() -> None:
    config = load_model_config()
    registry, _ = create_registry(config)

    with pytest.raises(
        DetectionServiceError,
        match="valid image",
    ):
        DetectionService().predict(
            b"not-an-image",
            registry,
        )


def test_detection_service_rejects_unready_registry() -> None:
    with pytest.raises(
        DetectionServiceError,
        match="not ready",
    ):
        DetectionService().predict(
            create_test_image(),
            ModelRegistry(),
        )


def test_detection_service_rejects_oversized_image() -> None:
    config = load_model_config()
    registry, _ = create_registry(config)

    with pytest.raises(
        DetectionServiceError,
        match="maximum size",
    ):
        DetectionService(
            max_image_bytes=4
        ).predict(
            create_test_image(),
            registry,
        )
