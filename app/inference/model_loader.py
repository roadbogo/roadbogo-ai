from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.domain.model import ModelCode, ModelSpec, ServingModelConfig


class ModelLoaderError(RuntimeError):
    """?? ?? ?? ?? ??? ???? ? ?? ? ???? ??."""


class InferenceModel(Protocol):
    names: dict[int, str]

    def predict(self, **kwargs: Any) -> object:
        """?? ??? ????."""


ModelFactory = Callable[[Path, str], InferenceModel]
DeviceValidator = Callable[[str], None]
WarmupRunner = Callable[
    [InferenceModel, ModelSpec, str, bool, int],
    None,
]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    spec: ModelSpec
    file_path: Path
    model: InferenceModel
    device: str
    half_precision: bool


class ModelLoader:
    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        device_validator: DeviceValidator | None = None,
        warmup_runner: WarmupRunner | None = None,
    ) -> None:
        self._model_factory = (
            model_factory
            if model_factory is not None
            else self._create_ultralytics_model
        )
        self._device_validator = (
            device_validator
            if device_validator is not None
            else self._validate_inference_device
        )
        self._warmup_runner = (
            warmup_runner
            if warmup_runner is not None
            else self._warmup_model
        )

    def load(
        self,
        config: ServingModelConfig,
        model_paths: Mapping[ModelCode, Path],
    ) -> dict[ModelCode, LoadedModel]:
        self._validate_model_paths(
            config=config,
            model_paths=model_paths,
        )

        execution = config.execution
        self._device_validator(execution.device)

        loaded_models: dict[ModelCode, LoadedModel] = {}

        for spec in config.get_enabled_models():
            model_path = model_paths[spec.model_code]

            try:
                model = self._model_factory(
                    model_path,
                    spec.task,
                )

                self._validate_class_names(
                    model=model,
                    spec=spec,
                )

                self._warmup_runner(
                    model,
                    spec,
                    execution.device,
                    execution.half_precision,
                    execution.batch_size,
                )
            except Exception as error:
                loaded_models.clear()

                raise ModelLoaderError(
                    "Failed to initialize inference model "
                    f"{spec.model_code.value}: {error}"
                ) from error

            loaded_models[spec.model_code] = LoadedModel(
                spec=spec,
                file_path=model_path,
                model=model,
                device=execution.device,
                half_precision=execution.half_precision,
            )

        return loaded_models

    @staticmethod
    def _validate_model_paths(
        config: ServingModelConfig,
        model_paths: Mapping[ModelCode, Path],
    ) -> None:
        enabled_codes = {
            model.model_code
            for model in config.get_enabled_models()
        }
        path_codes = set(model_paths)

        missing_codes = enabled_codes - path_codes
        unexpected_codes = path_codes - enabled_codes

        if missing_codes or unexpected_codes:
            missing = sorted(
                code.value
                for code in missing_codes
            )
            unexpected = sorted(
                code.value
                for code in unexpected_codes
            )

            raise ModelLoaderError(
                "Model loader paths do not match enabled models. "
                f"missing={missing}, unexpected={unexpected}"
            )

    @staticmethod
    def _validate_class_names(
        model: InferenceModel,
        spec: ModelSpec,
    ) -> None:
        expected_names = {
            class_id: class_spec.raw_name
            for class_id, class_spec in spec.classes.items()
        }
        actual_names = {
            int(class_id): class_name
            for class_id, class_name in model.names.items()
        }

        if actual_names != expected_names:
            raise ModelLoaderError(
                "Model class metadata does not match configuration. "
                f"expected={expected_names}, actual={actual_names}"
            )

    @staticmethod
    def _create_ultralytics_model(
        model_path: Path,
        task: str,
    ) -> InferenceModel:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ModelLoaderError(
                "Ultralytics is not installed. "
                "Install the project inference dependencies."
            ) from error

        return YOLO(
            str(model_path),
            task=task,
        )

    @staticmethod
    def _validate_inference_device(device: str) -> None:
        try:
            import torch
        except ImportError as error:
            raise ModelLoaderError(
                "PyTorch is not installed."
            ) from error

        if device == "cpu":
            return

        if device == "cuda":
            device_index = 0
        elif device.startswith("cuda:"):
            try:
                device_index = int(
                    device.split(":", maxsplit=1)[1]
                )
            except ValueError as error:
                raise ModelLoaderError(
                    f"Invalid CUDA device: {device}"
                ) from error
        else:
            raise ModelLoaderError(
                f"Unsupported inference device: {device}"
            )

        if not torch.cuda.is_available():
            raise ModelLoaderError(
                f"CUDA is not available for requested device: {device}"
            )

        device_count = torch.cuda.device_count()

        if device_index < 0 or device_index >= device_count:
            raise ModelLoaderError(
                "Requested CUDA device does not exist. "
                f"device={device}, device_count={device_count}"
            )

        torch.cuda.set_device(
            torch.device(f"cuda:{device_index}")
        )

    @staticmethod
    def _warmup_model(
        model: InferenceModel,
        spec: ModelSpec,
        device: str,
        half_precision: bool,
        batch_size: int,
    ) -> None:
        try:
            import numpy as np
            import torch
        except ImportError as error:
            raise ModelLoaderError(
                "Model warmup dependencies are not installed."
            ) from error

        image = np.zeros(
            (spec.input_size, spec.input_size, 3),
            dtype=np.uint8,
        )

        prediction_options: dict[str, object] = {
            "source": image,
            "imgsz": spec.input_size,
            "device": device,
            "batch": batch_size,
            "verbose": False,
        }

        if half_precision:
            prediction_options["quantize"] = 16

        results = model.predict(**prediction_options)

        if not results:
            raise ModelLoaderError(
                f"Model warmup returned no result: {spec.model_code.value}"
            )

        if device.startswith("cuda"):
            torch.cuda.synchronize(
                torch.device(device)
            )
