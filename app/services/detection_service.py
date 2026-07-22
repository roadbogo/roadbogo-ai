from collections.abc import Iterable
from threading import Lock
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from app.domain.model import ModelCode
from app.inference.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    RegisteredModel,
)
from app.schemas.detection import (
    BoundingBox,
    Detection,
    ImageInferenceResponse,
    ModelDetectionResult,
)


MAX_IMAGE_BYTES = 10 * 1024 * 1024


class DetectionServiceError(RuntimeError):
    """???? ?????? ?? ??? ??? ? ?? ? ????."""


class DetectionInputError(DetectionServiceError):
    """???? ??? ???? ???? ?? ? ????."""


class DetectionUnavailableError(DetectionServiceError):
    """?? ??? ??? ??? ?? ??? ? ????."""


class DetectionService:
    def __init__(
        self,
        *,
        max_image_bytes: int = MAX_IMAGE_BYTES,
    ) -> None:
        self._max_image_bytes = max_image_bytes
        self._inference_lock = Lock()

    def predict(
        self,
        image_bytes: bytes,
        model_registry: ModelRegistry,
        *,
        model_codes: Iterable[ModelCode] | None = None,
    ) -> ImageInferenceResponse:
        image = self._decode_image(image_bytes)

        if not model_registry.is_loaded:
            raise DetectionUnavailableError(
                "Inference models are not ready."
            )

        registered_models = self._resolve_registered_models(
            model_registry,
            model_codes,
        )
        image_height, image_width = image.shape[:2]
        model_results: list[ModelDetectionResult] = []

        with self._inference_lock:
            for registered_model in registered_models:
                model_results.append(
                    self._run_model(
                        image=image,
                        image_width=image_width,
                        image_height=image_height,
                        registered_model=registered_model,
                    )
                )

        total_detection_count = sum(
            result.detection_count
            for result in model_results
        )
        incident_detection_count = sum(
            detection.incident_target
            for result in model_results
            for detection in result.detections
        )

        return ImageInferenceResponse(
            image_width=image_width,
            image_height=image_height,
            model_count=len(model_results),
            total_detection_count=total_detection_count,
            incident_detection_count=incident_detection_count,
            model_results=model_results,
        )

    def _resolve_registered_models(
        self,
        model_registry: ModelRegistry,
        model_codes: Iterable[ModelCode] | None,
    ) -> tuple[RegisteredModel, ...]:
        if model_codes is None:
            return model_registry.list_models()

        requested_codes = tuple(model_codes)

        if not requested_codes:
            raise DetectionInputError(
                "At least one inference model must be requested."
            )

        if len(requested_codes) != len(set(requested_codes)):
            raise DetectionInputError(
                "Inference model codes must not contain duplicates."
            )

        try:
            registered_models = tuple(
                model_registry.get(model_code)
                for model_code in requested_codes
            )
        except ModelRegistryError as error:
            raise DetectionUnavailableError(
                "A requested inference model is not available."
            ) from error

        if not all(
            registered_model.is_loaded
            for registered_model in registered_models
        ):
            raise DetectionUnavailableError(
                "A requested inference model is not loaded."
            )

        return registered_models

    def _decode_image(
        self,
        image_bytes: bytes,
    ) -> NDArray[np.uint8]:
        if not image_bytes:
            raise DetectionInputError(
                "Uploaded image is empty."
            )

        if len(image_bytes) > self._max_image_bytes:
            raise DetectionInputError(
                "Uploaded image exceeds the maximum size of "
                f"{self._max_image_bytes} bytes."
            )

        encoded_image = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )
        image = cv2.imdecode(
            encoded_image,
            cv2.IMREAD_COLOR,
        )

        if image is None or image.size == 0:
            raise DetectionInputError(
                "Uploaded file is not a valid image."
            )

        return image

    def _run_model(
        self,
        *,
        image: NDArray[np.uint8],
        image_width: int,
        image_height: int,
        registered_model: RegisteredModel,
    ) -> ModelDetectionResult:
        model = registered_model.model
        device = registered_model.device

        if model is None or device is None:
            raise DetectionServiceError(
                "Registered model is not loaded: "
                f"{registered_model.spec.model_code.value}"
            )

        spec = registered_model.spec

        prediction_options: dict[str, object] = {
            "source": image,
            "imgsz": spec.input_size,
            "conf": spec.confidence_threshold,
            "iou": spec.iou_threshold,
            "device": device,
            "verbose": False,
        }

        if registered_model.half_precision:
            prediction_options["quantize"] = 16

        try:
            raw_results = model.predict(
                **prediction_options
            )
        except Exception as error:
            raise DetectionServiceError(
                "Inference execution failed for model "
                f"{spec.model_code.value}: {error}"
            ) from error

        if not isinstance(raw_results, (list, tuple)):
            raise DetectionServiceError(
                "Inference model returned an unsupported result "
                f"type: {spec.model_code.value}"
            )

        if not raw_results:
            raise DetectionServiceError(
                "Inference model returned no result: "
                f"{spec.model_code.value}"
            )

        detections = self._parse_detections(
            result=raw_results[0],
            registered_model=registered_model,
            image_width=image_width,
            image_height=image_height,
        )

        return ModelDetectionResult(
            model_code=spec.model_code,
            display_name=spec.display_name,
            detection_count=len(detections),
            detections=detections,
        )

    def _parse_detections(
        self,
        *,
        result: Any,
        registered_model: RegisteredModel,
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            return []

        xyxy_values = self._to_list(
            getattr(boxes, "xyxy", None)
        )
        confidence_values = self._to_list(
            getattr(boxes, "conf", None)
        )
        class_values = self._to_list(
            getattr(boxes, "cls", None)
        )

        if not (
            len(xyxy_values)
            == len(confidence_values)
            == len(class_values)
        ):
            raise DetectionServiceError(
                "Inference result tensor lengths do not match: "
                f"{registered_model.spec.model_code.value}"
            )

        detections: list[Detection] = []

        for bounding_values, confidence, class_id_value in zip(
            xyxy_values,
            confidence_values,
            class_values,
            strict=True,
        ):
            if len(bounding_values) != 4:
                raise DetectionServiceError(
                    "Inference bounding box must contain "
                    "four coordinates."
                )

            class_id = int(class_id_value)
            class_spec = registered_model.spec.classes.get(
                class_id
            )

            if class_spec is None:
                raise DetectionServiceError(
                    "Inference returned an unknown class ID. "
                    f"model={registered_model.spec.model_code.value}, "
                    f"class_id={class_id}"
                )

            x_min, y_min, x_max, y_max = (
                float(value)
                for value in bounding_values
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    raw_name=class_spec.raw_name,
                    class_code=class_spec.class_code,
                    incident_target=(
                        class_spec.incident_target
                    ),
                    confidence=round(
                        float(confidence),
                        6,
                    ),
                    bounding_box=BoundingBox(
                        x_min=round(
                            min(
                                max(x_min, 0.0),
                                float(image_width),
                            ),
                            3,
                        ),
                        y_min=round(
                            min(
                                max(y_min, 0.0),
                                float(image_height),
                            ),
                            3,
                        ),
                        x_max=round(
                            min(
                                max(x_max, 0.0),
                                float(image_width),
                            ),
                            3,
                        ),
                        y_max=round(
                            min(
                                max(y_max, 0.0),
                                float(image_height),
                            ),
                            3,
                        ),
                    ),
                )
            )

        return detections

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []

        detach = getattr(value, "detach", None)

        if callable(detach):
            value = detach()

        cpu = getattr(value, "cpu", None)

        if callable(cpu):
            value = cpu()

        tolist = getattr(value, "tolist", None)

        if not callable(tolist):
            raise DetectionServiceError(
                "Inference result tensor cannot be converted "
                "to a list."
            )

        result = tolist()

        if not isinstance(result, list):
            raise DetectionServiceError(
                "Inference result tensor conversion did not "
                "return a list."
            )

        return result
