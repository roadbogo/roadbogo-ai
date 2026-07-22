from collections.abc import Callable
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.services.detection_service import MAX_IMAGE_BYTES


ALLOWED_IMAGE_CONTENT_TYPES = frozenset({
    "image/jpeg",
    "image/png",
})


class ImageDownloadServiceError(RuntimeError):
    """Base error raised while downloading an inference image."""


class ImageDownloadInputError(ImageDownloadServiceError):
    """The downloaded resource cannot be used as an inference image."""


class ImageDownloadUnavailableError(ImageDownloadServiceError):
    """The remote image resource could not be retrieved."""


class ImageDownloadService:
    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        max_image_bytes: int = MAX_IMAGE_BYTES,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        if max_image_bytes < 1:
            raise ValueError(
                "max_image_bytes must be at least one."
            )

        self._timeout_seconds = timeout_seconds
        self._max_image_bytes = max_image_bytes
        self._opener = opener

    def download(
        self,
        url: str,
        *,
        expected_content_type: str,
    ) -> bytes:
        normalized_content_type = (
            expected_content_type
            .split(";", 1)[0]
            .strip()
            .lower()
        )

        if normalized_content_type not in (
            ALLOWED_IMAGE_CONTENT_TYPES
        ):
            raise ImageDownloadInputError(
                "Expected image MIME type is not supported."
            )

        parsed_url = urlsplit(url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
        ):
            raise ImageDownloadInputError(
                "Image download URL must use HTTP or HTTPS."
            )

        request = Request(
            url,
            method="GET",
            headers={
                "Accept": normalized_content_type,
                "User-Agent": "roadbogo-ai/0.1",
            },
        )

        try:
            with self._opener(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                status_code = getattr(
                    response,
                    "status",
                    200,
                )

                if (
                    status_code < 200
                    or status_code >= 300
                ):
                    raise ImageDownloadUnavailableError(
                        "Image download returned a non-success status."
                    )

                response_content_type = self._get_content_type(
                    response
                )

                if (
                    response_content_type
                    not in ALLOWED_IMAGE_CONTENT_TYPES
                ):
                    raise ImageDownloadInputError(
                        "Downloaded resource is not a supported image."
                    )

                if (
                    response_content_type
                    != normalized_content_type
                ):
                    raise ImageDownloadInputError(
                        "Downloaded image MIME type does not match "
                        "the requested MIME type."
                    )

                self._validate_content_length(response)

                image_bytes = response.read(
                    self._max_image_bytes + 1
                )
        except ImageDownloadServiceError:
            raise
        except (URLError, OSError) as error:
            raise ImageDownloadUnavailableError(
                "Image download failed."
            ) from error

        if not image_bytes:
            raise ImageDownloadInputError(
                "Downloaded image is empty."
            )

        if len(image_bytes) > self._max_image_bytes:
            raise ImageDownloadInputError(
                "Downloaded image exceeds the maximum size."
            )

        return image_bytes

    def _get_content_type(
        self,
        response: Any,
    ) -> str:
        content_type = response.headers.get(
            "Content-Type",
            "",
        )

        return (
            content_type
            .split(";", 1)[0]
            .strip()
            .lower()
        )

    def _validate_content_length(
        self,
        response: Any,
    ) -> None:
        raw_content_length = response.headers.get(
            "Content-Length"
        )

        if raw_content_length is None:
            return

        try:
            content_length = int(raw_content_length)
        except (TypeError, ValueError) as error:
            raise ImageDownloadUnavailableError(
                "Image download returned an invalid Content-Length."
            ) from error

        if content_length < 0:
            raise ImageDownloadUnavailableError(
                "Image download returned an invalid Content-Length."
            )

        if content_length > self._max_image_bytes:
            raise ImageDownloadInputError(
                "Downloaded image exceeds the maximum size."
            )
