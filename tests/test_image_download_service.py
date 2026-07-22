from urllib.error import URLError

import pytest

from app.services.image_download_service import (
    ImageDownloadInputError,
    ImageDownloadService,
    ImageDownloadUnavailableError,
)


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_type: str = "image/png",
        content_length: str | None = None,
        status: int = 200,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = {
            "Content-Type": content_type,
        }

        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        _exc_type,
        _exc_value,
        _traceback,
    ) -> None:
        return None

    def read(self, size: int) -> bytes:
        return self.body[:size]


def test_image_download_service_downloads_expected_image() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        captured["accept"] = request.get_header("Accept")
        captured["timeout"] = timeout

        return FakeResponse(
            b"image-bytes",
            content_length="11",
        )

    result = ImageDownloadService(
        timeout_seconds=3.5,
        opener=opener,
    ).download(
        "https://storage.example/frame.png",
        expected_content_type="image/png",
    )

    assert result == b"image-bytes"
    assert captured == {
        "url": "https://storage.example/frame.png",
        "accept": "image/png",
        "timeout": 3.5,
    }


def test_image_download_service_rejects_mime_mismatch() -> None:
    service = ImageDownloadService(
        opener=lambda *_args, **_kwargs: FakeResponse(
            b"image",
            content_type="image/jpeg",
        )
    )

    with pytest.raises(
        ImageDownloadInputError,
        match="does not match",
    ):
        service.download(
            "https://storage.example/frame.png",
            expected_content_type="image/png",
        )


def test_image_download_service_rejects_content_length() -> None:
    service = ImageDownloadService(
        max_image_bytes=4,
        opener=lambda *_args, **_kwargs: FakeResponse(
            b"1234",
            content_length="5",
        ),
    )

    with pytest.raises(
        ImageDownloadInputError,
        match="maximum size",
    ):
        service.download(
            "https://storage.example/frame.png",
            expected_content_type="image/png",
        )


def test_image_download_service_rejects_streamed_oversize() -> None:
    service = ImageDownloadService(
        max_image_bytes=4,
        opener=lambda *_args, **_kwargs: FakeResponse(
            b"12345",
        ),
    )

    with pytest.raises(
        ImageDownloadInputError,
        match="maximum size",
    ):
        service.download(
            "https://storage.example/frame.png",
            expected_content_type="image/png",
        )


def test_image_download_service_rejects_invalid_url() -> None:
    service = ImageDownloadService()

    with pytest.raises(
        ImageDownloadInputError,
        match="HTTP or HTTPS",
    ):
        service.download(
            "file:///tmp/frame.png",
            expected_content_type="image/png",
        )


def test_image_download_service_wraps_connection_error() -> None:
    def opener(*_args, **_kwargs):
        raise URLError("connection refused")

    service = ImageDownloadService(opener=opener)

    with pytest.raises(
        ImageDownloadUnavailableError,
        match="download failed",
    ):
        service.download(
            "https://storage.example/frame.png",
            expected_content_type="image/png",
        )


def test_image_download_service_rejects_error_status() -> None:
    service = ImageDownloadService(
        opener=lambda *_args, **_kwargs: FakeResponse(
            b"error",
            status=503,
        )
    )

    with pytest.raises(
        ImageDownloadUnavailableError,
        match="non-success status",
    ):
        service.download(
            "https://storage.example/frame.png",
            expected_content_type="image/png",
        )
