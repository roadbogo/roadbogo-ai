from app.core.config import Settings


def test_internal_api_settings_have_contract_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.internal_api_v1_prefix == (
        "/api/internal/v1"
    )
    assert settings.internal_api_key is None


def test_internal_api_key_loads_from_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "INTERNAL_API_KEY",
        "test-internal-secret",
    )

    settings = Settings(_env_file=None)

    assert settings.internal_api_key is not None
    assert (
        settings.internal_api_key.get_secret_value()
        == "test-internal-secret"
    )
    assert str(settings.internal_api_key) == "**********"
