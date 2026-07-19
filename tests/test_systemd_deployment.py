from pathlib import Path


UNIT_PATH = Path(
    "deploy/systemd/roadbogo-ai.service"
)
INSTALL_SCRIPT_PATH = Path(
    "deploy/systemd/install.sh"
)


def test_systemd_unit_uses_production_configuration() -> None:
    unit = UNIT_PATH.read_text(
        encoding="utf-8"
    )

    assert "User=roadbogo" in unit
    assert (
        "WorkingDirectory=/home/roadbogo/roadbogo-ai"
        in unit
    )
    assert "--host 0.0.0.0" in unit
    assert "--port 8010" in unit
    assert "--workers 1" in unit
    assert "--reload" not in unit
    assert "Restart=on-failure" in unit
    assert "WantedBy=multi-user.target" in unit


def test_systemd_install_script_validates_model_files() -> None:
    script = INSTALL_SCRIPT_PATH.read_text(
        encoding="utf-8"
    )

    assert "debris_yolov8s_832.pt" in script
    assert "prohibited_mobility_yolo11m_640.pt" in script
    assert "wildlife_yolo11l_640.pt" in script
    assert "systemctl daemon-reload" in script
    assert 'systemctl enable --now "${SERVICE_NAME}"' in script
