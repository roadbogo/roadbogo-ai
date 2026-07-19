#!/usr/bin/env bash

set -euo pipefail

SERVICE_NAME="roadbogo-ai.service"
PROJECT_DIR="/home/roadbogo/roadbogo-ai"
UNIT_SOURCE="${PROJECT_DIR}/deploy/systemd/${SERVICE_NAME}"
UNIT_TARGET="/etc/systemd/system/${SERVICE_NAME}"

required_files=(
  "${PROJECT_DIR}/.venv/bin/python"
  "${PROJECT_DIR}/models/debris/debris_yolov8s_832.pt"
  "${PROJECT_DIR}/models/prohibited-mobility/prohibited_mobility_yolo11m_640.pt"
  "${PROJECT_DIR}/models/wildlife/wildlife_yolo11l_640.pt"
)

for required_file in "${required_files[@]}"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "Required file is missing: ${required_file}" >&2
    exit 1
  fi
done

sudo install   --owner=root   --group=root   --mode=0644   "${UNIT_SOURCE}"   "${UNIT_TARGET}"

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

sudo systemctl   --no-pager   --full   status "${SERVICE_NAME}"
