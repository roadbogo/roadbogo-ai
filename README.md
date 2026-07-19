# roadbogo-ai

Roadbogo AI 서빙 서버입니다.

## Stack

- Python 3.11+
- FastAPI
- Uvicorn

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8010
```

AI API 문서는 `http://localhost:8010/docs` 에서 확인할 수 있습니다.

## Production Deployment

Ubuntu GPU 서버의 systemd 운영 배포 및 관리 방법은 아래 문서를 참고합니다.

- [systemd 운영 배포 가이드](deploy/systemd/README.md)
