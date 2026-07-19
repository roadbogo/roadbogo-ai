# Roadbogo AI systemd 운영 배포

Roadbogo AI FastAPI 추론 서버를 Ubuntu GPU 서버에서 systemd 서비스로 실행하기 위한 설정입니다.

## 운영 환경

- 프로젝트 경로: `/home/roadbogo/roadbogo-ai`
- 가상환경: `/home/roadbogo/roadbogo-ai/.venv`
- 서비스 이름: `roadbogo-ai.service`
- 운영 포트: `8010`
- Uvicorn worker: `1`

> 모델 중복 로딩과 GPU 메모리 증가를 방지하기 위해 worker 수는 반드시 1로 유지합니다.

## 최초 설치

프로젝트 루트에서 설치 스크립트를 실행합니다.

~~~bash
cd /home/roadbogo/roadbogo-ai
sudo bash deploy/systemd/install.sh
~~~

설치 스크립트는 다음 작업을 수행합니다.

1. Python 가상환경 존재 여부 확인
2. 필수 모델 파일 3개 존재 여부 확인
3. Unit 파일을 `/etc/systemd/system/roadbogo-ai.service`에 설치
4. systemd 설정 다시 읽기
5. 서비스 자동 시작 활성화 및 즉시 실행

## 서비스 상태 확인

~~~bash
sudo systemctl status roadbogo-ai.service
sudo systemctl is-enabled roadbogo-ai.service
sudo systemctl is-active roadbogo-ai.service
~~~

정상 상태에서는 `enabled`, `active`가 출력됩니다.

## Readiness 확인

~~~bash
curl -sS http://127.0.0.1:8010/health/ready
~~~

모델 3개가 정상적으로 로딩되면 HTTP 200 응답과 함께 `loaded_model_count`가 `3`으로 표시됩니다.

## 로그 확인

최근 로그:

~~~bash
sudo journalctl -u roadbogo-ai.service -n 100 --no-pager
~~~

실시간 로그:

~~~bash
sudo journalctl -u roadbogo-ai.service -f
~~~

## 서비스 제어

시작:

~~~bash
sudo systemctl start roadbogo-ai.service
~~~

중지:

~~~bash
sudo systemctl stop roadbogo-ai.service
~~~

재시작:

~~~bash
sudo systemctl restart roadbogo-ai.service
~~~

## 코드 배포 후 반영

~~~bash
cd /home/roadbogo/roadbogo-ai
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart roadbogo-ai.service
curl -sS http://127.0.0.1:8010/health/ready
~~~

모델 로딩이 완료되기 전까지 readiness 요청이 잠시 실패할 수 있습니다.

서비스 로그에서 모델 3개가 모두 로딩됐는지 확인합니다.

## Unit 파일 변경 후 반영

`deploy/systemd/roadbogo-ai.service`를 수정했다면 설치 스크립트를 다시 실행합니다.

~~~bash
cd /home/roadbogo/roadbogo-ai
sudo bash deploy/systemd/install.sh
sudo systemctl status roadbogo-ai.service
~~~

## 백엔드 연동 범위

- 본 배포 구성은 현재 AI Serving 애플리케이션의 실행, 자동 재시작 및 부팅 시 자동 시작을 담당하며 API 계약은 변경하지 않습니다.
- 현재 제공 API는 `POST /api/v1/inference/`이며 JPEG·PNG multipart 업로드 방식입니다.
- 1차 MVP AI-06의 최종 연동 목표는 `POST /api/internal/v1/inferences` JSON 요청과 내부 서버 인증 및 요청 추적 헤더를 사용하는 방식입니다.
- AI-06 내부 API 구현과 백엔드 통합 테스트는 계약 확정 후 별도의 기능 브랜치와 Pull Request에서 진행합니다.
- 그전까지 현재 이미지 업로드 API를 백엔드 최종 연동 계약으로 간주하지 않습니다.

## 주의사항

- `.env`, 비밀번호, 모델 가중치 파일은 Git에 포함하지 않습니다.
- 운영 Uvicorn worker는 반드시 `1`로 유지합니다.
- 모델 파일 3개가 모두 준비된 상태에서 서비스를 시작합니다.
- 장애 발생 시 `journalctl` 로그와 `/health/ready` 응답을 함께 확인합니다.
