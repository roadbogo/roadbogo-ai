# 도로보GO AI Serving 서버 API 설계서

> **목적**: FastAPI 기반 AI Serving 서버에서 제공해야 할 기본 API와 책임 범위를 정의한다.  
> **대상 프로젝트**: 도로보GO AI 기반 고속도로 위험 객체 실시간 탐지 및 관제·출동 지원 시스템  
> **기본 기술**: Python, FastAPI, YOLO, OpenCV, CUDA, Docker

---

## 1. AI Serving 서버 역할

AI Serving 서버는 CCTV 영상 또는 이미지에서 객체를 탐지하고, 연속 프레임을 분석하여 위험 판단에 필요한 정보를 서비스 백엔드에 전달한다.

```text
영상 또는 이미지 입력
→ 전처리
→ YOLO 객체 탐지
→ Bounding Box 후처리
→ 도로 ROI 판정
→ 동일 객체 추적
→ 지속시간·반복 횟수 계산
→ 위험도 분석
→ 표준 JSON 결과 반환
```

### 1.1 AI Serving 서버 담당 기능

- YOLO 모델 로딩 및 추론
- 이미지 전처리 및 후처리
- 차량·낙하물·야생동물 탐지
- Bounding Box 및 신뢰도 계산
- OpenCV 기반 도로 ROI 판정
- 연속 프레임 객체 추적
- 객체 지속시간 및 반복 탐지 횟수 계산
- 위험도 및 위험 단계 계산
- 모델 버전과 클래스 정보 제공
- GPU 및 추론 성능 모니터링

### 1.2 서비스 백엔드 담당 기능

다음 기능은 AI Serving 서버가 아니라 Flask 또는 별도의 서비스 백엔드가 담당한다.

- 사용자 인증 및 RBAC
- 사건 생성 및 데이터베이스 저장
- 동일 사건 중복 병합의 최종 처리
- 관제자의 사건 확인 및 판정
- 출동 담당자 배정
- WebSocket 실시간 알림
- 출동 및 조치 상태 관리
- 감사 로그와 처리 이력 저장

---

## 2. 기본 API 목록

| 구분 | Method | Endpoint | 기능 |
|---|---|---|---|
| 프로세스 상태 | `GET` | `/health/live` | FastAPI 프로세스 상태 확인 |
| 추론 준비 상태 | `GET` | `/health/ready` | 모델과 GPU 준비 상태 확인 |
| 활성 모델 정보 | `GET` | `/api/v1/models/active` | 현재 모델 버전, 클래스, 입력 크기 조회 |
| 단일 이미지 추론 | `POST` | `/api/v1/inferences/image` | 이미지 한 장에 대한 객체 탐지 |
| CCTV 프레임 분석 | `POST` | `/api/v1/streams/{cctv_id}/frames` | 연속 프레임 추론, 추적 및 위험 분석 |
| CCTV 분석 상태 초기화 | `DELETE` | `/api/v1/streams/{cctv_id}` | CCTV별 추적 상태 초기화 |
| 모델 재로딩 | `POST` | `/api/v1/admin/models/reload` | 서버 재시작 없이 모델 교체 |
| 서버 메트릭 | `GET` | `/metrics` | 추론 성능, 오류, GPU 상태 확인 |

---

## 3. 상태 확인 API

## 3.1 프로세스 상태 확인

FastAPI 애플리케이션 프로세스가 실행 중인지 확인한다.

```http
GET /health/live
```

### 응답 예시

```json
{
  "status": "ok",
  "service": "roadbogo-ai-serving",
  "timestamp": "2026-07-16T12:10:00+09:00"
}
```

### 설계 원칙

- 프로세스 생존 여부만 확인한다.
- 모델 또는 GPU 상태까지 검사하지 않는다.
- Docker, Kubernetes, systemd 등의 프로세스 상태 점검에 사용한다.

---

## 3.2 추론 준비 상태 확인

모델이 로딩되어 있고 실제 추론 요청을 처리할 수 있는지 확인한다.

```http
GET /health/ready
```

### 응답 예시

```json
{
  "status": "ready",
  "model_loaded": true,
  "device": "cuda:0",
  "gpu_available": true,
  "model_version": "roadbogo-yolo-v1.0.0"
}
```

### 오류 처리

모델 로딩 실패, GPU 초기화 실패 또는 추론 불가능 상태이면 다음 상태 코드를 반환한다.

```text
503 Service Unavailable
```

---

## 4. 활성 모델 정보 API

현재 AI Serving 서버에 로딩된 모델 정보를 조회한다.

```http
GET /api/v1/models/active
```

### 응답 예시

```json
{
  "model_id": "roadbogo-yolo",
  "version": "1.0.0",
  "framework": "ultralytics",
  "input_size": [640, 640],
  "device": "cuda:0",
  "classes": [
    {
      "id": 0,
      "name": "vehicle"
    },
    {
      "id": 1,
      "name": "fallen_object"
    },
    {
      "id": 2,
      "name": "wild_animal"
    }
  ],
  "confidence_threshold": 0.5,
  "iou_threshold": 0.45,
  "loaded_at": "2026-07-16T11:30:00+09:00"
}
```

### 주요 목적

- 프론트엔드와 백엔드의 클래스 하드코딩 방지
- 모델 버전 추적
- 입력 해상도 확인
- 현재 활성 모델과 장치 상태 확인

---

## 5. 단일 이미지 추론 API

이미지 한 장에 대해 객체 탐지를 수행한다.

```http
POST /api/v1/inferences/image
Content-Type: multipart/form-data
```

## 5.1 요청 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `image` | File | O | JPG 또는 PNG 이미지 |
| `request_id` | string | 선택 | 요청 추적용 고유 ID |
| `confidence` | float | 선택 | 탐지 신뢰도 임계값 |
| `iou_threshold` | float | 선택 | NMS IoU 임계값 |
| `include_image` | boolean | 선택 | Bounding Box 결과 이미지 반환 여부 |

## 5.2 응답 예시

```json
{
  "request_id": "REQ-20260716-0001",
  "model": {
    "name": "roadbogo-yolo",
    "version": "1.0.0",
    "input_size": [640, 640]
  },
  "image": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "class_id": 1,
      "class_name": "fallen_object",
      "confidence": 0.87,
      "bbox": {
        "x1": 120,
        "y1": 240,
        "x2": 220,
        "y2": 330
      }
    }
  ],
  "timing": {
    "preprocess_ms": 4.7,
    "inference_ms": 18.4,
    "postprocess_ms": 2.1,
    "total_ms": 25.2
  }
}
```

## 5.3 설계 원칙

단일 이미지 요청에서는 아래 값을 계산하지 않는다.

- `track_id`
- `duration_seconds`
- `repeat_count`

이 값들은 여러 프레임을 연속적으로 분석해야 계산할 수 있다.

---

## 6. CCTV 연속 프레임 분석 API

도로보GO 실제 관제 흐름에서 사용하는 핵심 API이다.

```http
POST /api/v1/streams/{cctv_id}/frames
Content-Type: multipart/form-data
```

## 6.1 요청 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|---:|---|
| `image` | File | O | CCTV 캡처 프레임 |
| `captured_at` | datetime | O | 실제 프레임 촬영 시각 |
| `frame_sequence` | integer | 선택 | 프레임 순번 |
| `road_roi` | JSON | 선택 | 해당 CCTV의 도로 영역 좌표 |
| `request_id` | string | 선택 | 요청 추적 ID |

## 6.2 응답 예시

```json
{
  "request_id": "REQ-CCTV032-10281",
  "cctv_id": "CCTV-032",
  "captured_at": "2026-07-16T12:10:12.425+09:00",
  "model": {
    "name": "roadbogo-yolo",
    "version": "1.0.0"
  },
  "detections": [
    {
      "class_id": 1,
      "class_name": "fallen_object",
      "confidence": 0.87,
      "bbox": [120, 240, 220, 330],
      "center": [170, 285],
      "track_id": "CCTV-032-TRACK-103",
      "is_inside_road_roi": true,
      "road_zone": "LANE_2"
    }
  ],
  "analysis": {
    "duration_seconds": 18.2,
    "repeat_count": 14,
    "risk_score": 92,
    "risk_level": "CRITICAL",
    "event_candidate": true,
    "reasons": [
      "ROAD_ROI_INSIDE",
      "PERSISTED_OVER_THRESHOLD",
      "REPEATED_DETECTION"
    ]
  },
  "timing": {
    "inference_ms": 18.4,
    "tracking_ms": 2.6,
    "analysis_ms": 1.5,
    "total_ms": 27.9
  }
}
```

## 6.3 주요 반환 정보

- CCTV ID
- 촬영 시각
- 모델명과 버전
- 객체 클래스
- 탐지 신뢰도
- Bounding Box
- 객체 중심 좌표
- Track ID
- 도로 ROI 내부 여부
- 차로 또는 도로 구역
- 객체 지속시간
- 반복 탐지 횟수
- 위험도 점수
- 위험 단계
- 사건 후보 여부
- 위험 판정 근거
- 추론 처리시간

---

## 7. CCTV 분석 상태 초기화 API

CCTV별로 메모리에 보관된 Track ID와 지속시간 정보를 초기화한다.

```http
DELETE /api/v1/streams/CCTV-032
```

### 응답 예시

```json
{
  "cctv_id": "CCTV-032",
  "status": "reset",
  "removed_tracks": 4
}
```

### 사용 상황

- CCTV 연결이 끊겼다가 다시 연결된 경우
- CCTV 스트림 주소가 변경된 경우
- 카메라 방향이 변경된 경우
- 다른 테스트 영상으로 교체한 경우
- 데모 시나리오를 처음부터 실행하는 경우

---

## 8. 모델 관리 API

## 8.1 모델 재로딩

FastAPI 서버를 재시작하지 않고 새로운 모델을 로딩한다.

```http
POST /api/v1/admin/models/reload
Content-Type: application/json
```

### 요청 예시

```json
{
  "model_path": "/models/roadbogo/best.pt",
  "model_version": "1.1.0",
  "warmup": true
}
```

### 응답 예시

```json
{
  "status": "loaded",
  "previous_version": "1.0.0",
  "current_version": "1.1.0",
  "device": "cuda:0",
  "load_time_ms": 2340
}
```

### 설계 원칙

- 관리자 또는 내부 서비스만 호출할 수 있어야 한다.
- 신규 모델 로딩 실패 시 기존 모델을 계속 사용해야 한다.
- 로딩 후 더미 입력을 이용해 Warm-up을 수행한다.
- 모델 교체 기록을 로그로 남긴다.

---

## 8.2 확장 모델 관리 API

사용자 모델 업로드와 다중 모델 관리 기능을 구현할 경우 다음 API를 추가할 수 있다.

| Method | Endpoint | 기능 |
|---|---|---|
| `POST` | `/api/v1/models` | 모델 파일 등록 |
| `GET` | `/api/v1/models` | 등록된 모델 목록 조회 |
| `GET` | `/api/v1/models/{model_id}` | 모델 상세 정보 조회 |
| `POST` | `/api/v1/models/{model_id}/activate` | 활성 모델 변경 |
| `DELETE` | `/api/v1/models/{model_id}` | 모델 삭제 |

### 보안 주의사항

PyTorch `.pt` 파일은 로딩 과정에서 임의 코드 실행 위험이 있을 수 있다.

따라서 사용자 업로드 모델은 다음과 같이 처리하는 것이 좋다.

- 격리된 Docker 컨테이너에서 실행
- 허용된 모델 구조인지 검사
- 가능하면 ONNX 형식으로 변환 후 실행
- 파일 크기 제한
- 실행시간 제한
- GPU 사용량 제한
- 실패 모델 자동 종료
- 사용자별 저장 공간과 호출 횟수 제한

---

## 9. 영상 파일 비동기 분석 API

동영상 전체를 분석해야 한다면 동기 요청 대신 작업 기반 API를 사용하는 것이 좋다.

## 9.1 작업 생성

```http
POST /api/v1/jobs/video
Content-Type: multipart/form-data
```

### 응답 예시

```json
{
  "job_id": "JOB-20260716-0012",
  "status": "queued"
}
```

## 9.2 작업 상태 조회

```http
GET /api/v1/jobs/JOB-20260716-0012
```

### 응답 예시

```json
{
  "job_id": "JOB-20260716-0012",
  "status": "processing",
  "progress": 64,
  "processed_frames": 1920,
  "total_frames": 3000
}
```

## 9.3 완료 결과 예시

```json
{
  "job_id": "JOB-20260716-0012",
  "status": "completed",
  "result_url": "/api/v1/jobs/JOB-20260716-0012/result",
  "detected_events": 3,
  "processing_time_seconds": 85.4
}
```

이 기능은 해커톤 MVP 필수 기능은 아니다.

---

## 10. 모니터링 API

Prometheus 등 모니터링 도구가 수집할 수 있는 메트릭을 제공한다.

```http
GET /metrics
```

## 10.1 주요 모니터링 항목

### 요청 상태

- 전체 추론 요청 수
- 성공 요청 수
- 실패 요청 수
- 현재 처리 중 요청 수
- 큐 대기 요청 수
- API별 응답시간

### 추론 성능

- 평균 전처리 시간
- 평균 추론 시간
- 평균 후처리 시간
- 전체 평균 처리시간
- P95 추론시간
- FPS
- 프레임 누락률

### GPU 상태

- GPU 사용률
- VRAM 사용량
- GPU 온도
- CUDA 오류 횟수
- 모델별 VRAM 사용량

### CCTV 상태

- CCTV별 마지막 요청 시각
- CCTV별 처리 프레임 수
- CCTV별 추론 오류 수
- CCTV별 활성 Track 수

---

## 11. 공통 오류 응답

모든 API에서 동일한 오류 응답 구조를 사용한다.

```json
{
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "AI model is not ready for inference.",
    "request_id": "REQ-20260716-0001",
    "details": null
  }
}
```

## 11.1 권장 HTTP 상태 코드

| 상태 코드 | 상황 |
|---:|---|
| `400` | 요청값 오류 |
| `401` | 인증 실패 |
| `403` | 접근 권한 없음 |
| `404` | 모델, CCTV 또는 작업을 찾을 수 없음 |
| `409` | 모델 전환 또는 상태 충돌 |
| `413` | 이미지나 영상 파일 크기 초과 |
| `415` | 지원하지 않는 파일 형식 |
| `422` | 요청 스키마 검증 실패 |
| `429` | 동시 요청 또는 사용량 제한 초과 |
| `500` | 내부 추론 오류 |
| `503` | 모델 또는 GPU 준비 안 됨 |
| `504` | 추론 제한시간 초과 |

## 11.2 권장 오류 코드

```text
INVALID_IMAGE
UNSUPPORTED_FILE_TYPE
FILE_SIZE_EXCEEDED
MODEL_NOT_FOUND
MODEL_NOT_READY
MODEL_LOAD_FAILED
GPU_NOT_AVAILABLE
CUDA_OUT_OF_MEMORY
INFERENCE_FAILED
INFERENCE_TIMEOUT
STREAM_NOT_FOUND
INVALID_ROI
RATE_LIMIT_EXCEEDED
```

---

## 12. 인증 및 네트워크 구성

AI Serving 서버를 외부에 직접 노출하지 않고 서비스 백엔드 또는 영상 Worker만 접근하도록 구성한다.

```text
CCTV Worker / Flask Backend
           ↓
     내부 API Key
           ↓
   FastAPI AI Serving
```

### 요청 헤더 예시

```http
Authorization: Bearer {INTERNAL_AI_API_KEY}
X-Request-ID: REQ-20260716-0001
X-Service-Name: roadbogo-worker
```

### 권장 보안 구성

- 내부망 또는 VPN에서만 접근
- 방화벽으로 허용 서버 제한
- 내부 API Key 적용
- 관리자 API 별도 권한 적용
- 운영 단계에서 mTLS 적용 검토
- 요청 크기와 호출 횟수 제한
- 파일 확장자와 MIME 타입 동시 검증
- 요청 및 모델 변경 감사 로그 기록

---

## 13. 권장 프로젝트 구조

```text
roadbogo-ai-serving/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── health.py
│   │   ├── inference.py
│   │   ├── streams.py
│   │   ├── models.py
│   │   └── metrics.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   ├── schemas/
│   │   ├── inference.py
│   │   ├── model.py
│   │   ├── stream.py
│   │   └── error.py
│   ├── services/
│   │   ├── model_service.py
│   │   ├── inference_service.py
│   │   ├── tracking_service.py
│   │   ├── roi_service.py
│   │   └── risk_service.py
│   ├── repositories/
│   │   └── model_repository.py
│   └── utils/
│       ├── image.py
│       └── timing.py
├── models/
│   └── roadbogo/
│       └── best.pt
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 14. MVP 우선 구현 순서

### 1단계: 서버 기본 구조

- FastAPI 프로젝트 생성
- 환경설정 분리
- 공통 오류 응답
- 요청 ID와 로그 처리
- `/health/live`
- `/health/ready`

### 2단계: 모델 추론

- YOLO 모델 로딩
- GPU 장치 선택
- 단일 이미지 전처리
- 객체 탐지
- 결과 후처리
- `/api/v1/models/active`
- `/api/v1/inferences/image`

### 3단계: CCTV 분석

- CCTV별 상태 저장
- 객체 추적
- ROI 판정
- 지속시간 계산
- 반복 탐지 계산
- 위험도 계산
- `/api/v1/streams/{cctv_id}/frames`
- `/api/v1/streams/{cctv_id}` 초기화

### 4단계: 운영 기능

- 관리자 모델 재로딩
- GPU 상태 수집
- Prometheus 메트릭
- 성능 및 오류 로그
- Docker 배포

---

## 15. 최종 MVP API 구성

```text
GET     /health/live
GET     /health/ready

GET     /api/v1/models/active
POST    /api/v1/inferences/image

POST    /api/v1/streams/{cctv_id}/frames
DELETE  /api/v1/streams/{cctv_id}

POST    /api/v1/admin/models/reload

GET     /metrics
```

---

## 16. 전체 연동 흐름

```text
ITS CCTV 또는 샘플 영상
→ 영상 Worker가 프레임 캡처
→ FastAPI AI Serving 서버에 분석 요청
→ YOLO 객체 탐지
→ OpenCV ROI·추적·지속시간 분석
→ 위험도와 사건 후보 여부 반환
→ Flask 서비스 백엔드가 사건 생성 또는 기존 사건 병합
→ MySQL에 사건과 탐지 근거 저장
→ WebSocket으로 관제 대시보드에 실시간 알림
→ 관제자 확인
→ 출동 담당자 배정
→ 현장 조치 및 사건 종료
```

---

## 17. 핵심 설계 결정

### 단일 이미지 API와 연속 프레임 API를 분리한다

#### 단일 이미지 API

```text
POST /api/v1/inferences/image
```

용도:

- 모델 기능 테스트
- 관리자 테스트
- 사용자 업로드 이미지 테스트
- 모델 버전별 결과 비교

#### 연속 프레임 API

```text
POST /api/v1/streams/{cctv_id}/frames
```

용도:

- 실제 CCTV 관제
- 객체 추적
- ROI 판정
- 지속시간 계산
- 반복 탐지 계산
- 위험도 산출
- 사건 후보 판단

두 API를 분리하면 단순 모델 테스트와 실제 관제 분석 로직이 섞이지 않아 유지보수와 성능 관리가 쉬워진다.
