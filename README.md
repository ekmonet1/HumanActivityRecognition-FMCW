# FallDetection based on FMCW Radar

FMCW 레이더(TI IWR series)에서 수집한 Range-Doppler Map(RDM)을 딥러닝으로 분류하는 실내 행동 인식 및 낙상 감지 시스템입니다.

---

## 시스템 개요

```
TI FMCW 레이더 (UART)
        ↓
  RDM 파싱 & 전처리
        ↓
  슬라이딩 윈도우 (24 프레임)
        ↓
     RDTNet 추론
        ↓
  FastAPI 웹 UI / Excel 통계
```

- **레이더**: TI IWR series (Range-Doppler Map 출력, 64×64, 8 FPS)
- **분류 클래스**: `fall`, `walk_away`, `walk_toward`, `squat`, `sit`, `stand`, `none` (7종)
- **최종 모델**: `RDTNet_model51.pth`

---

## 모델 구조 — RDTNet

```
입력: (batch × 24프레임, 1, 64, 64)  ← 24프레임 슬라이딩 윈도우
         │
   Conv2D_Module_5          ← 2D CNN (공간적 특징 추출)
    AvgPool4 → DepthwiseConv → PointwiseConv (×4)
    → GlobalAvgPool → (batch×24, C)
         │
    reshape: (batch, C, 24) ← 프레임을 시계열로 재구성
         │
   Conv1D_Module_2          ← 1D CNN (시간적 특징 집계)
    DepthwiseConv1D → PointwiseConv1D (×3) → FC
         │
    Softmax → 7클래스 확률
```

| 구성 요소 | 역할 |
|---|---|
| `Conv2D_Module_5` | 각 RDM 프레임에서 공간 특징 추출 |
| `Conv1D_Module_2` | 24프레임 시퀀스에서 시간 특징 집계 |
| 슬라이딩 윈도우 (24프레임) | 약 3초 구간의 동작 인식 |

---

## 파일 구조

```
FallDetection-base-on-FMCW_DH/
├── README.md
├── 64x64x10_sub.cfg            # TI 레이더 설정 파일
├── model/
│   └── RDTNet_model51.pth      # 최종 학습 모델
└── code/
    ├── Network.py                          # RDTNet 모델 정의
    ├── Function.py                         # 전처리 유틸 (RDM_prepare, normalize 등)
    ├── Dataset_reader.py                   # 학습용 데이터셋 로더
    ├── Train.py                            # 학습 코드
    ├── optimize_fastAPI.py                 # 실시간 추론 (UART + FastAPI 웹 UI)
    ├── optimize_for_excel_after_reject.py  # 오프라인 .npy 후처리 + Excel 통계
    └── static/
        ├── splash.html                     # 시작 화면
        ├── main.html                       # 실시간 모니터링 UI
        └── summary.html                   # 활동 요약 UI
```

---

## 사용법

### 1. 학습 (`Train.py`)

```python
# Train.py 상단 파라미터 수정
state     = 'train'   # 'train' 또는 'verify'
MODEL_NUM = 54        # 저장할 모델 번호
train_dir = r"경로\list_0618\train.txt"
valid_dir = r"경로\list_0618\test.txt"
```

```bash
python Train.py
```

- 데이터는 클래스별 폴더 구조로 정리된 RDM `.npy` 파일을 사용
- `Dataset_reader.py`의 `make_dataset_list()`로 `train.txt` / `test.txt` 생성
- 검증 정확도 또는 fall recall 향상 시 자동 저장

### 2. 실시간 추론 (`optimize_fastAPI.py`)

레이더를 PC에 USB 연결 후 실행:

```bash
python optimize_fastAPI.py
```

- 시리얼 포트: CLI → `COM4` (115200 baud), Data → `COM3` (921600 baud)
- 레이더 설정 파일: `64x64x10_sub.cfg`
- 모델 경로: `../model/RDTNet_model33.pth` (필요 시 수정)
- 웹 UI: `http://127.0.0.1:8000`

**동작 확정 로직**: 최근 5프레임 모두 동일 클래스 & 확률 85% 이상 → 확정

### 3. 오프라인 후처리 (`optimize_for_excel_after_reject.py`)

레이더 없이 저장된 `.npy` 프레임을 재생하여 분석:

```bash
python optimize_for_excel_after_reject.py
```

**설정 (스크립트 상단 수정)**:

```python
RDM_SEQ_DIR = r"경로\rdm_frames_XXXXXXXX"  # .npy 파일 폴더
MODEL_PATH  = r"경로\model\RDTNet_model52.pth"
SUMMARY_XLSX_PATH  = r"경로\통계분석.xlsx"
SUMMARY_SHEET_NAME = "summary_시트명"
```

**확정 로직**: 클래스별 연속 윈도우 + 임계 확률 충족 시 이벤트 카운트

| 클래스 | 확정 윈도우 | 확률 임계값 |
|---|---|---|
| sit / stand | 6프레임 | 90% |
| squat | 6프레임 | 90% |
| fall | 6프레임 | 90% |
| walk / none | 6프레임 | 90% |

- 종료 시 Excel summary 시트에 자동으로 행 추가 (시간별 활동 카운트 + 지속시간)

---

## 환경

```
Python 3.9+
torch
fastapi
uvicorn
pyserial
pyqtgraph
numpy
openpyxl
```

```bash
pip install torch fastapi uvicorn pyserial pyqtgraph numpy openpyxl
```

---

## 데이터 형식

- **RDM**: `(64, 64)` int16 → float32 변환 후 min-max 정규화
- **슬라이딩 윈도우**: 24프레임 → `(24, 1, 64, 64)` 텐서로 모델 입력
- **데이터 저장 구조**:
  ```
  data_MMDD/
  ├── fall/      ← npy 파일들
  ├── walk_away/
  ├── walk_toward/
  ├── squat/
  ├── sit/
  ├── stand/
  └── none/
  ```
