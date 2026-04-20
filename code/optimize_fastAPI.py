import os
import time
import datetime
import threading
import queue
import serial
import numpy as np
from collections import deque
import torch
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from Network import RDTNet   # 본인 구조에 맞게 import
from Function import *       # 필요 함수만 있으면 됨 (예: normalize_to_0_1)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import threading
from pathlib import Path
import uvicorn

# FastAPI 전역 객체 및 공유 데이터
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# HTML/JS/CSS 위치 등록
# static 폴더 경로 (현재 파일 기준 상대 경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 정적 파일 제공
# static 폴더를 루트에 바로 마운트
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# ————————————————————————————————
# main.html 을 두 경로에서 모두 서빙
# ————————————————————————————————

# 기존: "/"와 "/main.html" 둘 다 main.html 반환  -> 변경: "/"는 splash.html, "/main.html"은 그대로
@app.get("/")
def get_splash():
    return FileResponse(os.path.join(STATIC_DIR, "splash.html"))

@app.get("/main.html")
def get_main():
    return FileResponse(os.path.join(STATIC_DIR, "main.html"))

# (선택) summary 페이지도 직접 라우트하고 싶다면 추가
@app.get("/summary.html")
def get_summary():
    return FileResponse(os.path.join(STATIC_DIR, "summary.html"))
# ————————————————————————————————
# status API
# ————————————————————————————————
@app.get("/status")
def get_status():
    # === [ADD 2/3] durations 포함해서 내려주기 ===
    # return {**status_data, "durations": durations}
    return {**status_data, "durations": _durations_snapshot()}

# --- 설정 ---
CONFIG_FILE   = "64x64x10_sub.cfg"
MODEL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'RDTNet_model33.pth')
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SERIAL_CLI    = 'COM4'
SERIAL_DATA   = 'COM3'
RDM_SHAPE     = (64, 64)
WINDOW        = 24  # sliding window 길이
# CLASS_NAMES   = ['fall', 'walk_away', 'walk_toward', 'squat', 'sit', 'stand']
CLASS_NAMES   = ['fall', 'walk_away', 'walk_toward', 'squat', 'sit', 'stand', 'none']
FRAME_INTERVAL = 0.125  # 1/8초

stand_idx = CLASS_NAMES.index('stand')
squat_idx = CLASS_NAMES.index('squat')

status_data = {
    "squat_count": 0,
    "stand_count": 0,
    "current_action": "none",
    "class_probs": {
        "none": 1.0, "walk_away": 0.0, "walk_toward": 0.0,
        "squat": 0.0, "sit": 0.0, "stand": 0.0, "fall": 0.0
    }
}

# === [ADD 1/3] durations & state ===
# UI에서 읽을 키: walk_sec, sit_sec, stand_sec
# (호환용으로 walk_toward_sec / walk_away_sec도 같이 유지)
durations = {
    "walk_sec": 0.0,
    "walk_toward_sec": 0.0,
    "walk_away_sec": 0.0,
    "sit_sec": 0.0,
    "stand_sec": 0.0,
}

# 현재 진행 중인 구간 상태
span_active    = None         # 'walk' | 'sit' | 'stand' | None
span_start_ts  = None         # 구간 시작 timestamp (초)
last_walk_kind = None         # 'walk_toward' | 'walk_away' | None

# 레이블 → 카테고리 매핑
LABEL2CAT = {
    "walk_toward": "walk",
    "walk_away":   "walk",
    "sit":         "sit",
    "stand":       "stand",
}

def _add_elapsed(cat: str, start_ts: float, end_ts: float):
    """확정 구간 [start_ts, end_ts) 를 cat에 누적."""
    if start_ts is None or end_ts is None:
        return
    el = max(0.0, float(end_ts) - float(start_ts))
    if el <= 0.0:
        return

    if cat == "walk":
        durations["walk_sec"] += el
        # 호환: 'none' 포함 시간을 마지막 walk 종류에 귀속
        if last_walk_kind == "walk_toward":
            durations["walk_toward_sec"] += el
        elif last_walk_kind == "walk_away":
            durations["walk_away_sec"] += el
    elif cat == "sit":
        durations["sit_sec"] += el
    elif cat == "stand":
        durations["stand_sec"] += el

# === [ADD] 현재 열린 구간까지 포함한 durations 스냅샷 ===
def _durations_snapshot():
    snap = {
        "walk_sec":        durations.get("walk_sec", 0.0),
        "walk_toward_sec": durations.get("walk_toward_sec", 0.0),
        "walk_away_sec":   durations.get("walk_away_sec", 0.0),
        "sit_sec":         durations.get("sit_sec", 0.0),
        "stand_sec":       durations.get("stand_sec", 0.0),
    }
    # 열린 구간이 있으면, 지금 시각까지의 경과를 더해 UI에 실시간으로 보여준다
    now_ts = time.time()
    if span_active in ("walk", "sit", "stand") and span_start_ts is not None:
        el = max(0.0, now_ts - float(span_start_ts))
        if span_active == "walk":
            snap["walk_sec"] += el
            # 마지막 walk 방향에 귀속 (walk→none 포함 규칙 유지)
            if last_walk_kind == "walk_toward":
                snap["walk_toward_sec"] += el
            elif last_walk_kind == "walk_away":
                snap["walk_away_sec"] += el
        elif span_active == "sit":
            snap["sit_sec"] += el
        elif span_active == "stand":
            snap["stand_sec"] += el
    return snap


# --- UART 설정/파싱 ---
def serialConfig(cfg):
    CLIport  = serial.Serial(SERIAL_CLI, 115200)
    Dataport = serial.Serial(SERIAL_DATA, 921600)
    with open(cfg) as f:
        for line in f:
            CLIport.write((line.strip() + '\n').encode())
            time.sleep(0.01)
    return CLIport, Dataport

def parseConfigFile(cfg):
    configParameters = {}
    with open(cfg) as f:
        for line in f:
            sp = line.strip().split()
            if not sp: 
                continue
            if sp[0] == "profileCfg":
                startFreq        = float(sp[2])
                idleTime         = float(sp[3])
                rampEndTime      = float(sp[5])
                freqSlopeConst   = float(sp[8])
                numAdcSamples    = int(sp[10])
                digOutSampleRate = int(sp[11])
            elif sp[0] == "frameCfg":
                chirpStartIdx = int(sp[1])
                chirpEndIdx   = int(sp[2])
                numLoops      = int(sp[3])
    numChirps = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"]       = numChirps
    configParameters["numRangeBins"]         = numAdcSamples
    configParameters["rangeIdxToMeters"]     = (3e8 * digOutSampleRate * 1e3) / (
        2 * freqSlopeConst * 1e12 * numAdcSamples
    )
    configParameters["dopplerResolutionMps"] = 3e8 / (
        2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numChirps
    )
    return configParameters

# --- UART → RDM 파싱 ---
byteBuffer       = np.zeros(2**15, dtype='uint8')
byteBufferLength = 0

def readAndParseData18xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength
    magicWord = [2,1,4,3,6,5,8,7]
    MSG_RDM   = 5

    # 1) 버퍼에 시리얼 읽기
    readBuf = Dataport.read(Dataport.in_waiting)
    vec     = np.frombuffer(readBuf, dtype='uint8')
    cnt     = len(vec)
    if byteBufferLength + cnt < byteBuffer.size:
        byteBuffer[byteBufferLength:byteBufferLength+cnt] = vec
        byteBufferLength += cnt

    # 2) 매직워드 검색 및 TLV 파싱
    if byteBufferLength > 16:
        locs   = np.where(byteBuffer == magicWord[0])[0]
        starts = [i for i in locs if np.all(byteBuffer[i:i+8] == magicWord)]
        if starts:
            s = starts[0]
            if 0 < s < byteBufferLength:
                byteBuffer[:byteBufferLength-s] = byteBuffer[s:byteBufferLength]
                byteBuffer[byteBufferLength-s:]  = 0
                byteBufferLength                -= s

            word     = [1, 2**8, 2**16, 2**24]
            if byteBufferLength < 16:
                return None
            totalLen = int(np.dot(byteBuffer[12:16], word))
            if byteBufferLength < totalLen:
                return None

            # 헤더 스킵
            idx = 8 + 4 + 4 + 4 + 4 + 4 + 4  # magic+ver+len+plat+frameNum+timeCPU+numObj
            if byteBufferLength < idx + 4:
                return None
            numTLVs = int(np.dot(byteBuffer[idx:idx+4], word))
            idx += 4 + 4  # skip numTLVs field + subFrameNum

            rdm_frame = None
            for _ in range(numTLVs):
                if byteBufferLength < idx + 8:
                    break
                tlv_type   = int(np.dot(byteBuffer[idx:idx+4], word)); idx += 4
                tlv_length = int(np.dot(byteBuffer[idx:idx+4], word)); idx += 4
                if tlv_type == MSG_RDM:
                    vals_needed  = configParameters["numRangeBins"] * configParameters["numDopplerBins"]
                    bytes_needed = 2 * vals_needed
                    if byteBufferLength < idx + bytes_needed:
                        break
                    payload = byteBuffer[idx:idx+bytes_needed]
                    arr     = payload.view(np.int16)
                    rdm     = arr.reshape(
                        configParameters["numDopplerBins"],
                        configParameters["numRangeBins"],
                        order='F'
                    )
                    rdm_frame = rdm
                    idx += bytes_needed

            # 버퍼 정리
            byteBuffer[:byteBufferLength-totalLen]        = byteBuffer[totalLen:byteBufferLength]
            byteBuffer[byteBufferLength-totalLen:]        = 0
            byteBufferLength                            -= totalLen

            return rdm_frame

    return None


# --- 모델 로드 ---
def load_model(path):
    model = RDTNet([[1,32,64,128,256,512],[512,512,512,512]])
    ckpt  = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model

# --- 메인 루프 ---

def main():
    def run_fastapi():
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

    # main() 내부 시작 전에 FastAPI 서버 실행
    threading.Thread(target=run_fastapi, daemon=True).start()

    # 시리얼 포트 및 레이더 파라미터 설정
    CLIport, Dataport  = serialConfig(CONFIG_FILE)
    configParams      = parseConfigFile(CONFIG_FILE)

    # 모델 로드
    model = load_model(MODEL_PATH)

    # 감지 기록 버퍼 (최근 5회)
    detection_history = deque(maxlen=5)
    last_label        = None
    last_shown_class  = None
    squat_count = 0
    stand_count = 0
    in_squat_sequence    = False
    in_stand_sequence    = False
    squat_frame_counter = 0
    stand_frame_counter = 0

    # 슬라이딩 윈도우 초기화 (+ timestamp 버퍼)
    frame_buffer     = deque(maxlen=WINDOW)
    timestamp_buffer = deque(maxlen=WINDOW)
    frame_idx = 0

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 모델 추론 시작 (윈도우 크기={WINDOW})...")

    try:
        while True:
            # 1) UART → RDM fetch 시간 측정
            uart_start = time.time()
            rdm = None
            while rdm is None:
                rdm = readAndParseData18xx(Dataport, configParams)
                time.sleep(0.001)
            uart_end   = time.time()

            # 2) RDM 처리(형변환·크롭·플롯) 시간 측정
            rdm = rdm.astype(np.float32)
            if rdm.shape != RDM_SHAPE:
                rdm = rdm[:RDM_SHAPE[0], :RDM_SHAPE[1]]

            # 버퍼에 저장
            frame_buffer.append(rdm)
            timestamp_buffer.append(uart_end)

            frame_idx += 1

            # 3) 윈도우가 채워지면 모델 inference 시간 측정
            if len(frame_buffer) == WINDOW:
                input_tensor = RDM_prepare(frame_buffer, DEVICE)
                with torch.no_grad():
                    probs = model(input_tensor).cpu().numpy().squeeze(0) * 100

                # 결과 출력
                idx = np.argmax(probs)
                cls = CLASS_NAMES[idx]
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Frame {frame_idx:4d}: {cls:12} ({probs[idx]:.2f}%)")

                print("  Probabilities:", ", ".join([f"{name}: {p:.1f}%" for name, p in zip(CLASS_NAMES, probs)]))
                
                now = datetime.datetime.now().strftime('%H:%M:%S')
                # print(f"[{now}] Frame {frame_idx:4d}: {cls:12} ({probs[idx]:.2f}%)")

                # 연속 스쿼트 프레임 카운트
                if cls == 'squat' and probs[squat_idx] >= 85:
                    squat_frame_counter += 1
                else:
                    squat_frame_counter = 0
                    in_squat_sequence = False

                # 연속 stand 프레임 카운트
                if cls == 'stand' and probs[stand_idx] >= 85:
                    stand_frame_counter += 1
                else:
                    stand_frame_counter = 0
                    in_stand_sequence = False

                # 5프레임 연속 달성 시
                if squat_frame_counter >= 4:
                    if not in_squat_sequence:
                        squat_count += 1
                        in_squat_sequence = True
                        print(f"[{now}] >>> 총 스쿼트 개수: {squat_count}")
                # 5프레임 연속 달성 시
                if stand_frame_counter >= 6:
                    if not in_stand_sequence:
                        stand_count += 1
                        in_stand_sequence = True
                        print(f"[{now}] >>> 총 stand 개수: {stand_count}")

                # 감지 기록 업데이트
                detection_history.append((cls, probs[idx]))

                if len(detection_history) == 5:
                    last_classes = [c for c,_ in detection_history]
                    last_probs   = [p for _,p in detection_history]
                    if all(c == cls for c in last_classes) and all(p >= 85 for p in last_probs):
                        status_data["squat_count"]     = squat_count
                        status_data["stand_count"]     = stand_count
                        status_data["current_action"]  = cls
                        status_data["class_probs"]     = {name: round(float(p/100), 3) for name, p in zip(CLASS_NAMES, probs)}

                        # === [ADD 3/3] 시간 누적 FSM ===
                        now_ts = timestamp_buffer[-1]  # 방금 확정된 프레임의 시각

                        global span_active, span_start_ts, last_walk_kind

                        # 이번에 확정된 레이블의 카테고리
                        new_cat = LABEL2CAT.get(cls, None)

                        if new_cat in ("walk", "sit", "stand"):
                            # walk/sit/stand가 확정된 경우
                            if span_active is None:
                                # 새 구간 시작
                                span_active   = new_cat
                                span_start_ts = now_ts
                                if new_cat == "walk":
                                    last_walk_kind = cls  # toward/away 기억
                            else:
                                if new_cat == span_active:
                                    # 같은 카테고리 계속 (walk_toward→walk_away 등)
                                    if new_cat == "walk":
                                        last_walk_kind = cls  # 마지막 walk 종류 갱신
                                    # 구간은 열린 채로 유지 (NONE 오면 함께 포함시킬 예정)
                                else:
                                    # 다른 카테고리로 전환됨 ⇒ 이전 구간 종료(= NONE 포함 X)
                                    _add_elapsed(span_active, span_start_ts, now_ts)
                                    # 새 구간 시작
                                    span_active   = new_cat
                                    span_start_ts = now_ts
                                    last_walk_kind = cls if new_cat == "walk" else last_walk_kind

                        elif cls == "none":
                            # NONE이 확정된 경우
                            if span_active in ("walk", "sit", "stand"):
                                # NONE을 같은 구간에 포함 → 아직 닫지 않음
                                # (다음에 다른 동작/낙상 등으로 바뀔 때 그 시점에 한꺼번에 종료)
                                pass
                            else:
                                # 이미 비활성 상태면 아무 것도 안 함
                                pass

                        else:
                            # 그 외(예: fall, squat 등) ⇒ 이전 구간이 열려있다면 여기서 종료
                            if span_active in ("walk", "sit", "stand"):
                                _add_elapsed(span_active, span_start_ts, now_ts)
                                span_active   = None
                                span_start_ts = None
                                # walk 구간 아니면 last_walk_kind는 유지해도 무방

                    detection_history.clear()
                
            # 프레임 간 인터벌 유지
            dt = time.time() - uart_start
            if dt < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - dt)

    except KeyboardInterrupt:
        print("\n[INFO] 키보드 인터럽트: 프로그램 종료")
        CLIport.write(b"sensorStop\n")
        CLIport.close()
        Dataport.close()

if __name__ == "__main__":

    main()
