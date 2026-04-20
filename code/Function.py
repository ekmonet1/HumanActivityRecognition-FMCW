import random
import numpy as np
import torch



# =============================================================================
# 함수: amplitude_filtering
# =============================================================================
# 입력: 
#   - array: 원본 RDM 데이터를 담은 tensor (예: amplitude 정보가 담긴 2차원 tensor)
#   - amplitude_threshold: amplitude 기준 임계값. 이 값보다 작은 픽셀은 0으로 설정
#
# 역할:
#   - 전체 array를 순회하면서 각 원소가 amplitude_threshold보다 작으면 0으로 만듦.
#   - 기존 시스템에서는 낙상 감지를 위한 threshold 적용을 통한 전처리 역할
#
# 주의:
#   - 다중 분류 시스템에서는 'threshold'를 적용하지 않고 원본 데이터를 그대로 활용할 수 있으므로
#     이 함수 호출을 제거하거나 수정할 필요가 있음.
#
# 출력:
#   - threshold 처리가 적용된 tensor (동일한 shape)
def amplitude_filtering(array, amplitude_threshold):
    # 배열의 행(i)와 열(j)을 모두 순회
    for i in range(array.size(0)):
        for j in range(array.size(1)):
            # 임계값보다 작으면 해당 픽셀을 0으로 설정
            if array[i, j] < amplitude_threshold:
                array[i, j] = 0
    return array


# =============================================================================
# 함수: calculate_barycenter
# =============================================================================
# 입력:
#   - data: RDM 또는 가공된 데이터 (tensor)로, 픽셀 값이 0보다 큰 부분에 대해 계산
#   - count_threshold: 유효한 점(픽셀)의 최소 개수, 이보다 적으면 barycenter 계산을 하지 않음(0 반환)
#   - debug: 디버깅 출력 여부 (True/False)
#
# 역할:
#   - data에서 값이 0보다 큰 픽셀들의 행 인덱스들의 가중 평균(중심, barycenter)을 계산
#   - 낙상 감지 시스템에서 몸의 중심(바리센터) 위치 변화를 통해 이상 상황을 감지하는데 활용됨
#
# 주의:
#   - 다중 분류 시스템으로 전환할 경우, 낙상 감지에 필요한 barycenter 정보가
#     반드시 사용되는지 확인하고, 필요 없다면 이 함수를 제거하거나 수정할 수 있음.
#
# 출력:
#   - barycenter: 계산된 중심 값 (조건에 미달하면 0)
#   - count: 기준을 충족한 픽셀의 개수
def calculate_barycenter(data, count_threshold, debug):
    temp_barycenter = 0
    count = 0
    # data의 모든 위치에 대해 반복
    for i in range(data.size(0)):
        for j in range(data.size(1)):
            # 값이 양수인 픽셀만 포함시킴
            if data[i, j] > 0:
                temp_barycenter = temp_barycenter + i  # 행 인덱스를 누적
                count = count + 1
    # 유효한 픽셀 수가 일정 개수 이상이면 평균 계산, 아니면 0
    if count > count_threshold:
        barycenter = temp_barycenter / count
    else:
        barycenter = 0

    if debug:
        print('count=', count, 'barycenter=', barycenter)
    return barycenter, count


# =============================================================================
# 함수: FFT_2D
# =============================================================================
# 입력:
#   - data1: 2차원 RDM의 일부 혹은 하나의 chirp 데이터 (numpy array, shape: [chirps, samples])
#   - para: 파라미터 딕셔너리 (예: 'chirps', 'samples', 'fft_Range', 'fft_Vel' 등)
#
# 역할:
#   - 1차원 Hanning 윈도우를 적용한 후, 각 chirp에 대해 range FFT 수행
#   - 평균 제거 (Mean cancellation) 과정을 수행하여 클러터(잡음)를 감소
#   - 그 후 doppler FFT를 수행하여 2D FFT (RDM)를 생성
#
# 주의:
#   - 여러분의 시스템은 DCA1000 대신 UART TLV 형식에서 RDM을 추출할 예정이므로,
#     기존의 FFT_2D 함수의 호출 및 데이터 형식 부분을 변경할 필요가 있음.
#
# 출력:
#   - sigDopplerFFT: doppler FFT 결과, 2차원 복소수 numpy array (shape: [chirps, samples])
def FFT_2D(data1, para):
    # Hanning window를 적용하기 위해 동일 shape의 빈 배열 생성 (복소수 타입)
    sigRangeWin1 = np.zeros((para['chirps'], para['samples']), dtype=complex)
    # Hanning 윈도우 생성 (양쪽 끝 제외: [1: para['samples']+1])
    window = np.hanning(para['samples'] + 2)[1:para['samples'] + 1]

    # 각 chirp에 대해 윈도우 적용 (element-wise 곱셈)
    for l in range(0, para['chirps']):
        sigRangeWin1[l] = np.multiply(data1[l, :], window)

    # range FFT 처리를 위한 빈 배열 초기화
    sigRangeFFT = np.zeros((para['chirps'], para['samples']), dtype=complex)

    # 각 chirp에 대해 FFT 수행 (fft_Range 크기의 FFT)
    for l in range(0, para['chirps']):
        sigRangeFFT[l] = np.fft.fft(sigRangeWin1[l], para['fft_Range'])

    # 평균 제거: 모든 chirp에 대해 FFT 결과의 평균을 빼줌 (여기서 128은 하드코딩된 chirps 수)
    avg = np.sum(sigRangeFFT, axis=0) / 128
    for l in range(0, para['chirps']):
        sigRangeFFT[l, :] = sigRangeFFT[l, :] - avg

    # Doppler FFT 처리를 위한 빈 배열 초기화
    sigDopplerFFT = np.zeros((para['chirps'], para['samples']), dtype=complex)
    # 각 range bin에 대해 Doppler FFT 수행 후, fftshift 적용 (fft_Vel 크기)
    for n in range(0, para['samples']):
        sigDopplerFFT[:, n] = np.fft.fftshift(np.fft.fft(sigRangeFFT[:, n], para['fft_Vel']))

    return sigDopplerFFT


# =============================================================================
# 함수: data_to_RDM
# =============================================================================
# 입력:
#   - path: 데이터 파일의 경로 (텍스트 파일로 저장된 복소수 데이터)
#   - para: 파라미터 딕셔너리 (샘플 수, chirps, Rx, fft_Range, fft_Vel 등 포함)
#   - Rx_num: 사용할 Rx(수신 안테나) 개수. 데이터의 첫 차원이 Rx에 해당됨.
#
# 역할:
#   - 주어진 파일에서 데이터를 읽어들인 후, 파라미터에 맞게 리쉐이프하여 (Rx, chirps, samples)
#   - 각 Rx에 대해 FFT_2D를 적용한 후, amplitude(절대값)를 구함
#   - 여러 Rx의 결과를 평균 내어 최종 RDM (Range-Doppler Map)을 생성
#
# 주의:
#   - DCA1000 데이터를 기준으로 작성됨. 여러분의 시스템에서는 UART TLV 방식의 데이터 형식에 맞게 수정해야 함.
#   - 입력 RDM의 크기가 달라질 경우 (예: 128x32) 파라미터 'samples', 'chirps' 등도 수정 필요.
#
# 출력:
#   - RDM: 전치(transpose)를 적용한 최종 RDM (numpy array, shape: [samples, chirps]가 아닌 [chirps, samples]의 전치)
def data_to_RDM(path, para, Rx_num):
    # 텍스트 파일에서 복소수 데이터를 읽어옴
    in_data1 = np.loadtxt(path, dtype=complex, comments='#', delimiter=None, converters=None, skiprows=0,
                          unpack=False, ndmin=0)
    # 읽어온 데이터를 Rx, chirps, samples 형태로 리쉐이프 (기존 DCA1000 데이터 형식)
    in_data1 = in_data1.reshape((para['Rx'], para['chirps'], para['samples']))  # (Rxs, chirps, samples)
    # 결과를 누적할 빈 배열 초기화
    sigDopplerFFT = np.zeros(shape=(para['chirps'], para['samples']), )
    # 각 수신 안테나(Rx)에 대해 FFT_2D 적용 후, 절대값(진폭)을 누적
    for Rx in range(Rx_num):
        sigDopplerFFT = sigDopplerFFT + np.abs(FFT_2D(in_data1[Rx], para))
    # Rx개수 만큼의 평균을 취함
    sigDopplerFFT_sf = (sigDopplerFFT) / Rx_num

    # 전치하여 최종 RDM 반환 (이 코드에서는 전치하여 [samples, chirps] 형태로 만듦)
    RDM = sigDopplerFFT_sf.T
    return RDM


# =============================================================================
# 함수: find_last_index
# =============================================================================
# 입력:
#   - input_list: 검색 대상 리스트
#   - target: 찾고자 하는 값
#
# 역할:
#   - input_list 내에서 target 값이 마지막으로 등장한 인덱스를 반환.
#
# 출력:
#   - target이 마지막에 등장하는 인덱스, 없으면 None 반환.
def find_last_index(input_list, target):
    indices = []
    for i in range(len(input_list)):
        if input_list[i] == target:
            indices.append(i)
    if indices:
        return indices[-1]
    else:
        return None


# =============================================================================
# 함수: get_para
# =============================================================================
# 역할:
#   - 시스템에서 사용되는 고정 파라미터들을 딕셔너리 형태로 생성 및 반환.
#   - 파라미터에는 ADC 비트수, 시작 주파수, Rx/Tx 수, 샘플 수, chirps 수, FFT 크기 등이 포함됨.
#
# 주의:
#   - 여러분의 시스템에서 RDM 데이터 크기가 변경된다면 (예: 128x32) 'samples', 'chirps' 등 관련 파라미터를 수정해야 함.
#
# 출력:
#   - para: 각종 파라미터를 담은 딕셔너리
def get_para():
    # 상수 정의
    light_speed = 3e8
    Start_Frequency = 60.25e9

    # # 샘플링 주파수 (초기 DCA1000 기준)
    # Fs = 6 * 10 ** 6  # data2 Sampling frequency
    # Sample_rate = Fs

    # para = {
    #     'light_speed': 3e8,
    #     'numADCBits': 16,  # ADC 샘플 데이터의 비트 수
    #     'start_frequency': 60.25e9,
    #     'lamda': light_speed / Start_Frequency,
    #     'Rx': 4,         # 수신 안테나 수 (DCA1000 기준)
    #     'Tx': 1,         # 송신 안테나 수
    #     'sample_rate': Sample_rate,
    #     'sweepslope': 60e12,
    #     'samples': 128,  # 한 chirp 당 샘플 수 (변경 필요: 예를 들어 32로 변경)
    #     'chirps': 128,   # 한 프레임 당 chirps 수 (데이터 크기에 따라 수정 가능)
    #     'Tchirp': 160e-6,  # chirp 시간 (초 단위)
    #     'frames': 9000,
    #     'Tframe': 100,  # frame 간 시간 간격 (ms)
    #     'fft_Range': 128,  # FFT 수행시 range FFT 사이즈 (samples와 동일)
    #     'fft_Vel': 128,    # Doppler FFT 사이즈 (chirps와 동일)
    #     'num_crop': 3,
    #     'max_value': 1e+04  # IWR6843 장비 기준 최대 값 (필요시 조정)    
    # }

    # Config 기준 설정
    start_frequency_GHz = 77            # profileCfg 0 77 ...
    idle_time_us = 7                    # profileCfg ... 7 ...
    ramp_end_time_us = 27.23           # profileCfg ... 27.23 ...
    freq_slope_MHz_us = 100            # profileCfg ... 100 ...
    adc_samples = 64                   # profileCfg ... 64 ...
    sample_rate_ksps = 4166            # profileCfg ... 4166 ...
    chirp_loops = 64                   # frameCfg ... 64 ...
    frame_periodicity_ms = 125         # frameCfg ... 125 ...

    # 파생 값 계산
    start_frequency = start_frequency_GHz * 1e9             # Hz
    freq_slope = freq_slope_MHz_us * 1e6                    # Hz/us
    sample_rate = sample_rate_ksps * 1e3                    # Hz
    T_chirp = (idle_time_us + ramp_end_time_us) * 1e-6      # sec
    lambda_radar = light_speed / start_frequency
    chirps_per_frame = chirp_loops                         # chirpCfg 0/1 포함이지만 chirpStart=0, chirpEnd=0 → 1개 chirp만 사용
    doppler_fft_size = chirp_loops                         # Doppler FFT
    range_fft_size = adc_samples                           # Range FFT

    para = {
        'light_speed': light_speed,
        'numADCBits': 16,
        'start_frequency': start_frequency,
        'lamda': lambda_radar,
        'Rx': 4,
        'Tx': 1,
        'sample_rate': sample_rate,
        'sweepslope': freq_slope,
        'samples': adc_samples,
        'chirps': chirp_loops,
        'Tchirp': T_chirp,
        'frames': 0,                        # frameCfg에서 0 → 무한 전송
        'Tframe': frame_periodicity_ms,
        'fft_Range': range_fft_size,
        'fft_Vel': doppler_fft_size,
        'num_crop': 3,
        'max_value': 1e4
    }

    return para


# =============================================================================
# 함수: normalize_to_0_1 (중복: Dataset_reader.py에서도 동일 함수 사용)
# =============================================================================
# 입력:
#   - arr: 정규화를 원하는 torch tensor
#
# 역할:
#   - tensor 내 최소, 최대값을 이용하여 0~1 사이로 정규화
#
# 출력:
#   - 정규화된 tensor
def normalize_to_0_1(arr):
    min_val = torch.min(arr)
    max_val = torch.max(arr)
    normalized_arr = 1 * (arr - min_val) / (max_val - min_val)
    return normalized_arr


# =============================================================================
# 함수: RDM_prepare
# =============================================================================
# 입력:
#   - RDarray_list: 여러 프레임의 RDM 데이터를 담고 있는 리스트 (예: deque 형태로 쌓여 있음)
#   - device: torch.device (GPU 혹은 CPU)
#
# 역할:
#   - RDarray_list를 numpy array로 변환 후, torch tensor로 변환하고
#     CNN/TCN에 입력할 수 있는 shape으로 리쉐이프.
#
#   현재 shape: (16, 1, 62, 50) → 이는 프레임 수, 채널, 높이, 너비를 의미.
#
# 주의:
#   - 다중 분류를 위해 RDM 데이터의 크기가 128x32로 변경된다면,
#     여기서 리쉐이프하는 부분을 수정해야 함.
#
# 출력:
#   - data_buffer: 모델 입력에 적합한 형식의 tensor (dtype: float32, device에 맞게 할당)
def RDM_prepare(RDarray_list, device):
    # 리스트를 numpy array로 변환 (dtype: float)
    data_buffer = np.array(RDarray_list, dtype=float)
    # numpy array를 torch tensor로 변환
    data_buffer = torch.from_numpy(data_buffer)
    # 현재 리쉐이프: 16 프레임, 1 채널, 62행, 50열
    # → 다중 분류용 RDM 크기(예: 128x32)로 변경할 경우 이 부분 수정 필요!
    data_buffer = data_buffer.reshape((24, 1, 64, 64))
    # tensor의 데이터 타입을 float32로 변경
    data_buffer = data_buffer.to(torch.float32)
    # 모델이 위치한 device (GPU/CPU)로 이동
    data_buffer = data_buffer.to(device=device)

    return data_buffer


# =============================================================================
# 함수: seed_setting
# =============================================================================
# 입력:
#   - seed: 고정할 seed 값 (예: 11)
#
# 역할:
#   - reproducibility(재현성)을 위해 PyTorch, numpy, random 모듈에 대해 seed 고정
#
# 출력: 없음 (전역적으로 seed 설정)
def seed_setting(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # CuDNN 관련 설정: 연산의 결정론적 결과 보장
    torch.backends.cudnn.deterministic = True


# =============================================================================
# 클래스: BCEFocalLoss (Binary Cross Entropy Focal Loss)
# =============================================================================
# 역할:
#   - focal loss를 사용하여 hard examples(어려운 샘플)에 대해 더 큰 가중치를 부여하는 손실 함수.
#
# 입력:
#   - gamma: focusing parameter (기본값 2)
#   - alpha: balancing parameter (기본값 0.25)
#   - reduction: 손실의 집계 방법 ('mean' 또는 'sum')
#
# 주의:
#   - 원래 이진 분류용으로 설계되었으며, 다중 분류로 변경할 경우 손실 함수 역시 CrossEntropyLoss 등
#     다중 분류용으로 변경 필요.
#
# 출력:
#   - forward 메서드에서 계산된 loss 값을 반환
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = predict  # 예측 값
        # Focal loss 공식에 의한 loss 계산
        loss = - ((1 - self.alpha) * ((1 - pt + 1e-5) ** self.gamma) * (target * torch.log(pt + 1e-5)) + 
                  self.alpha * ((pt + 1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt + 1e-5)))

        # reduction 처리: mean 또는 sum
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
