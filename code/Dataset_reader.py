import random
import os
import numpy as np
import torch

# PyTorch Dataset 사용을 위한 모듈
from torch.utils.data import Dataset

# 콘솔 입력 시 Ctrl 관련 처리 문제를 방지하기 위해 환경변수 설정 (특정 환경에서 오류가 발생할 수 있으므로 사용)
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

RADARDATA_PATH = r"D:\Desktop\FallDetection-base-on-FMCW_DH\data_0618"  # 원래 DCA1000 등에서 추출된 데이터가 저장된 경로
LIST_PATH = r"D:\Desktop\FallDetection-base-on-FMCW_DH\list_0618"       #  데이터 경로 목록(txt 파일들)을 저장할 폴더


# ===============================
# 함수: normalize_to_0_1
# ===============================
# 입력: torch tensor (arr)
# 역할: 입력 tensor의 최솟값과 최댓값을 이용해 min-max 정규화를 수행하여 0~1 사이의 값으로 변환
# 출력: 정규화된 tensor
def normalize_to_0_1(arr):
    min_val = torch.min(arr)   # 텐서 내 최소 값 계산
    max_val = torch.max(arr)   # 텐서 내 최대 값 계산
    # min-max 정규화 (범위 0~1)
    normalized_arr = 1 * (arr - min_val) / (max_val - min_val)
    return normalized_arr

# ===============================
# 함수: make_dataset_list
# ===============================
# 입력:
#   folder_path: 데이터들이 저장된 최상위 폴더 경로 (예: 모션별로 폴더가 구성됨)
#   txt_dir: train.txt와 test.txt 파일을 저장할 디렉토리 경로
#   train_ratio: 전체 데이터 중 학습 데이터로 사용할 비율 (예: 0.7)
#
# 역할:
#   - folder_path 내의 각 모션 폴더(현재는 폴더명이 'fall'이면 낙상, 그 외는 비낙상으로 취급)를 읽어옴  
#   - 각 파일의 전체 경로와 해당 라벨(낙상: 1, 그 외: 0)을 문자열로 저장한 리스트 생성
#   - train_ratio에 따라 학습/테스트 데이터로 분리한 후 각각 train.txt와 test.txt 파일에 기록
#
# 주의:
#   현재는 이진 분류(낙상 vs 비낙상)로 라벨링되어 있음. 다중 분류를 위해서는 라벨 처리 부분을 수정해야 함.
def make_dataset_list(folder_path, txt_dir, train_ratio):
    label_map = {
        'fall': 0,
        'walk_away': 1,
        'walk_toward': 2,
        'squat': 3,
        'sit': 4,
        'stand': 5 ,
        'none' : 6
    }

    # label_map = {
    #     'fall': 0,
    #     'walk': 1,
    #     'sit': 2,
    #     'bend': 3,
    #     'squat': 4
    # }

    train_list, test_list = [], []

    for motion in os.listdir(folder_path):
        if motion not in label_map:
            print(f"[무시됨] 알 수 없는 클래스: {motion}")
            continue

        class_flag = label_map[motion]
        motion_dir = os.path.join(folder_path, motion)

        for top_path, sub_path, file in os.walk(motion_dir):
            print(top_path)
            data_list = []
            for i in range(len(file)):
                data_list.append(os.path.join(top_path, file[i]))
            random.shuffle(data_list)

            for i in range(0, int(len(file) * train_ratio)):
                train_data = data_list[i] + ' ' + str(class_flag) + '\n'
                train_list.append(train_data)

            for i in range(int(len(file) * train_ratio), len(file)):
                test_data = data_list[i] + ' ' + str(class_flag) + '\n'
                test_list.append(test_data)

    print('train:', len(train_list), '   test:', len(test_list))

    # 리스트를 정렬 및 무작위 섞기 (기본적으로 데이터를 섞어준 후 정렬)
    random.shuffle(train_list)
    test_list.sort()
    train_list.sort()
    # random.shuffle(test_list)   # 필요에 따라 테스트 데이터도 무작위로 섞을 수 있음 (현재 주석 처리)

    # 지정된 txt_dir 경로에 train.txt 파일 생성
    with open(os.path.join(txt_dir,'train.txt'), 'w', encoding='UTF-8') as f:
        for train_img in train_list:
            f.write(str(train_img))

    # 지정된 txt_dir 경로에 test.txt 파일 생성
    with open(os.path.join(txt_dir , 'test.txt'), 'w', encoding='UTF-8') as f:
        for test_img in test_list:
            f.write(test_img)

# ===============================
# 클래스: Fall_Dataset (PyTorch Dataset)
# ===============================
# 역할:
#   - train.txt나 test.txt 파일로부터 각 데이터 파일의 경로와 라벨 정보를 읽어옴
#   - __getitem__ 함수에서 numpy 배열(.npy 파일 등)을 로드하고, 이를 float32 tensor 및 라벨 tensor로 변환하여 반환
#
# 입력:
#   root_dir: 데이터의 파일 경로와 라벨 정보가 기록된 텍스트 파일 경로
#
# 출력 (__getitem__):
#   image: numpy 파일을 로드한 후 float32 형식으로 변환한 데이터 (일반적으로 RDM 이미지 데이터)
#   label: 해당 데이터의 라벨 (torch tensor 형태)
#   path: 데이터 파일의 경로 (디버깅 또는 추후 확인용)
class Fall_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # txt 파일을 읽어 key: 파일 경로, value: 라벨 딕셔너리를 생성
        self.img_label = self.load_annotations()

        # 딕셔너리에서 파일 경로와 라벨을 각각 리스트로 저장
        self.img = list(self.img_label.keys())
        self.label = [label for label in list(self.img_label.values())]

    def __len__(self):
        # 전체 데이터 개수를 반환
        return len(self.img)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 파일을 np.load를 통해 불러옴
        image = np.load(self.img[idx], allow_pickle=True)
        image = image.astype(np.float32)  # 데이터 타입을 float32로 변환
        label = self.label[idx]
        label = torch.from_numpy(np.array(label))  # 라벨을 torch tensor로 변환
        path = self.img[idx]  # 데이터 파일 경로 (디버깅 혹은 후속 처리에 유용)

        return image, label, path

    # txt 파일 내의 각 줄을 읽어 파일 경로와 라벨 정보를 딕셔너리로 생성
    def load_annotations(self):
        data_infos = {}
        with open(self.root_dir) as f:
            # 각 줄을 공백 기준으로 분할 (파일 경로와 라벨)
            lines = f.readlines()
            # print(f"Lines in txt: {len(lines)}")  # 줄 수 확인
            samples = [x.strip().split() for x in lines]
            # print(f"Parsed samples: {samples[:5]}")  # 앞 5개 출력해보기
            for filename, gt_label in samples:
                # 라벨을 int64 타입의 numpy 배열로 저장
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos
    # print(f"self.img_label length: {len(self.img_label)}")
    # print(f"self.img_label: {self.img_label}")

# ===============================
# 메인 실행부 (파일이 독립적으로 실행될 경우)
# ===============================
if __name__ == '__main__':

    radardata_path = RADARDATA_PATH  # 원래 DCA1000 등에서 추출된 데이터가 저장된 경로
    list_path = LIST_PATH     #  데이터 경로 목록(txt 파일들)을 저장할 폴더

    # list_path가 없으면 생성
    if not os.path.exists(list_path):
        os.makedirs(list_path)
    for motion in os.listdir(radardata_path):
        motion_dir = os.path.join(radardata_path, motion)

    # 위에서 정의한 make_dataset_list 함수를 호출하여 train.txt, test.txt 파일 생성
    make_dataset_list(radardata_path, list_path, 0.7)
