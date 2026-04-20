import torch.nn
from torch import nn

# dropout 확률: 과적합을 방지하기 위해 사용 (수정 가능)
dropout_rate = 0.5

# -------------------------------
# 파라미터 설정
# -------------------------------
NUM_CLASSES = 7
MID_DIM = 32

# =============================================================================
# 클래스: Conv1D_Module_2
# =============================================================================
# 역할:
#   - 1D 컨볼루션 네트워크로, 입력된 1D 시퀀스 데이터를 추가적인 특징 추출 및
#     최종 분류를 위한 fully connected (FC) 계층에 전달할 특징 벡터로 가공
#
# 입력:
#   - channels_block: 리스트로, 계층별 채널 수를 지정
#     예) [in_channels, mid_channels, out_channels]
#     (원래 네트워크는 최종 출력이 1로 구성되어 이진 분류(sigmoid)로 나오지만,
#      다중 분류로 변경할 경우 마지막 FC layer의 출력 수와 활성화 함수를 수정해야 함)
#
# 출력:
#   - 최종 FC 계층을 거쳐 나온 결과 (원래는 1개의 스칼라 값; 다중 분류를 원한다면 클래스 수로 수정 필요)
class Conv1D_Module_2(nn.Module):
    def __init__(self, channels_block) -> None:
        super().__init__()
        # -------------------------------------------------------------------------
        # 첫 번째 그룹 컨볼루션:
        # - 입력: channels_block[0] 채널
        # - 그룹별 컨볼루션: 각 채널별로 별도 처리 (depthwise convolution)
        # - 커널 크기 3: (1D 신호의 지역 특징 추출)
        self.group_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[0],
                      out_channels=channels_block[0],
                      groups=channels_block[0],
                      kernel_size=3,
                      bias=True),
            nn.BatchNorm1d(channels_block[0]),
        )
        # -------------------------------------------------------------------------
        # 첫 번째 포인트 컨볼루션:
        # - 1x1 컨볼루션을 사용해 채널 수를 확장 혹은 축소 (채널 간 상호작용)
        # - ReLU 활성화와 1D max pooling (특징 축소)
        self.point_conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[0],
                      out_channels=channels_block[1],
                      kernel_size=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 시퀀스 길이를 절반으로 줄임
        )
        # -------------------------------------------------------------------------
        # 두 번째 그룹 컨볼루션:
        # - 이전 단계의 출력에 대해 각 채널별로 3 크기의 필터를 적용
        self.group_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[1],
                      out_channels=channels_block[1],
                      groups=channels_block[1],
                      kernel_size=3,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[1]),
        )
        # -------------------------------------------------------------------------
        # 두 번째 포인트 컨볼루션:
        # - 1x1 컨볼루션으로 채널 수를 변경하고, ReLU와 MaxPool를 적용
        self.point_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[1],
                      out_channels=channels_block[2],
                      kernel_size=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 추가로 길이 축소
        )        # -------------------------------------------------------------------------
        # 세 번째 그룹 컨볼루션:
        # - 이전 단계의 출력에 대해 각 채널별로 3 크기의 필터를 적용
        self.group_conv3 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[2],
                      out_channels=channels_block[2],
                      groups=channels_block[2],
                      kernel_size=3,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[2]),
        )
        # -------------------------------------------------------------------------
        # 세 번째 포인트 컨볼루션:
        # - 1x1 컨볼루션으로 채널 수를 변경하고, ReLU와 MaxPool를 적용
        self.point_conv3 = nn.Sequential(
            nn.Conv1d(in_channels=channels_block[2],
                      out_channels=channels_block[3],
                      kernel_size=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm1d(channels_block[3]),
            nn.ReLU(),
        )
        # -------------------------------------------------------------------------
        # Flatten, dropout 및 Fully Connected 계층:
        # - Flatten: 컨볼루션 결과를 1D 벡터로 변환
        # - fc1: 중간 차원으로 축소 (현재 2 * channels_block[2] 크기를 입력받음)
        # - fc2: 최종 출력 (원래 출력 1개; 다중 분류 시 이 부분을 클래스 개수로 변경하고, 활성화 함수를 softmax 등으로 수정)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(4 * channels_block[2], MID_DIM, bias=True)
        self.fc2 = nn.Linear(MID_DIM, NUM_CLASSES, bias=True)  # 다중 분류 시 출력 수 변경 필요

    def forward(self, input):
        # input shape: (batch, channels, sequence_length)
        output = self.group_conv1(input)
        output = self.point_conv1(output)
        output = self.group_conv2(output)
        output = self.point_conv2(output)

        # # -------------------------------
        # # [추가] 채널 축 max pooling
        # # → (batch, 256, 14) → (batch, 128, 14)
        # # nn.MaxPool1d는 dim=2 (seq_len)에만 적용 가능 → 채널 축 pooling 불가
        # # workaround: unsqueeze + MaxPool2d
        # output = output.unsqueeze(-1)   # (batch, 256, 14, 1)
        # output = torch.nn.functional.max_pool2d(output, kernel_size=(2,1))  # (batch, 128, 14, 1)
        # output = output.squeeze(-1)     # (batch, 128, 14)
        # # -------------------------------
        
        output = self.flatten(output)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        # 최종 출력: 원래는 단일 스칼라 (0~1, sigmoid 처리됨)
        return output

# =============================================================================
# 클래스: Conv2D_Module_5
# =============================================================================
# 역할:
#   - 2D 컨볼루션 네트워크로, 입력된 RDM 이미지(예: 62x50 또는 변경된 크기)를 대상으로
#     특징 추출을 진행한 뒤 global average pooling에 준하는 처리를 통해
#     (batch, feature) 형태의 벡터를 출력
#
# 입력:
#   - channels_block: 리스트로, 2D 컨볼루션 블록의 각 단계별 채널 구성을 정의
#     예) [in_channels, mid_channels1, mid_channels2, mid_channels3, mid_channels4, mid_channels5]
#     (입력 RDM 이미지의 채널 수는 보통 1이지만, 필요에 따라 수정 가능)
#
# 출력:
#   - 최종 특징 벡터: 각 채널에 대해 공간적 평균(pooling) 결과 (shape: (batch, channels))
#
# 주의:
#   - 여러분의 새로운 시스템에서 입력 이미지 크기가 (128x32)로 변경되면, 각 maxpool 계층과
#     padding, 커널 크기를 조정해야 할 수 있음.
class Conv2D_Module_5(nn.Module):
    def __init__(self, channels_block) -> None:
        super().__init__()
        # -------------------------------------------------------------------------
        # 첫 번째 2D 컨볼루션 블록:
        # - Conv2D: 3x3 필터, 'same' padding으로 입력과 동일한 spatial 크기를 유지한 채로
        #   채널을 channels_block[1]로 변경
        # - BatchNorm, ReLU, 그리고 2x2 MaxPool (spatial 크기를 절반으로 축소)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels_block[0],
                      out_channels=channels_block[1],
                      kernel_size=3,
                      padding='same',
                      bias=True),
            nn.BatchNorm2d(channels_block[1]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4)
        )
        # -------------------------------------------------------------------------
        # 첫 번째 그룹 컨볼루션 블록:
        # - 각 채널별 depthwise conv: 입력 채널을 그대로 유지하며 지역 정보 추출
        self.group_conv1 = nn.Sequential(
            nn.Conv2d(channels_block[1], channels_block[1], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[1], bias=True),
            nn.BatchNorm2d(channels_block[1]),
        )
        # -------------------------------------------------------------------------
        # 첫 번째 포인트 컨볼루션 블록:
        # - 1x1 컨볼루션을 통해 채널 수를 channels_block[2]로 변경 후, ReLU와 MaxPool
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channels_block[1], channels_block[2], kernel_size=1, bias=True),
            nn.BatchNorm2d(channels_block[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # -------------------------------------------------------------------------
        # 두 번째 그룹 컨볼루션 블록:
        self.group_conv2 = nn.Sequential(
            nn.Conv2d(channels_block[2], channels_block[2], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[2], bias=True),
            nn.BatchNorm2d(channels_block[2]),
        )
        # -------------------------------------------------------------------------
        # 두 번째 포인트 컨볼루션 블록:
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(channels_block[2], channels_block[3], 1, bias=True),
            nn.BatchNorm2d(channels_block[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # -------------------------------------------------------------------------
        # 세 번째 그룹 컨볼루션 블록:
        self.group_conv3 = nn.Sequential(
            nn.Conv2d(channels_block[3], channels_block[3], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[3], bias=True),
            nn.BatchNorm2d(channels_block[3]),
        )
        # -------------------------------------------------------------------------
        # 세 번째 포인트 컨볼루션 블록:
        self.point_conv3 = nn.Sequential(
            nn.Conv2d(channels_block[3], channels_block[4], 1, bias=True),
            nn.BatchNorm2d(channels_block[4]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # -------------------------------------------------------------------------
        # 네 번째 그룹 컨볼루션 블록:
        self.group_conv4 = nn.Sequential(
            nn.Conv2d(channels_block[4], channels_block[4], kernel_size=3, stride=1, padding='same',
                      groups=channels_block[4], bias=True),
            nn.BatchNorm2d(channels_block[4]),
        )
        # -------------------------------------------------------------------------
        # 네 번째 포인트 컨볼루션 블록:
        self.point_conv4 = nn.Sequential(
            nn.Conv2d(channels_block[4], channels_block[5], 1, bias=True),
            nn.BatchNorm2d(channels_block[5]),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        # -------------------------------------------------------------------------
        # Flatten 계층: 최종 feature map을 1D 벡터로 변환하는 대신,
        # 여기서는 후에 평균(pooling)으로 각 채널의 값을 하나로 집약할 예정.
        self.flatten = nn.Flatten()

    def forward(self, input):
        # input shape: (batch, channels, height, width)
        output = self.conv1(input)
        output = self.group_conv1(output)
        output = self.point_conv1(output)
        output = self.group_conv2(output)
        output = self.point_conv2(output)
        output = self.group_conv3(output)
        output = self.point_conv3(output)
        output = self.group_conv4(output)
        output = self.point_conv4(output)
        # Global average pooling: 마지막 두 차원(height, width)에 대해 평균을 취함
        output = output.mean(dim=-1)  # width 차원 평균
        output = output.mean(dim=-1)  # height 차원 평균
        # 최종 출력: (batch, channels) 형태의 특징 벡터
        return output

# =============================================================================
# 클래스: RDTNet
# =============================================================================
# 역할:
#   - 전체 네트워크를 구성하며, Conv2D와 Conv1D 모듈을 이어붙여 최종 분류 결과를 생성함.
#   - Conv2D_Module_5: 입력 RDM 이미지(예: 기존 62x50, 다중 분류용이면 128x32 등)를 대상으로
#     2D CNN 처리를 수행하여 특징 벡터를 추출
#   - Conv1D_Module_2: Conv2D의 출력을 시퀀스 형태로 재구성한 후, 1D CNN 및 FC 계층으로 최종 분류 수행
#
# 입력:
#   - inputs: 모델에 입력되는 tensor, shape는 RDM_prepare 등에서 만들어짐 (예: (batch, channels, height, width))
#
# 출력:
#   - output: 최종 분류 결과 (원래는 단일 스칼라, 다중 분류 시 클래수 개수로 수정 필요)
#   - 현재 마지막에 torch.sigmoid()를 적용하여 0~1 범위의 값으로 변환 (다중 분류라면 softmax로 변경)
class RDTNet(nn.Module):
    def __init__(self, channels_block) -> None:
        super().__init__()
        # -------------------------------------------------------------------------
        # Conv2D_Net: 2D CNN 모듈, 예를 들어 channels_block[0] 구조 사용
        self.Conv2D_Net = Conv2D_Module_5(channels_block[0])
        # Conv1D_Net: 1D CNN 모듈, 예를 들어 channels_block[1] 구조 사용
        self.Conv1D_Net = Conv1D_Module_2(channels_block[1])
        # feature_num: Conv1D 모듈에 입력되는 채널 수 (channels_block[1][0])
        self.feature_num = channels_block[1][0]

    def forward(self, inputs):
        # 입력: tensor shape (batch, channels, height, width)
        batch_size = inputs.shape[0] // 24
        # 1. 2D CNN 처리: RDM 이미지에서 공간적 특징 추출
        data = self.Conv2D_Net(inputs)  # 출력 shape: (batch, feature) → 실제 feature vector
        # 2. 재구성: 2D CNN 출력의 채널을 1D 시퀀스로 재구성함
        #    여기서 16은 시퀀스 길이(예: 여러 프레임에 해당하는 시간적 정보)로 고정되어 있으므로,
        #    다중 분류를 위해 RDM 데이터 크기나 데이터 구성에 따라 이 부분도 수정 필요.
        data = torch.reshape(data, (-1, self.feature_num, 24))
        # 3. 1D CNN 처리: 재구성된 시퀀스 데이터에 대해 특징 추출 후 분류 수행
        output = self.Conv1D_Net(data)
        # print("Conv2D_Net output.shape:", output.shape)
        # 4. 최종 활성화: 현재 sigmoid를 사용 (이진 분류용) → 다중 분류 시 softmax나 다른 활성화 함수 사용 고려
        output = torch.softmax(output,dim = 1)
        return output
