import shutil              # 파일 및 디렉토리 복사, 삭제 등을 위한 모듈
import time                # 시간 측정을 위한 모듈
import os                  # 파일/디렉토리 경로 조작 관련 모듈
import torch               # PyTorch 주요 모듈
from torch import optim    # 최적화(optimizer) 관련 모듈
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 학습률 스케줄러: 검증 손실이 감소하지 않을 때 학습률 감소
from torch.utils.data import DataLoader                # 데이터셋 관리를 위한 DataLoader
from torch.utils.tensorboard import SummaryWriter      # Tensorboard 로깅을 위한 모듈

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# 사용자 정의 모듈: 모델, 데이터셋, 손실함수, 시드 설정 등
from Network import RDTNet      # 네트워크(모델) 정의 모듈
from Dataset_reader import Fall_Dataset  # 데이터셋 구성 모듈 (현재는 fall 관련 이진 분류용)
from Function import BCEFocalLoss, seed_setting


# -------------------------------
# 재현성을 위한 시드 설정
# -------------------------------
seed = 11
seed_setting(seed)

# -------------------------------
# 파라미터 설정
# -------------------------------
EPOCHS = 200
TEST_BATCH = 32
TRAIN_BATCH = 32
# CH_LIST = [[1, 16, 32, 64, 128, 256], [256, 256, 256, 256]]
# CH_LIST = [[1, 32, 64, 128, 256, 512], [512, 512, 512, 512]]
# CH_LIST = [[1, 2, 4, 8, 16, 32], [32, 32, 32, 32]]
# CH_LIST = [[1, 8, 16, 32, 64, 128], [128, 128, 128, 128]]
CH_LIST = [[1, 64, 128, 256, 512, 1024], [1024, 1024, 1024, 1024]]
RANGE = 64
DOPPLER = 64
FRAME = 24
CLASSES =  ['fall','walk_away','walk_toward','squat','sit','stand' ,'none']
# CLASSES =  ['fall','walk_away','walk_toward','squat','sit','stand']
# -------------------------------
# 경로 설정
# -------------------------------
valid_dir = r"D:\Desktop\FallDetection-base-on-FMCW_DH\list_0618\test.txt"
train_dir = r"D:\Desktop\FallDetection-base-on-FMCW_DH\list_0618\train.txt"

MODEL_NUM = 54
PTH_PATH = 'model/RDTNet_model54.pth'

# -------------------------------
# 모드 설정: 현재 'verify' (추론) 모드로 지정되어 있음.
# 'train' 모드로 변경할 경우 학습 루프가 실행됨.
# ※ 다중 분류로 전환할 경우, 데이터셋과 라벨(예: fall 외 다른 동작)을 반영하도록 수정 필요
# -------------------------------
# state = 'train'
state = 'verify'

# GPU 선택: 특정 GPU 번호 지정 (여기서는 '2'번 GPU 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda")

def inference(model, testloader, device):
    model.eval()
    ll = 0
    sum_acc = 0
    classes = CLASSES
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_actual = {classname: 0 for classname in classes}
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    with torch.no_grad():
        for inputs, labels, _ in testloader:
            ll += len(inputs)
            inputs = torch.reshape(inputs, (len(inputs) * FRAME, 1, RANGE, DOPPLER))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for label, pred in zip(labels, preds):
                true_class = classes[label.item()]
                pred_class = classes[pred.item()]
                if label == pred:
                    correct_pred[true_class] += 1
                else:
                    false_positive[pred_class] += 1
                    false_negative[true_class] += 1
                total_pred[pred_class] += 1
                total_actual[true_class] += 1
            sum_acc += (preds == labels).sum().item()

    print("Inference 결과:")
    for classname in classes:
        TP = correct_pred[classname]
        FP = false_positive[classname]
        FN = false_negative[classname]
        total = total_actual[classname]

        accuracy = 100.0 * TP / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Class [{classname}] → "
              f"Acc: {accuracy:.2f}%, "
              f"Prec: {precision:.2f}, "
              f"Recall: {recall:.2f}, "
              f"F1: {f1_score:.2f}  "
              f"({TP}/{total if total > 0 else 1})")
        # print("# ---------------------------------------------------- #")
        
        acc = 100.0 * TP / total if total > 0 else 0.0
        # print(f"Class [{classname}] → Accuracy: {acc:.2f}% ({TP}/{total})")
    print("Overall Accuracy: {:.2f}%".format(100.0 * sum_acc / ll))


# =============================================================================
# 함수: train
# =============================================================================
# 입력:
#   - model: 학습할 신경망 모델
#   - trainloader: 학습 데이터셋 DataLoader
#   - train_batch_size: 학습 배치 크기
#   - epoch: 현재 에포크 번호 (학습 진행 상황 표시용)
#   - device: 모델과 데이터를 할당할 장치 (GPU/CPU)
#   - optimizer: 최적화 알고리즘 (예: SGD)
#   - criterion: 손실 함수 (현재 BCEFocalLoss; 다중 분류로 변경 시 CrossEntropyLoss 등으로 변경)
#
# 역할:
#   - 모델을 학습 모드로 전환한 후, DataLoader를 통해 받은 각 배치에 대해
#     데이터 reshape, 순전파(forward), 손실 계산, 역전파(backpropagation), 파라미터 업데이트 및 정확도 집계 등을 수행.
#
# 주의:
#   - 현재 입력 데이터는 torch.reshape를 통해 ((len(inputs))*16, 1, 62, 50)로 재구성함.
#     → 이는 기존 프레임 수 16과 RDM 크기 62×50에 기반합니다.
#     여러분이 다중 분류를 위해 RDM의 크기(예: 128×32) 혹은 프레임 수를 바꾼다면 이 reshape 부분을 수정해야 함.
#   - 출력 예측은 (outputs > 0.5)로 이진 분류 threshold를 적용하는데, 다중 분류로 변경 시 argmax 등을 사용해야 함.
#
# 출력:
#   - 모델(업데이트된 파라미터)과 평균 학습 손실을 반환.
def train(model, trainloader, train_batch_size, epoch, device, optimizer, criterion):
    model.train()    # 모델을 학습 모드로 전환 (dropout, BatchNorm 등 활성화)
    start_time = time.time()
    train_loss = 0   # 전체 배치의 누적 손실
    ll = 0           # 처리한 전체 데이터 개수
    sum_acc = 0      # 누적 정확도 (예측과 라벨이 동일한 경우 수)

    # DataLoader에서 각 배치 반복
    for batch_idx, (inputs, labels, _) in enumerate(trainloader):
        bs = len(inputs)            # 현재 배치 크기
        ll = ll + bs                # 전체 샘플 수 누적
        # 입력 데이터 reshape:
        # 기존에는 한 샘플당 16 프레임, 1 채널, 62×50 RDM 이미지로 구성됨.
        # 다중 분류 혹은 RDM 크기/프레임 수 변경 시 이 부분을 수정 필요!
        inputs = torch.reshape(inputs, ((len(inputs)) * FRAME, 1, RANGE, DOPPLER))
        # print("input: ",inputs.shape)  # 디버깅
        # 데이터를 device(GPU)로 이동
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()       # 기울기 초기화
        outputs = model(inputs)     # 순전파(forward) 실행
        # outputs = outputs.view(-1, FRAME, outputs.shape[-1])  # (batch_size, FRAME, classes)
        # outputs = torch.mean(outputs, dim=1)                  # (batch_size, classes)
        # print("outputs.shape: ", outputs.shape)
        # print("batch_size:", len(inputs) // FRAME)
        # print("FRAME:", FRAME)

        labels = labels.view(-1).long()  # CrossEntropyLoss expects 1D class indices

        # 손실 계산:
        # 현재 BCEFocalLoss를 사용하여 이진 분류 손실을 계산함.
        # 다중 분류라면 출력과 라벨의 shape, 손실 함수(예: CrossEntropyLoss) 등 수정 필요.
        loss = criterion(outputs, labels)

        # 출력에 대해 threshold 0.5를 적용하여 예측값 생성 (이진 분류용)
        # predict = (outputs > 0.5).data.squeeze()

        preds = torch.argmax(outputs, dim=1)
        loss.backward()             # 역전파(backpropagation)를 통한 gradient 계산
        optimizer.step()            # 파라미터 업데이트

        # 정확도 집계: 예측값과 라벨이 같은 경우 합산
        sum_acc += (preds == labels).sum().item()
        train_loss += loss.item()

        # 학습 진행 상황 출력: 전체 배치의 일정 비율마다 로그 출력
        interval = max(1,int(len(trainloader)/5))
        if batch_idx % interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}".format(
                    epoch,
                    ll,
                    len(trainloader.dataset),
                    100.0 * batch_idx / len(trainloader),
                    loss.data.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )

    # 평균 학습 손실 계산
    average_train_loss = train_loss / (len(trainloader.dataset) / train_batch_size)
    end_time = time.time()
    elapse_time = end_time - start_time
    accuracy = 100 * sum_acc / ll   # 학습 정확도 (%)
    print('average_train_loss', average_train_loss, 'elapse_time', elapse_time, 'train_acc:', accuracy)
    return model, average_train_loss


# =============================================================================
# 함수: test
# =============================================================================
# 입력:
#   - model: 평가할 모델 (검증 모드)
#   - testloader: 검증 데이터셋 DataLoader
#   - test_batch_size: 평가 시 배치 크기
#   - criterion: 손실 함수
#   - device: GPU/CPU
#   - best_acc: 현재까지 기록된 최고 정확도 (모델 저장 기준)
#   - best_fall_acc: 낙상(또는 특정 클래스) 정확도 (이진 분류 기준; 다중 분류의 경우 수정 필요)
#   - save_path: 모델 저장 경로 (새로운 최고 성능 시 모델 파일 저장)
#
# 역할:
#   - 모델을 평가 모드로 전환한 후, 데이터셋에 대해 순전파 실행, 손실 및 정확도 계산,
#     각 클래스별(현재는 'non-fall'와 'fall') 정확도 집계 및 로그 출력
#   - 기존 최고 정확도와 비교하여 모델 성능이 개선되었으면 파일로 저장
#
# 주의:
#   - 현재 클래스 리스트는 ['non-fall', 'fall']로 이진 분류에 맞게 설정되어 있음.
#     다중 분류로 전환 시 클래스 이름 리스트 및 정확도 집계 방식을 변경해야 함.
#
# 출력:
#   - 최신 best_acc, best_fall_acc, 그리고 평균 테스트 손실을 반환.
from collections import defaultdict

def test(model, testloader, test_batch_size, criterion, device, best_acc, best_fall_acc, save_path):
    model.eval()
    ll = 0
    sum_acc = 0
    test_loss = 0
    classes = CLASSES
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_actual = {classname: 0 for classname in classes}

    # 각 클래스별 FP(예측은 클래스인데 실제는 아님) 계산용
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    for i, (inputs, labels, _) in enumerate(testloader):
        ll += len(inputs)
        inputs = torch.reshape(inputs, (len(inputs) * FRAME, 1, RANGE, DOPPLER))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        preds = torch.argmax(outputs, dim=1)
        sum_acc += (preds == labels).sum().item()

        for label, pred in zip(labels, preds):
            true_class = classes[label.item()]
            pred_class = classes[pred.item()]
            if label == pred:
                correct_pred[true_class] += 1
            else:
                false_positive[pred_class] += 1   # FP: 예측은 맞았다고 했는데 실제는 아님
                false_negative[true_class] += 1   # FN: 실제로는 해당 클래스인데 예측은 다름

            total_pred[pred_class] += 1          # 예측된 개수
            total_actual[true_class] += 1        # 실제 해당 클래스 개수

    average_test_loss = test_loss / (len(testloader.dataset) / test_batch_size)

    print("Class-wise Metrics:")
    avg_acc = 0
    for classname in classes:
        TP = correct_pred[classname]
        FP = false_positive[classname]
        FN = false_negative[classname]
        total = total_actual[classname]

        accuracy = 100.0 * TP / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Class [{classname}] → "
            f"Acc: {accuracy:.2f}%, "
            f"Prec: {precision:.2f}, "
            f"Recall: {recall:.2f}, "
            f"F1: {f1_score:.2f}  "
            f"({TP}/{total if total > 0 else 1})")

        avg_acc += accuracy

    print("Average accuracy: {:.2f}%".format(avg_acc / len(classes)))
    test_acc = 100.0 * sum_acc / ll
    fall_acc = 100.0 * correct_pred["fall"] / total_actual["fall"] if total_actual["fall"] > 0 else 0
    # print('Test accuracy: %.4f ' % test_acc)
    print("# ---------------------------------------------------- #")

    # 모델 저장 조건 동일
    if best_acc < 99:
        if test_acc > best_acc:
            best_acc = test_acc
            best_fall_acc = fall_acc
            print('<<<<<<saving model(Acc rise)')
            torch.save(model.state_dict(), save_path)
        elif (abs(test_acc - best_acc) < 1e-5) & (fall_acc > best_fall_acc):
            print('<<<<<<saving model(Recall rates rise)')
            best_fall_acc = fall_acc
            torch.save(model.state_dict(), save_path)
        print('The best test accuracy is %.4f' % best_acc)
        print('Its fall accuracy is %.4f' % best_fall_acc)
        print(' ')
    else:
        if ((best_acc >= 99) & (fall_acc > best_fall_acc) & (test_acc >= 99)):
            print('<<<<<<saving model(Recall rates rise)')
            best_acc = test_acc
            best_fall_acc = fall_acc
            torch.save(model.state_dict(), save_path)
        elif (abs(fall_acc - best_fall_acc) < 1e-5) & (test_acc > best_acc):
            print('<<<<<<saving model(Acc rates rise)')
            best_acc = test_acc
            best_fall_acc = fall_acc
            torch.save(model.state_dict(), save_path)
        print(f"Current fall accuracy: {fall_acc:.4f}")
        print(f"The best test accuracy is {best_acc:.4f}")
        print(f"The best recorded fall accuracy is {best_fall_acc:.4f}")

        print(' ')

    return best_acc, best_fall_acc, average_test_loss



# =============================================================================
# 메인 실행부: 'verify'와 'train' 모드로 분기
# =============================================================================

# ==== [추가] 윈도(24프레임 묶음) 단위 CM 그리기 함수 ====
def _plot_cm(cm, classes, normalize=False, title="Confusion Matrix", out="cm.png"):
    if normalize:
        with np.errstate(all="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm.astype(np.float64), row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum!=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    fmt = ".2f" if normalize else "d"
    thresh = (cm.max() + cm.min()) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color="black" if val > thresh else "white", fontsize=9)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

# numpy만으로 혼동행렬/리포트
def _compute_cm_report(y_true, y_pred, classes):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    K = len(classes)

    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(np.float64), np.maximum(row_sum, 1), where=row_sum!=0)

    lines = []
    acc_overall = cm.trace() / max(cm.sum(), 1)
    for k, name in enumerate(classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        support = cm[k, :].sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        lines.append(f"{name:>12s} | P={prec:6.3f}  R={rec:6.3f}  F1={f1:6.3f}  Support={int(support)}")

    report = "\n".join(lines) + f"\nOverall Acc = {acc_overall*100:.2f}%"
    return cm, cm_norm, report


def confusion_window_level(model, loader, device, frame, r, d, classes, save_prefix="cm"):
    """
    모델 출력 모양 자동 감지:
      - (B, K) 이면 이미 윈도 평균된 결과 → 그대로 사용
      - (B*FRAME, K) 이면 (B, FRAME, K)로 바꿔 프레임 평균 후 사용
      - 그 외: (B*F, K)로 추정 가능하면 F를 추론해 평균
    """
    model.eval()
    K = len(classes)
    all_true, all_pred = [], []

    with torch.no_grad():
        for inputs, labels, _ in loader:
            B = len(inputs)
            x = torch.reshape(inputs, (B * frame, 1, r, d)).to(device)
            y = labels.to(device).view(-1)  # (B,)

            logits = model(x)  # 기대: (B, K) 또는 (B*FRAME, K)
            if logits.ndim != 2 or logits.shape[-1] != K:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}, K={K}")

            N = logits.shape[0]  # 배치 차원
            if N == B:
                # 이미 윈도 단위 결과
                logits_seq = logits
            elif N == B * frame:
                # 프레임 단위 결과 → 윈도 평균
                logits_seq = logits.view(B, frame, K).mean(dim=1)
            else:
                # (B*F, K) 일반화: F 추정
                if N % B == 0:
                    F = N // B
                    logits_seq = logits.view(B, F, K).mean(dim=1)
                    print(f"[info] detected frame count F={F} for this checkpoint")
                else:
                    raise RuntimeError(
                        f"Cannot infer window grouping: logits.shape={tuple(logits.shape)}, "
                        f"B={B}, FRAME={frame}"
                    )

            pred = torch.argmax(logits_seq, dim=1)  # (B,)
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    cm, cm_norm, report = _compute_cm_report(y_true, y_pred, classes)

    # 저장 (PNG + CSV)
    _plot_cm(cm,      classes, normalize=False,
             title="Confusion Matrix (Window-level, raw)",
             out=f"{save_prefix}_cm_window_raw.png")
    _plot_cm(cm_norm, classes, normalize=True,
             title="Confusion Matrix (Window-level, normalized)",
             out=f"{save_prefix}_cm_window_norm.png")

    # CSV도 같이 남기기
    def _save_cm_csv(arr, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("," + ",".join(classes) + "\n")
            for i, name in enumerate(classes):
                row = [name] + [str(x) for x in (arr[i].round(6) if arr.dtype.kind=='f' else arr[i])]
                f.write(",".join(row) + "\n")

    _save_cm_csv(cm,      f"{save_prefix}_cm_window_raw.csv")
    _save_cm_csv(cm_norm, f"{save_prefix}_cm_window_norm.csv")

    print(f"[saved] {save_prefix}_cm_window_raw.png / _norm.png / _raw.csv / _norm.csv")
    print("\n=== [Window-level] Report ===")
    print(report)
    return cm, cm_norm, report


if state == 'verify':
    # -------------------------
    # 검증 모드:
    # 1. 모델 생성 후 기존 저장된 가중치 로드.
    # 2. 테스트 데이터셋(Fall_Dataset)을 이용하여 inference() 실행.
    # ※ 이진 분류라 'non-fall'와 'fall'로 분류함.
    #     → 다중 분류 시 데이터셋, 모델 출력, 평가 방식 전반을 수정해야 함.
    # -------------------------
    model = RDTNet(CH_LIST)
    pth_path = PTH_PATH
    model_pth = torch.load(pth_path)
    model.load_state_dict(model_pth)
    model.to(device)

    # valid_dir = "D:\Desktop\FallDetection-base-on-FMCW_annotation\list\test.txt"
    val_dataset = Fall_Dataset(valid_dir)
    test_batch_size = TEST_BATCH
    test_loader = DataLoader(val_dataset, test_batch_size, shuffle=True)

        # 윈도(=24프레임 묶음) 단위 혼동행렬 생성
    base = os.path.splitext(os.path.basename(pth_path))[0]
    confusion_window_level(
        model, test_loader, device,
        frame=FRAME, r=RANGE, d=DOPPLER,
        classes=CLASSES,
        save_prefix=base
    )

    inference(model, test_loader, device)

elif state == 'train':
    # -------------------------
    # 학습 모드:
    # 1. 학습/검증 데이터를 불러오고 DataLoader 구성.
    # 2. Tensorboard 로깅 디렉토리 생성 및 기존 로그 삭제 (있을 경우).
    # 3. 모델 생성, 최적화, 손실 함수(BCEFocalLoss) 설정.
    # 4. 학습 및 검증 루프 실행 후 성능 개선 시 모델 저장.
    #
    # 주의:
    #   - 입력 데이터 reshape, BCEFocalLoss, 클래스 라벨(현재 이진 분류) 등은 다중 분류로 전환 시 수정 필요.
    # -------------------------
    bianhao = MODEL_NUM
    log_dir = 'runs/model' + str(bianhao)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # train_dir = r"D:\Desktop\FallDetection-base-on-FMCW_annotation\list\train.txt"
    # valid_dir = r"D:\Desktop\FallDetection-base-on-FMCW_annotation\list\test.txt"

    train_dataset = Fall_Dataset(train_dir)
    val_dataset = Fall_Dataset(valid_dir)
    train_batch_size = TRAIN_BATCH
    test_batch_size = TEST_BATCH

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    # print(f"train_dataset 개수: {len(train_dataset)}")
    # print(f"train_loader batch 개수: {len(train_loader)}")
    test_loader = DataLoader(val_dataset, test_batch_size, shuffle=True)

    channel_list = CH_LIST
    model = RDTNet(channel_list)  # 모델 생성 (기존 구조는 이진 분류용)
    model.eval()
    model.to(device=device)

    # 최적화 설정: SGD with learning rate, weight_decay 등
    # 
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    # BCEFocalLoss: 이진 분류용 손실 함수.
    # 다중 분류 시 CrossEntropyLoss 등으로 교체 및 출력 차원에 맞게 수정 필요.
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0      # 최고 정확도 초기값
    fall_acc = 0      # (현재는 특정 클래스 정확도; 수정 필요)
    saving_dir = 'model/'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # ReduceLROnPlateau: 검증 손실이 개선되지 않으면 학습률 감소
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(EPOCHS):
        model, train_loss = train(model, train_loader, train_batch_size, epoch, device, optimizer, criterion)

        if best_acc < 99:
            filename = saving_dir + 'RDTNet_model' + str(bianhao) + '.pth'
        else:
            filename = saving_dir + 'RDTNet_model' + str(bianhao) + '_epoch' + str(epoch) + '.pth'

        best_acc, fall_acc, average_test_loss = test(
            model, test_loader, test_batch_size, criterion, device, best_acc, fall_acc, filename
        )
        scheduler.step(average_test_loss)

        writer.add_scalar(tag="accuracy", scalar_value=best_acc, global_step=epoch)
        writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag="test_loss", scalar_value=average_test_loss, global_step=epoch)

    writer.close()