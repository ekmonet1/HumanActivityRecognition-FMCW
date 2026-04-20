#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_action_log_nopandas.py
- pandas 없이 표준 라이브러리 + matplotlib 만 사용
- 사용 예:
  python viz_action_log_nopandas.py --log D:/path/your_log.txt --outdir ./out --xaxis ts
  (xaxis: ts | frame)
"""

import re
import os
import csv
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

import matplotlib.pyplot as plt

CLASSES = ["fall", "squat", "stand", "sit", "walk_away", "walk_toward", "none"]

HDR_RE = re.compile(
    r'^\[(?P<ts>\d{2}:\d{2}:\d{2})\]\s*Frame\s*(?P<frame>\d+):\s*(?P<label>\w+)\s*\((?P<conf>[\d.]+)%\)'
)
PROB_RE = re.compile(r'^\s*Probabilities:\s*(?P<body>.+)$')
PAIR_RE = re.compile(r'([A-Za-z_]+)\s*:\s*([\d.]+)%')

COLORS = {
    "fall": "#e41a1c",
    "squat": "#377eb8",
    "stand": "#4daf4a",
    "sit": "#984ea3",
    "walk_away": "#ff7f00",
    "walk_toward": "#a65628",
    "none": "#999999",
}

def to_sec(ts_str: str) -> int:
    t = datetime.strptime(ts_str, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

def parse_log(lines: List[str]) -> List[Dict]:
    rows = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        m = HDR_RE.match(line)
        if m:
            ts_str = m.group("ts")
            frame = int(m.group("frame"))
            label = m.group("label")
            conf = float(m.group("conf"))

            probs = {c: 0.0 for c in CLASSES}  # 기본 0
            if i + 1 < n:
                nxt = lines[i+1].rstrip("\n")
                pm = PROB_RE.match(nxt)
                if pm:
                    body = pm.group("body")
                    for k, v in PAIR_RE.findall(body):
                        k = k.strip()
                        if k in probs:
                            probs[k] = float(v)
                    i += 1  # 확률 라인 소비

            row = {
                "ts_str": ts_str,
                "frame": frame,
                "pred": label,
                "pred_conf": conf,
                **{f"p_{c}": probs[c] for c in CLASSES},
            }
            rows.append(row)
        i += 1

    # 정렬
    rows.sort(key=lambda r: r["frame"])
    if not rows:
        return rows

    base = to_sec(rows[0]["ts_str"])
    for r in rows:
        r["t_sec"] = to_sec(r["ts_str"]) - base
        # argmax
        best_c, best_v = None, -1.0
        for c in CLASSES:
            v = r[f"p_{c}"]
            if v > best_v:
                best_v, best_c = v, c
        r["argmax_cls"] = best_c
    return rows

def find_segments(rows: List[Dict], by_key="pred") -> List[Dict]:
    if not rows:
        return []
    segs = []
    cur = rows[0][by_key]
    start_idx = 0
    for idx in range(1, len(rows)):
        if rows[idx][by_key] != cur:
            seg = rows[start_idx:idx]
            segs.append({
                "label": cur,
                "start_idx": start_idx,
                "end_idx": idx - 1,
                "start_frame": seg[0]["frame"],
                "end_frame": seg[-1]["frame"],
                "start_t_sec": seg[0]["t_sec"],
                "end_t_sec": seg[-1]["t_sec"],
                "length_frames": seg[-1]["frame"] - seg[0]["frame"] + 1,
                "length_sec": seg[-1]["t_sec"] - seg[0]["t_sec"],
            })
            cur = rows[idx][by_key]
            start_idx = idx
    seg = rows[start_idx:]
    segs.append({
        "label": cur,
        "start_idx": start_idx,
        "end_idx": len(rows) - 1,
        "start_frame": seg[0]["frame"],
        "end_frame": seg[-1]["frame"],
        "start_t_sec": seg[0]["t_sec"],
        "end_t_sec": seg[-1]["t_sec"],
        "length_frames": seg[-1]["frame"] - seg[0]["frame"] + 1,
        "length_sec": seg[-1]["t_sec"] - seg[0]["t_sec"],
    })
    return segs

def plot_pred_over_time(rows: List[Dict], outpath: str, xaxis: str):
    if not rows:
        return
    x = [r["t_sec"] if xaxis == "ts" else r["frame"] for r in rows]
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = [class_to_idx[r["pred"]] for r in rows]

    plt.figure(figsize=(12, 4))
    # step plot
    plt.step(x, y, where='post', linewidth=2, color="#333333", alpha=0.6)
    # scatter by class color
    scatter_colors = [COLORS[r["pred"]] for r in rows]
    plt.scatter(x, y, c=scatter_colors, s=16, alpha=0.9)

    plt.yticks(list(range(len(CLASSES))), CLASSES)
    plt.xlabel("Time (s)" if xaxis == "ts" else "Frame")
    plt.title("Predicted Class over Time")
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_probs_over_time(rows: List[Dict], outpath: str, xaxis: str):
    if not rows:
        return
    x = [r["t_sec"] if xaxis == "ts" else r["frame"] for r in rows]

    plt.figure(figsize=(12, 6))
    for c in CLASSES:
        y = [r[f"p_{c}"] for r in rows]
        plt.plot(x, y, label=c, linewidth=1.8, alpha=0.9, color=COLORS[c])
    plt.ylim(-2, 102)
    plt.ylabel("Probability (%)")
    plt.xlabel("Time (s)" if xaxis == "ts" else "Frame")
    plt.title("Per-Class Probabilities over Time")
    plt.legend(ncol=4, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def write_csv(path: str, headers: List[str], rows_list: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows_list:
            # 누락 키는 빈칸 방지
            w.writerow({k: r.get(k, "") for k in headers})



# ====== 유틸: 이동평균(스무딩) ======
def _moving_avg(x, win=7):
    x = np.asarray(x, dtype=float)
    if win <= 1 or len(x) == 0:
        return x
    # 가장자리 왜곡 줄이려고 가장자리 복제 패딩 후 same 컨볼루션
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=float) / win
    y = np.convolve(xp, k, mode="same")[pad:-pad]
    return y

# ====== 2) Top-k 영역 그래프 (rows 버전) ======
def plot_topk_area_rows(rows, outpath, xaxis="ts", k=3, win=7, classes=None, colors=None):
    if classes is None:
        classes = ["fall","squat","stand","sit","walk_away","walk_toward","none"]
    if colors is None:
        colors = {
            "fall": "#e41a1c","squat": "#377eb8","stand": "#4daf4a","sit": "#984ea3",
            "walk_away": "#ff7f00","walk_toward": "#a65628","none": "#999999",
        }

    x = [r["t_sec"] if xaxis=="ts" else r["frame"] for r in rows]
    # 클래스별 확률 배열 만들고 스무딩
    smoothed = {}
    for c in classes:
        vals = [r[f"p_{c}"] for r in rows]
        smoothed[c] = _moving_avg(vals, win=win)

    # 시점별 상위 k만 남기기
    topk_vals = []
    for t in range(len(rows)):
        cand = [(c, smoothed[c][t]) for c in classes]
        cand.sort(key=lambda x: x[1], reverse=True)
        keep = {c for c, _ in cand[:k]}
        top_vals = [smoothed[c][t] if c in keep else 0.0 for c in classes]
        topk_vals.append(top_vals)

    Y = np.array(topk_vals).T  # [num_classes, T]
    keep_idx = np.where(Y.sum(axis=1) > 0)[0]
    kept_classes = [classes[i] for i in keep_idx]
    Y = Y[keep_idx, :]

    plt.figure(figsize=(12,6))
    plt.stackplot(x, Y, labels=kept_classes, colors=[colors[c] for c in kept_classes], alpha=0.85)
    plt.ylim(0, 100)
    plt.ylabel(f"Top-{k} Prob. Sum (%)")
    plt.xlabel("Time (s)" if xaxis=="ts" else "Frame")
    plt.title(f"Top-{k} Class Probabilities (smoothed, win={win})")
    plt.legend(ncol=4, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ====== 3) 전환 시점 세로선 + 라벨 (rows 버전) ======
def plot_pred_with_transitions_rows(rows, outpath, xaxis="ts", classes=None, colors=None, annotate_max=12):
    if classes is None:
        classes = ["fall","squat","stand","sit","walk_away","walk_toward","none"]
    if colors is None:
        colors = {
            "fall": "#e41a1c","squat": "#377eb8","stand": "#4daf4a","sit": "#984ea3",
            "walk_away": "#ff7f00","walk_toward": "#a65628","none": "#999999",
        }
    class_to_idx = {c:i for i,c in enumerate(classes)}
    x = [r["t_sec"] if xaxis=="ts" else r["frame"] for r in rows]
    y = [class_to_idx[r["pred"]] for r in rows]

    plt.figure(figsize=(12,4))
    plt.step(x, y, where='post', linewidth=2, color="#333")
    plt.scatter(x, y, c=[colors[r["pred"]] for r in rows], s=14, alpha=0.9)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Time (s)" if xaxis=="ts" else "Frame")
    plt.title("Predicted Class with Transition Markers")
    plt.grid(True, axis='x', linestyle='--', alpha=0.25)

    # 전환 인덱스 찾기
    changes = []
    for i in range(1, len(rows)):
        if rows[i]["pred"] != rows[i-1]["pred"]:
            changes.append(i)
    # 라벨 개수 제한
    if len(changes) > annotate_max:
        step = max(1, len(changes)//annotate_max)
        changes = changes[::step]

    for idx in changes:
        xv = x[idx]
        prev = rows[idx-1]["pred"]
        curr = rows[idx]["pred"]
        dt = float(rows[idx]["t_sec"] - rows[idx-1]["t_sec"])
        plt.axvline(x=xv, color="#999", linestyle=":", alpha=0.8)
        plt.text(xv, 0.2, f"{prev}→{curr}\nΔt={dt:.2f}s", rotation=90,
                 va="bottom", ha="center", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.8))

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ====== 4) 엔트로피(불확실성) 트랙 (rows 버전) ======
def plot_entropy_rows(rows, outpath, xaxis="ts", classes=None, win=7):
    if classes is None:
        classes = ["fall","squat","stand","sit","walk_away","walk_toward","none"]
    x = [r["t_sec"] if xaxis=="ts" else r["frame"] for r in rows]

    # 확률 [0,1]로, 스무딩 후 엔트로피
    P = []
    for c in classes:
        vals = [r[f"p_{c}"] for r in rows]
        vals = _moving_avg(vals, win=win) / 100.0
        P.append(vals)
    P = np.clip(np.array(P).T, 1e-9, 1.0)   # shape [T, C]
    H = -(P * np.log(P)).sum(axis=1)
    H_max = np.log(len(classes))
    H_norm = H / H_max * 100.0

    plt.figure(figsize=(12,2.8))
    plt.plot(x, H_norm, linewidth=2)
    plt.ylabel("Uncertainty (%)")
    plt.xlabel("Time (s)" if xaxis=="ts" else "Frame")
    plt.ylim(0, 100)
    plt.title(f"Prediction Uncertainty (Shannon Entropy, win={win})")
    plt.grid(True, linestyle='--', alpha=0.3)

    thr = np.percentile(H_norm, 70)
    plt.fill_between(x, 0, H_norm, where=(H_norm>=thr), alpha=0.15)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ====== 5) 전이 행렬 히트맵 (rows 버전) ======
def plot_transition_matrix_rows(rows, outpath, classes=None):
    if classes is None:
        classes = ["fall","squat","stand","sit","walk_away","walk_toward","none"]
    idx = {c:i for i,c in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=int)
    prev = rows[0]["pred"]
    for i in range(1, len(rows)):
        cur = rows[i]["pred"]
        if cur != prev:
            counts[idx[prev], idx[cur]] += 1
        prev = cur

    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = counts.sum(axis=1, keepdims=True)
        rates = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums>0)

    fig, ax = plt.subplots(1,2, figsize=(12,4), gridspec_kw={'width_ratios':[1,1.1]})
    im1 = ax[0].imshow(counts, cmap="Blues")
    ax[0].set_title("Transition Count")
    ax[0].set_xticks(range(len(classes))); ax[0].set_xticklabels(classes, rotation=45, ha='right')
    ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = counts[i,j]
            if v>0:
                ax[0].text(j, i, str(v), ha='center', va='center',
                           color=("black" if (row_sums[i]>0 and v<row_sums[i][0]/2+1) else "white"))
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)

    im2 = ax[1].imshow(rates*100.0, vmin=0, vmax=100, cmap="Oranges")
    ax[1].set_title("Transition Rate (%)")
    ax[1].set_xticks(range(len(classes))); ax[1].set_xticklabels(classes, rotation=45, ha='right')
    ax[1].set_yticks(range(len(classes))); ax[1].set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = rates[i,j]*100.0
            if v>0:
                ax[1].text(j, i, f"{v:.0f}", ha='center', va='center',
                           color=("black" if v<50 else "white"))
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="입력 로그 파일 경로")
    ap.add_argument("--outdir", default="./out", help="출력 디렉토리")
    ap.add_argument("--xaxis", choices=["ts", "frame"], default="ts",
                    help="x축 선택: ts(상대 초), frame(프레임 번호)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    rows = parse_log(lines)
    if not rows:
        print("[!] 파싱 결과가 비었습니다. 로그 형식을 확인하세요.")
        return

    # CSV 저장
    parsed_headers = (
        ["ts_str", "t_sec", "frame", "pred", "pred_conf", "argmax_cls"] +
        [f"p_{c}" for c in CLASSES]
    )
    parsed_csv = os.path.join(args.outdir, "parsed_log.csv")
    write_csv(parsed_csv, parsed_headers, rows)

    segs = find_segments(rows, by_key="pred")
    seg_headers = [
        "label","start_idx","end_idx","start_frame","end_frame",
        "start_t_sec","end_t_sec","length_frames","length_sec"
    ]
    seg_csv = os.path.join(args.outdir, "segments.csv")
    write_csv(seg_csv, seg_headers, segs)

    # 그림
    pred_png = os.path.join(args.outdir, "pred_class_over_time.png")
    probs_png = os.path.join(args.outdir, "probs_over_time.png")
    plot_pred_over_time(rows, pred_png, args.xaxis)
    plot_probs_over_time(rows, probs_png, args.xaxis)
    # ---- 추가 시각화 결과물 ---
    plot_topk_area_rows(rows, os.path.join(args.outdir, "top3_area.png"), xaxis=args.xaxis, k=3, win=7)
    plot_pred_with_transitions_rows(rows, os.path.join(args.outdir, "pred_with_transitions.png"), xaxis=args.xaxis, annotate_max=14)
    plot_entropy_rows(rows, os.path.join(args.outdir, "uncertainty_entropy.png"), xaxis=args.xaxis, win=7)
    plot_transition_matrix_rows(rows, os.path.join(args.outdir, "transition_matrix.png"))


    print("[완료]")
    print(f"- 프레임별 CSV: {parsed_csv}")
    print(f"- 세그먼트 CSV: {seg_csv}")
    print(f"- 예측 클래스 그림: {pred_png}")
    print(f"- 확률 변화 그림: {probs_png}")


if __name__ == "__main__":
    main()
