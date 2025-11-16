import os
import numpy as np
import cv2
from sklearn.metrics import f1_score

# 计算IoU（前景 + 背景）
def calculate_iou_big(pred_mask, gt_mask):
    # 前景 IoU
    foreground_intersection = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    foreground_union = np.sum(np.logical_or(pred_mask == 1, gt_mask == 1))
    foreground_iou = foreground_intersection / foreground_union if foreground_union != 0 else 0

    # 背景 IoU
    background_intersection = np.sum(np.logical_and(pred_mask == 0, gt_mask == 0))
    background_union = np.sum(np.logical_or(pred_mask == 0, gt_mask == 0))
    background_iou = background_intersection / background_union if background_union != 0 else 0

    # 总体 IoU：前景和背景 IoU 的平均值
    total_iou = (foreground_iou + background_iou) / 2
    return total_iou

# 计算IoU（前景）
def calculate_iou_small(pred_mask, gt_mask):
    # 交集：预测为前景且真实标签也为前景的区域
    intersection = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    # 并集：预测为前景或真实标签为前景的区域
    union = np.sum(np.logical_or(pred_mask == 1, gt_mask == 1))
    # 计算IoU，防止除以0
    iou = intersection / union if union != 0 else 0
    return iou

# 计算F1的函数
def calculate_f1(pred_mask, gt_mask):
    # True Positives
    tp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))  # 预测为前景且确实是
    # False Positives
    fp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 0))  # 预测为前景但其实不是
    # False Negatives
    fn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 1))  # 预测为背景但其实是前景

    # Precision 和 Recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return f1


def calculate_metrics(predict_dir, gt_dir):
    f1_list = []
    iou_big_list = []
    iou_small_list = []

    predict_files = os.listdir(predict_dir)
    for file_name in predict_files:
        if file_name.endswith('.png'):
            # 加载预测掩码和真实掩码
            pred_mask = cv2.imread(os.path.join(predict_dir, file_name), cv2.IMREAD_GRAYSCALE) / 255

            # 生成真实掩码文件名
            gt_file_name = file_name.replace('.png', '.png')
            gt_mask = cv2.imread(os.path.join(gt_dir, gt_file_name), cv2.IMREAD_GRAYSCALE) / 255

            # 计算F1
            f1 = calculate_f1(pred_mask, gt_mask)
            f1_list.append(f1)

            # 计算IoU（前景 + 背景）
            iou_big = calculate_iou_big(pred_mask, gt_mask)
            iou_big_list.append(iou_big)

            # 计算IoU（前景）
            iou_small = calculate_iou_small(pred_mask, gt_mask)
            iou_small_list.append(iou_small)

    # 计算所有指标的平均值
    avg_f1 = np.mean(f1_list) if f1_list else 0
    avg_iou_big = np.mean(iou_big_list) if iou_big_list else 0
    avg_iou_small = np.mean(iou_small_list) if iou_small_list else 0

    return avg_f1, avg_iou_big, avg_iou_small

# 设置文件夹路径
predict_dir = r'C:\Users\86181\Desktop\sim2teal\Rebuttal\AAAI\Stage2_RES_DILATE'
gt_dir = r'C:\Users\86181\Desktop\sim2teal\Rebuttal\drive\DRIVE\test\gt'

# 计算平均指标
avg_f1, avg_iou_big, avg_iou_small = calculate_metrics(predict_dir, gt_dir)
print(avg_iou_big)
print(avg_iou_small)
avg = (avg_iou_big + avg_iou_small) / 2.0
print(f"Average F1: {avg_f1:.5f}")
print(f"Average IoU: {avg:.5f}")
