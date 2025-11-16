import time
import Box_judging
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamPredictor
import warnings

warnings.filterwarnings("ignore")
# 判断点是否在四边形内（使用向量叉积方法）
def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# 输入： 原图 + 蒙版 -》 输出： 图像最终结果（数组形式）
def show_mask(image, mask, box_coords):
    # 创建一个黑色背景的图像
    result_image = np.zeros_like(image)

    # 获取矩形框的四个顶点坐标
    polygon = [(box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]),
               (box_coords[4], box_coords[5]), (box_coords[6], box_coords[7])]

    # 循环遍历每个像素，判断是否在矩形框内
    for i in range(image.shape[0]):  # 遍历图像的每一行
        for j in range(image.shape[1]):  # 遍历图像的每一列
            if is_point_in_polygon((j, i), polygon):  # 判断该点是否在框内
                # 如果在框内，则根据掩码的值设置颜色
                if mask[i, j] == 1:  # 前景区域
                    result_image[i, j] = [255, 255, 255]  # 白色
                else:  # 背景区域
                    result_image[i, j] = [0, 0, 0]  # 黑色
            else:
                # 如果不在框内，则设为黑色
                result_image[i, j] = [0, 0, 0]

    return result_image


def calculate_min_square(x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n):
    # 计算矩形的中心点
    cx = (x0n + x1n + x2n + x3n) / 4
    cy = (y0n + y1n + y2n + y3n) / 4

    # 获取四个点的最小值和最大值，确定矩形的边界
    min_x = min(x0n, x1n, x2n, x3n)
    max_x = max(x0n, x1n, x2n, x3n)
    min_y = min(y0n, y1n, y2n, y3n)
    max_y = max(y0n, y1n, y2n, y3n)

    # 计算矩形的宽度和高度
    width = max_x - min_x
    height = max_y - min_y

    # 判断是否能覆盖64x64的正方形
    if width <= 64 and height <= 64:
        return 64
    else:
        # 计算最小的正方形边长，确保覆盖矩形
        square_size = max(width, height)
        return square_size


# SAM模型初始化
sam_checkpoint = "sam_vit_b_01ec64.pth"  
model_type = "vit_b"  
device = "cuda"  # "cpu" or "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)  
predictor = SamPredictor(sam)  

current_dir = os.path.dirname(os.path.abspath(__file__))
# 定义文件夹路径
input_dir = os.path.join(current_dir, "dataset", "imgs")
txt_O_dir = os.path.join(current_dir, "dataset", "original_txt")  
output_dir = os.path.join(current_dir, "dataset", "buffer")  
txt_selected_dir = os.path.join(current_dir, "dataset", "txt_topk")

if not os.path.exists(txt_selected_dir):
    os.makedirs(txt_selected_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

counter = 0
top_k = 5  # 定义要保留的top K值   

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):

        # O_prompt_path
        base_filename_O = os.path.splitext(filename)[0]
        txt_filename_O = f"{base_filename_O}_O.txt"  # 对应的txt文件名
        txt_file_path_O = os.path.join(txt_O_dir, txt_filename_O)
        input_points_O = []
        input_labels_O = []  # 用来存储每个点的标签（背景点为0，对象点为1）
        value_list = []


        with open(txt_file_path_O, 'r') as f:
            for i, line in enumerate(f):
                coords = list(map(float, line.split()))
                if len(coords) == 10:
                    # 提取四个背景点坐标
                    x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n, target_x, target_y = coords

                    # 将背景点作为prompt输入到SAM
                    input_points_O = np.array([[x0n, y0n], [x1n, y1n], [x2n, y2n], [x3n, y3n], [target_x, target_y]])
                    input_labels_O = np.array([0, 0, 0, 0, 1])  # 背景点标签
                    # 获取图片路径并读取
                    photoPath = os.path.join(input_dir, filename)
                    image = cv2.imread(photoPath)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

                    predictor.set_image(image)
                    masks_with_O_prompt, scores, logits = predictor.predict(
                        point_coords=input_points_O,
                        point_labels=input_labels_O,
                        multimask_output=False,
                    )

                    # 生成预测掩码后，绘制矩形框
                    imageResult_with_prompt = image.copy()

                    # 将掩码应用到图像
                    mask = masks_with_O_prompt[0]  
                    
                    imageResult_with_prompt = show_mask(imageResult_with_prompt, mask,
                                                        (x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n))

                    pil_image = Image.fromarray(imageResult_with_prompt)  # 转换为PIL图像
                    imageResult_with_prompt_rgb = pil_image.convert('RGB')
                    square_size = calculate_min_square(x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n)
                    h, w, _ = imageResult_with_prompt.shape
                    cx = (x0n + x1n + x2n + x3n) / 4
                    cy = (y0n + y1n + y2n + y3n) / 4
                    half_size = square_size / 2
                    x1_crop = max(0, int(cx - half_size))
                    y1_crop = max(0, int(cy - half_size))
                    x2_crop = min(w, int(cx + half_size))
                    y2_crop = min(h, int(cy + half_size))

                if square_size > 0:  # 确保裁剪区域有正方形大小
                    # 裁剪图像
                    h, w, _ = imageResult_with_prompt.shape
                    cx = (x0n + x1n + x2n + x3n) / 4
                    cy = (y0n + y1n + y2n + y3n) / 4
                    half_size = square_size / 2
                    x1_crop = max(0, int(cx - half_size))
                    y1_crop = max(0, int(cy - half_size))
                    x2_crop = min(w, int(cx + half_size))
                    y2_crop = min(h, int(cy + half_size))

                    if (x2_crop - x1_crop) == (y2_crop - y1_crop) and (x2_crop - x1_crop) > 0:
                        cropped_image = imageResult_with_prompt_rgb.crop((x1_crop, y1_crop, x2_crop, y2_crop))
                        cropped_image_array = np.array(cropped_image)
                        # 保存裁剪后的图像到output_dir
                        temp_filename = f"{base_filename_O}_crop_{i}.png"
                        cropped_image_path = os.path.join(output_dir, temp_filename)
                        cropped_image.save(cropped_image_path)

                        # 调用SAM_check函数并获取value值
                        value = Box_judging.SAM_check(cropped_image_path)  
                        # 将value和对应的索引i保存到列表
                        value_list.append((value, i, cropped_image_path))
                        print(value, i)
                        if os.path.exists(cropped_image_path):
                            os.remove(cropped_image_path)
                    else:
                        print(f"Skipping {filename}: The cropped area is not a valid square.")
                else:
                    print(f"Skipping {filename}: The calculated square size is invalid.")

        # 排序并获取top K
        value_list.sort(key=lambda x: x[0])  # 按value升序排序
        top_k_indices = [x[1] for x in value_list[:top_k]]  # 提取出前K个索引
        print(f"File {filename} Top {top_k} Indices: {top_k_indices}")
        # 删除除了top K外的图像
        for _, _, path in value_list[top_k:]:
            if os.path.exists(path):
                os.remove(path) 
            # 保存_selected.txt文件
            selected_lines = []
            with open(txt_file_path_O, 'r') as f:
                lines = f.readlines()
                for idx in top_k_indices:
                    line = lines[idx]
                    parts = line.split()
                    # 保留前 8/10 个数据，并用逗号分隔
                    # selected_line = ','.join(parts[:8]) + ',crack\n'  # 添加换行符
                    selected_line = ' '.join(parts[:10]) + '\n'  # 添加换行符
                    selected_lines.append(selected_line)

            selected_txt_path = os.path.join(txt_selected_dir, f"{base_filename_O}_O.txt")
            with open(selected_txt_path, 'w') as f:
                f.writelines(selected_lines)

            print(f"Processed {filename}, saved top K images and selected lines.")
