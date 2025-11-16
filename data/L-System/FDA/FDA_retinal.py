import shutil
import numpy as np
from PIL import Image, ImageEnhance
from __init__ import FDA_source_to_target_np
from pathlib import Path
import os
import math
import cv2
import random

def _rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop):
    """
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        rotation_degree: The degree of rotation on the image.
        do_crop: Do cropping if it is True.
    Returns:
        A rotated image.
    """
    # Rotate the given image with the given rotation degree
    if rotation_degree != 0:

        #image = image.rotate(math.radians(rotation_degree), Image.NEAREST, expand=False)
        image = image.rotate(rotation_degree, Image.NEAREST, expand=False)

        # Center crop to ommit black noise on the edges
        if do_crop == True:
            lrr_width, lrr_height = _largest_rotated_rect(output_height, output_width, math.radians(rotation_degree))
            box = (output_width / 2.0 - lrr_width / 2.0, (output_height - lrr_height) / 2,
                   output_width / 2.0 + lrr_width / 2.0, output_height / 2.0 + lrr_height / 2.0)
            resized_image = image.crop(box)
            image = resized_image.resize((output_height, output_width))

    return image

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    #row, col, _ = X_imgs[0].shape
    row, col, _ = X_imgs.shape
    # Gaussian distribution parameters
    mean = 0
    var = 1
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col))  # np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(X_imgs.shape, np.float32)

    if len(X_imgs.shape) == 2:
        noisy_image = X_imgs + gaussian
    else:
        noisy_image[:, :, 0] = X_imgs[:, :, 0] + gaussian
        noisy_image[:, :, 1] = X_imgs[:, :, 1] + gaussian
        noisy_image[:, :, 2] = X_imgs[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

def _largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = h if (w < h) else w
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def replace_color(img, src_clr = [255,255,255],dst_clr = [255,255,255]):
    img_arr = np.asarray(img, dtype=np.double)
    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()
    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]
    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img

def replace_color_binary(img, src_clr = [0,0,0],dst_clr = [255,255,255]):
    img_arr = np.asarray(img, dtype=np.double)
    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()
    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]
    r_img[img > src_color] = dst_clr[0]
    g_img[img > src_color] = dst_clr[1]
    b_img[img > src_color] = dst_clr[2]
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1, 2, 0)
    return dst_img

current_dir = os.path.dirname(os.path.abspath(__file__))
txt_src_files = os.path.join(current_dir, "../txt_data")
tar_file = os.path.join(current_dir, "Single_image")
src_file = os.path.join(current_dir, "../fake_gtvessel_thin")
project_root = Path(current_dir).parent.parent.parent
Save_image = project_root / "data" / "RoLabelImg_Transform" / "img"
Save_txt = project_root / "data" / "RoLabelImg_Transform" / "txt"

if not os.path.exists(Save_image):
    os.makedirs(Save_image)
# if not os.path.exists(Save_label):
#     os.makedirs(Save_label)
if not os.path.exists(Save_txt):
    os.makedirs(Save_txt)
    
files_src = os.listdir(src_file)
files_tar = os.listdir(tar_file)
counter = 0

for i in range(len(files_src)):
    tarlist = random.sample(files_tar, 1)
    for j in range(len(tarlist)):
        vessel_image_path = os.path.join(src_file, files_src[i])
        target_domain_path = os.path.join(tar_file, tarlist[j])
        im_src = Image.open(vessel_image_path).convert('RGB')
        im_trg = Image.open(target_domain_path).convert('RGB')
        
        save_image_path = os.path.join(Save_image, f"{counter}.jpg")
        # save_label_path = os.path.join(Save_label, f"{counter}.gif")  # 注释掉label保存

        im_src = im_src.resize((512, 512), Image.BICUBIC)
        im_trg = im_trg.resize((512, 512), Image.BICUBIC)
        
        # Turn on/off rotation augmentation
        # angle = np.random.uniform(0, 180)
        angle = 0
        im_src = _rotate_and_crop(im_src, 512, 512, angle, True)
        im_src = np.asarray(im_src, np.float32)
        im_src_nochange = im_src.copy()
        im_trg = np.asarray(im_trg, np.float32)
        
        # print("img_src", im_src.shape)
        im_src = im_src.transpose((2, 0, 1))
        im_src_nochange = im_src_nochange.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        
        # print("im_src", np.max(im_src))
        # print("im_trg", np.max(im_trg))
        
        L = np.random.uniform(0.2, 0.3)
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=L)
        
        # im_label = im_src_nochange.copy()
        
        src_in_trg = src_in_trg.transpose((1, 2, 0))
        p_bular = random.random()

        if p_bular > 0.5:
            src_in_trg = cv2.GaussianBlur(src_in_trg, (13, 13), 0)
        else:
            src_in_trg = cv2.GaussianBlur(src_in_trg, (3, 3), 0)

        """""
        if p_bular > 0.9:
            src_in_trg = cv2.GaussianBlur(src_in_trg, (3, 3), 0)
        """""
        
        p_noise = random.random()
        
        # im_label = im_label.transpose((1, 2, 0))
        # im_label = replace_color(im_label)
        # im_label = replace_color_binary(im_label)
        
        # print("src_in_trg", np.max(src_in_trg))
        # print("src_in_trg_shape", src_in_trg.shape)
        # print("im_label_max", np.max(im_label)) 

        img_FDA = np.clip(src_in_trg, 0, 255.)
        # im_label = np.clip(im_label, 0, 255.)  
        
        p_constrast = random.random()
        if p_constrast > 0.8:
            Contrast = np.random.uniform(0.9, 1.05)
            # Contrast = np.random.uniform(0.8, 0.95)
            PIL_image = Image.fromarray(img_FDA.astype('uint8'))
            # PIL_image = ImageEnhance.Contrast(PIL_image).enhance(0.5)
            PIL_image = ImageEnhance.Contrast(PIL_image).enhance(Contrast)
            img_FDA = np.asarray(PIL_image)
            
        Image.fromarray((img_FDA).astype('uint8')).convert('RGB').save(save_image_path)
        # print(f"counter = {counter}, src_file = {files_src[i]}")
        
        base_filename = os.path.splitext(files_src[i])[0]
        txt_file_path = os.path.join(txt_src_files, base_filename + '.txt')

        if os.path.exists(txt_file_path):  
            xml_dest_path = os.path.join(Save_txt, f"{counter}.txt")
            shutil.copy(txt_file_path, xml_dest_path)
            # print(f"Copied txt file: {txt_file_path} -> {xml_dest_path}")
        else:
            print(f"Warning: TXT file not found: {txt_file_path}")

        counter += 1