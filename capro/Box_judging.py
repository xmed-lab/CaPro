from skimage.color import rgb2gray
import torch
import torch.nn as nn
from torch.autograd import Variable
import VQGAN
# Torchvision
import torchvision.transforms as transforms
import cv2
# Matplotlib
import matplotlib.pyplot as plt

# OS
import argparse
from common.logger import AverageMeter
from common.evaluation import Evaluator


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def create_model(Model_Type):
    if Model_Type == "AE":
        autoencoder = Autoencoder()
        #print_model(autoencoder.encoder, autoencoder.decoder)
    elif Model_Type == "VQVAE":
        # autoencoder = Autoencoder()
        autoencoder = VQGAN.VQGAN()
        #print_model(autoencoder.encoder, autoencoder.decoder) 

    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
    return autoencoder


Model_Type = "VQVAE"
autoencoder = create_model(Model_Type)
Load_path = "Pretraining_64_net.pth"
autoencoder.load_state_dict(torch.load(Load_path))
autoencoder = autoencoder.cuda() 
autoencoder.eval()   
Img_size = 64 
criterion_mse = nn.MSELoss()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(Img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])


def SAM_check(Img_ptah): 

    image = cv2.imread(Img_ptah)
    image = cv2.resize(image ,(Img_size,Img_size))  
    image = rgb2gray(image)
    image = cv2.normalize(image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    det_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).cuda()
            
    with torch.no_grad():
        output,_ ,qloss = autoencoder(det_img)
    qloss = qloss.item()
    return qloss


