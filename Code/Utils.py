import torch.nn as nn
import torch
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def modify_DeiT():
    DeiT = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    net = nn.Sequential(DeiT)
    net[0].head = nn.Linear(in_features=192, out_features=64, bias=True)
    del net[0].blocks[1:]
    net[0].blocks
    #print(net)
    return net

def preprocess(state):
    img = state.astype(np.float32) 
    #step1: a common formula converting  converting RGB images into gray images: 
    #img_grey = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114  #shape = (210,160,1)  
    img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #step2: Crop out the non-game area of the image 
    image_gameArea = img_grey[0:172,:]      #shape = (172,160,1)  
    #step3: reshape into 84*84
    image_small = cv2.resize(image_gameArea, (224, 224), interpolation=cv2.INTER_AREA)  # shape(84,84)
    return image_small
    
def process_pre_states(states):
    imgs = [preprocess(s) for s in states]
    imgs = np.array(imgs)
    return torch.from_numpy(imgs).unsqueeze(0)
