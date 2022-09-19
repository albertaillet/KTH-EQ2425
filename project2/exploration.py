# %% Project imports

import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm

# for typing
from numpy import ndarray
from typing import Union
detector_type = Union[cv2.xfeatures2d_SIFT, cv2.xfeatures2d_SURF]

DATA_FOLDER = 'data2'
OUT_FOLDER = 'output2'
PLAY_FOLDER = 'playground2'

#  %% Load data

def load_data(img_folder: str):
    '''Loads all the images from img folder and stores them into 2 separate dictionaries (client and server)'''
    imgs = {
        'client': {},
        'server': {}
    }
    for folder in os.listdir(img_folder):
        folder_loc = f'{img_folder}/{folder}'
        for img in os.listdir(folder_loc):
            img_loc = f'{folder_loc}/{img}'
            imgs[folder][img] = cv2.imread(img_loc)
    return imgs

def save_img(img: ndarray, name: str, kp: list=None, folder: str=PLAY_FOLDER):
    'Saves the image to disk, if given adds keypoints'
    if kp is not None:
        colors=[(0,255,0), (255,0,0)]
        for idx, k in enumerate(kp):
            img = cv2.drawKeypoints(img, keypoints=k, outImage=None, color=colors[idx])
    # save image using cv
    cv2.imwrite(f'{folder}/{name}.jpg', img)

def plot(x, y, xlabel, ylabel, title, xlim=None, filename=None):
    plt.plot(x, y, 'x-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
        plt.xticks(x[::3])
    plt.ylim(0, 1)
    plt.grid()

    if filename:
        plt.savefig(filename)
    
    plt.show()

def detect_keypoints(img: ndarray, detector: detector_type) -> list:
    return detector.detect(img, None)

# %% Data loading

imgs = load_data('data2')

# dicts with structure {'img_name': img as ndarray}
client_imgs = imgs['client']
server_imgs = imgs['server']

# %%

# Create SIFT detector
# have to choose the parameters!
edgeT = 10
contrastT = 0.125
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT)

# dicts with structure {'obj_number': list with SIFT keypoints}
client_kp = {}
server_kp = {}

# detect keypoints in client images 
for idx, name in tqdm(enumerate(client_imgs.keys())):
    obj_number = idx + 1
    kp = detect_keypoints(client_imgs[name], sift)
    client_kp[obj_number] = kp

for idx, name in tqdm(enumerate(server_imgs.keys())):
    obj_number = int(idx / 3) + 1
    if obj_number not in server_kp.keys():
        server_kp[obj_number] = []
    kp = detect_keypoints(server_imgs[name], sift)
    server_kp[obj_number] += kp
# %%
# compute the avg number of features per object
avg_features_client = 0
avg_features_server = 0
for obj in client_kp.keys():
    avg_features_client += len(client_kp[obj])
    avg_features_server += len(server_kp[obj]) / 3

print(f'The average number of features is:\nserver images: {int(avg_features_server / 50)}\nclient images: {int(avg_features_client / 50)}')



# %%
