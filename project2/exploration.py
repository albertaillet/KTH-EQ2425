# %% Project imports and functions
import os
from random import random
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans
from tqdm import tqdm
import re



# for typing
from numpy import ndarray
from typing import List, Union
detector_type = Union[cv2.xfeatures2d_SIFT, cv2.xfeatures2d_SURF]

DATA_FOLDER = 'data2'
OUT_FOLDER = 'output2'
PLAY_FOLDER = 'playground2'

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

def get_obj_number(img_name: str, start: str = 'obj', end: str = '_') -> int:
    result = re.search(f'{start}(.*){end}', img_name)
    return int(result.group(1))

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

def hi_kmeans(data: list, b: int, depth: int):
    '''
    Builds a hierarchical tree using the given keypoints.
    :param b: number of cluster for each iteration Tree building algorithm.
    :param depth: number of iterations to build the Tree.
    expl: https://github.com/epignatelli/scalable-recognition-with-a-vocabulary-tree/blob/master/_test_Scalable%20Recognition%20with%20a%20Vocabulary%20Tree%20copy.ipynb
    '''

    Kmeans = KMeans(n_clusters=b, random_state=0)
    tree_struc = {}
    for t in range(depth):
        new_labels = KMeans.fit_predict(X=data)

def get_descr_list(server_desc_dict):
    l = list()
    for key in server_desc_dict.keys():
        for desc in server_desc_dict[key]:
            l.append(desc)
    return l

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

client_kp = {}
client_desc = {}
server_kp = {}
server_desc = {}
# %%
# detect keypoints in client images 
for img_name in tqdm(client_imgs.keys()):
    obj_number = get_obj_number(img_name)
    kp, desc = sift.detectAndCompute(client_imgs[img_name], None)
    client_kp[obj_number] = kp
    client_desc[obj_number] = desc

# %%
# store keypoints and descriptors in a dictionary based on object number
for img_name in tqdm(server_imgs.keys()):
    obj_number = get_obj_number(img_name)
    if obj_number not in server_kp.keys():
        server_kp[obj_number] = []
        server_desc[obj_number] = []
    
    kp, desc = sift.detectAndCompute(server_imgs[img_name], None)
    server_kp[obj_number] += kp
    for d in desc:
        server_desc[obj_number].append(d)

# %%
# compute the avg number of features per object
avg_features_client = 0
avg_features_server = 0
for obj in client_kp.keys():
    avg_features_client += len(client_kp[obj])
    avg_features_server += len(server_kp[obj]) / 3

print(f'The average number of features is:\nserver images: {int(avg_features_server / 50)}\nclient images: {int(avg_features_client / 50)}')

# %%
server_desc_list = get_descr_list(server_desc)
