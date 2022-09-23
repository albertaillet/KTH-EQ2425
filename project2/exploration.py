# %% Project imports and functions
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
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
    return imgs['client'], imgs['server']

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

def get_descr_list(server_desc_dict: dict) -> ndarray:
    l = list()
    for key in server_desc_dict.keys():
        for desc in server_desc_dict[key]:
            l.append(desc)
    return np.array(l)

def tree_pass(query_desc, tree, b):
    '''Passes down the tree and return the key of the leaf'''
    best_dist = +np.inf
    best_cluster = -1
    for cluster_number in range(b):
        dist = np.linalg.norm(query_desc - tree[cluster_number]['centroid'])
        if dist < best_dist:
            best_dist = dist
            best_cluster = cluster_number
    if tree['centroids_are_leaf']:
        return tree[best_cluster]['vis_word_index']
    return tree_pass(query_desc, tree[best_cluster]['subtree'], b)

def get_query_tf_vector(img):
    # sift feature extraction
    _, desc = sift.detectAndCompute(img, None)
    vis_words = np.zeros(np.power(b, depth))
    for d in desc:
        vis_words[tree_pass(query_desc=d, tree=tree, b=b)] += 1
    tf = vis_words / sum(vis_words)
    return tf

class HI:
    def __init__(self):
        self.counter = 0

    def hi_kmeans(self, data: list, b: int, depth: int, current_depth: int = 0):
        # probably must use recursive programming for efficient implementation, and a class
        '''
        Builds a hierarchical tree using the given keypoints.
        :param b: number of cluster for each iteration Tree building algorithm.
        :param depth: number of iterations to build the Tree.
        https://github.com/epignatelli/scalable-recognition-with-a-vocabulary-tree/blob/master/_test_Scalable%20Recognition%20with%20a%20Vocabulary%20Tree%20copy.ipynb
        https://github.com/Pranshu258/svtor
        '''

        tree_dict = {i: {} for i in range(b)}
        tree_dict['centroids_are_leaf'] = False

        if len(data) < b or depth == current_depth:
            print(f'Leaf {self.counter}/{np.power(b, depth)}')
            return None, True

        KM = KMeans(n_clusters=b, random_state=0, n_init=4)
        KM.fit(X=data)

        centroids = KM.cluster_centers_
        labels = KM.labels_

        for centroid_number, centroid in enumerate(centroids):
            tree_dict[centroid_number]['centroid'] = centroid
            chosen_idxs = np.where(labels == centroid_number)[0]
            selected_pts = data[chosen_idxs]
            tree_dict[centroid_number]['subtree'], is_leaf = self.hi_kmeans(selected_pts, b=b, depth=depth, current_depth=current_depth + 1)
            if is_leaf:
                tree_dict['centroids_are_leaf'] = True
                tree_dict[centroid_number]['vis_word_index'] = self.counter
                self.counter += 1
        return tree_dict, False
# %% Data loading and feature extraction

client_imgs, server_imgs = load_data('data2')

# Create SIFT detector
# have to choose the parameters!
edgeT = 10
contrastT = 0.12
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT)

client_desc = {}
server_desc = {}

# client images
n_client_desc = 0
n_client_imgs = 0
for img_name in tqdm(client_imgs.keys()):
    n_client_imgs += 1
    obj_number = get_obj_number(img_name)
    _, desc = sift.detectAndCompute(client_imgs[img_name], None)
    client_desc[obj_number] = desc
    n_client_desc += len(desc)

# server images
n_server_desc = 0 # total number of descriptors
n_server_imgs = 0 # total server images
for img_name in tqdm(server_imgs.keys()):
    obj_number = get_obj_number(img_name)
    n_server_imgs += 1
    if obj_number not in server_desc.keys():
        server_desc[obj_number] = []
    _, desc = sift.detectAndCompute(server_imgs[img_name], None)
    for d in desc:
        server_desc[obj_number].append(d)
    n_server_desc += len(desc)
    

print(f'The average number of features is:\nserver images: {int(n_server_desc / n_server_imgs)}\nclient images: {int(n_client_desc / n_client_imgs)}')

# %%
    
server_desc_list = get_descr_list(server_desc)
HI_ob = HI()
b = 4
depth = 5
tree, _ = HI_ob.hi_kmeans(data=server_desc_list, b=b, depth=depth)

# %%

server_scores = {'vis_words_count': {}, 'tf': {}, 'tot_vis_words': {}}
pre_idf = np.zeros(shape=(np.power(b,depth)))
K = len(server_desc.keys())
idf_found = np.zeros(shape=(K, np.power(b, depth)))

for obj_number in server_desc.keys():
    server_scores['vis_words_count'][obj_number] = np.zeros(shape=(np.power(b, depth)))
    for desc in tqdm(server_desc[obj_number]):
        idx = tree_pass(query_desc=desc, tree=tree, b=b)
        server_scores['vis_words_count'][obj_number][idx] += 1
        if idf_found[obj_number - 1][idx] == 0:
            pre_idf[idx] += 1
            idf_found[obj_number - 1][idx] += 1
    
    server_scores['tf'][obj_number] = server_scores['vis_words_count'][obj_number] / len(server_desc[obj_number])
idf = K / pre_idf

# %% Get weights

n_vis_words = np.power(b, depth)
weights = np.zeros(shape=(n_vis_words, K))
for i in range(n_vis_words):
    for j in range(1, K + 1):
            weights[i][j - 1] = server_scores['tf'][j][i] * np.log(idf[i])

# print(client_imgs.keys())
# tf = get_query_tf_vector(client_imgs['obj9_t1.JPG'])
# query_tfidf = tf * idf
# query_tfidf = np.reshape(query_tfidf, newshape=(-1,1))
# sim_mtx = np.abs(weights - query_tfidf)
# final = np.sum(sim_mtx, axis=0)
# final_idx = np.argmin(final) + 1
# print(final_idx)

score = 0
for i in tqdm(range(1,K + 1)):
    tf = get_query_tf_vector(client_imgs[f'obj{i}_t1.JPG'])
    query_tfidf = tf * idf
    query_tfidf = np.reshape(query_tfidf, newshape=(-1,1))
    sim_mtx = np.abs(weights - query_tfidf)
    final = np.sum(sim_mtx, axis=0)
    final_idx = np.argmin(final) + 1
    print(f'{final_idx} -- {i}')
    if final_idx == i:
        score += 1
print(f'recall rate: {score / (K + 1)}')

 # %%
