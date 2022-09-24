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

def load_data(img_folder: str, n: int = 50) -> dict:
    '''Loads all the images from img folder and stores them into 2 separate dictionaries (client and server)'''
    imgs = {
        'client': {},
        'server': {}
    }
    for folder in os.listdir(img_folder):
        folder_loc = f'{img_folder}/{folder}'
        for img in os.listdir(folder_loc):
            if get_obj_number(img) <= n:
                imgs[folder][img] = cv2.imread(f'{folder_loc}/{img}')
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
    '''
    Extracts the object number from the image name
    :param img_name: name of the image
    :param start: string that precedes the object number
    :param end: string that follows the object number
    :return: object number
    '''
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
    '''
    Returns a list of all the descriptors in the server_desc_dict.
    :param server_desc_dict: dictionary containing all the descriptors for each image
    :return: list of all the descriptors
    '''
    l = list()
    for key in server_desc_dict.keys():
        for desc in server_desc_dict[key]:
            l.append(desc)
    return np.array(l)
    
class HI:
    def __init__(self, b, depth):
        self.counter = 0
        self.b = b
        self.depth = depth

    def build_tree(self, data: list):
        self.tree = self.hi_kmeans(data)

    def hi_kmeans(self, data: list, current_depth: int = 0):
        '''
        Builds a hierarchical tree using the given keypoints.
        :return: hierarchical tree as a dictionary
        '''
        tree_dict = {i: {} for i in range(self.b)}


        KM = KMeans(n_clusters=self.b, random_state=0, n_init=4)
        KM.fit(X=data)

        centroids = KM.cluster_centers_
        labels = KM.labels_

        for centroid_number, centroid in enumerate(centroids):
            tree_dict[centroid_number]['centroid'] = centroid
            

            chosen_idxs = np.where(labels == centroid_number)[0]

            if len(chosen_idxs) <= self.b or self.depth == current_depth + 1:
                tree_dict[centroid_number]['is_leaf'] = True
                tree_dict[centroid_number]['vis_word_index'] = self.counter
                print(f'Leaf {self.counter + 1}/{np.power(self.b, self.depth)}')
                self.counter += 1
                continue
            else:
                tree_dict[centroid_number]['is_leaf'] = False
                selected_pts = data[chosen_idxs]
                tree_dict[centroid_number]['subtree'] = self.hi_kmeans(selected_pts, current_depth=current_depth + 1)
                
        return tree_dict

    def tree_pass(self, query_desc: ndarray, tree: dict) -> int:
        '''
        Passes down the tree using a certain descriptor vector and returns the key of the leaf.
        :param query_desc: vector containing a descriptor for an img
        :param tree: hierarchical tree you want to descend from the root till a leaf node
        :return: index of the visual word the descriptor belongs to
        '''
        best_dist = +np.inf
        best_cluster = -1
        for cluster_number in range(self.b):
            dist = np.linalg.norm(query_desc - tree[cluster_number]['centroid'])
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster_number
        if tree[best_cluster]['is_leaf']:
            return tree[best_cluster]['vis_word_index']
        return self.tree_pass(query_desc, tree[best_cluster]['subtree'])

    def get_query_tf_vector(self, img: ndarray, perc_desc: float = 1) -> ndarray:
        '''
        Returns the tf vector for a query image.
        :param img: query image
        :param perc_desc: percentage of descriptors to use
        :return: tf vector
        '''
        # sift feature extraction
        _, desc = sift.detectAndCompute(img, None)
        desc = desc[:int(perc_desc * len(desc))]
        vis_words = np.zeros(self.counter)
        for d in desc:
            vis_words[self.tree_pass(query_desc=d, tree=self.tree)] += 1
        tf = vis_words / sum(vis_words)
        return tf

    def recall_rate(self, K: int, topKbest: int = 1, perc_desc: float = 1.0, sim: str = 'l1') -> float:
        '''
        Returns the recall rate for the given parameters.
        :param K: number of images to consider for the query
        :param topKbest: number of best predictions to consider for the recall rate
        :param perc_desc: percentage of descriptors to consider for the query
        :return: recall rate
        '''
        recall_t = 0
        for i in tqdm(range(1, K + 1)):
            tf = self.get_query_tf_vector(client_imgs[f'obj{i}_t1.JPG'], perc_desc=perc_desc)

            query_tfidf = tf * self.idf
            query_tfidf = np.reshape(query_tfidf, newshape=(-1,1))
            sim_mtx = np.abs(self.weights - query_tfidf)
            scores = np.sum(sim_mtx, axis=0)
            final = []
            for idx, score in enumerate(scores):
                final.append((score, idx + 1))
            preds = [x[1] for x in sorted(final)[:topKbest]]
            if i in preds:
                recall_t += 1
        return recall_t / K

    def get_server_TFIDF_weights(self, server_desc):
        '''
        Returns the TFIDF weights for the server images.
        :param server_desc: dictionary containing all the descriptors for each image
        :return weights: TFIDF weights for the server images
        :return idf: inverse document frequency vector
        '''
        K = len(server_desc.keys())
        n_vis_words = self.counter
        server_scores = {'vis_words_count': {}, 'tf': {}, 'tot_vis_words': {}}
        pre_idf = np.zeros(shape=n_vis_words)
        idf_found = np.zeros(shape=(K, n_vis_words))
        for obj_number in server_desc.keys():
            server_scores['vis_words_count'][obj_number] = np.zeros(shape=n_vis_words)
            for desc in tqdm(server_desc[obj_number]):
                idx = self.tree_pass(query_desc=desc, tree=self.tree)
                server_scores['vis_words_count'][obj_number][idx] += 1
                if idf_found[obj_number - 1][idx] == 0:
                    pre_idf[idx] += 1
                    idf_found[obj_number - 1][idx] += 1
            
            server_scores['tf'][obj_number] = server_scores['vis_words_count'][obj_number] / len(server_desc[obj_number])

        idf = K / pre_idf

        
        weights = np.zeros(shape=(n_vis_words, K))
        for i in range(n_vis_words):
            for j in range(1, K + 1):
                weights[i][j - 1] = server_scores['tf'][j][i] * np.log(idf[i])
                
        self.weights, self.idf = weights, idf

    
# %% Data loading and feature extraction

# Load client and server images
client_imgs, server_imgs = load_data('data2', n=50)


# Create SIFT detector
# have to choose the parameters!
edgeT = 10
contrastT = 0.12
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT)

client_desc = {}
server_desc = {}

# Extract descriptor for client images
n_client_desc = 0
n_client_imgs = 0
for img_name in tqdm(client_imgs.keys()):
    n_client_imgs += 1
    obj_number = get_obj_number(img_name)
    _, desc = sift.detectAndCompute(client_imgs[img_name], None)
    client_desc[obj_number] = desc
    n_client_desc += len(desc)

# Extract descriptor for server images
n_server_desc = 0 
n_server_imgs = 0 
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

# %% Building the Vocabulary Tree using b = 4 and depth = 3

b = 4
depth = 3
perc_descr = 1.0
HI_ob = HI(b, depth)
HI_ob.build_tree(data=get_descr_list(server_desc))

# Compute TFIDF weights 
HI_ob.get_server_TFIDF_weights(server_desc)

# Querying and recall score
K = len(server_desc.keys())
top_1_recall = HI_ob.recall_rate(K, topKbest=1, perc_desc=perc_descr)
top_5_recall = HI_ob.recall_rate(K, topKbest=5, perc_desc=perc_descr)

print(f'Top-1 recall rate: {top_1_recall} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} descriptors ({perc_descr * 100}% of them)')
print(f'Top-5 recall rate: {top_5_recall} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} descriptors ({perc_descr * 100}% of them)')

# %% Building the Vocabulary Tree using b=4 and depth=5
b = 4
depth = 5
perc_descr = 1.0
HI_ob = HI(b, depth)
HI_ob.build_tree(data=get_descr_list(server_desc))

# Compute TFIDF weights 
HI_ob.get_server_TFIDF_weights(server_desc)

# Querying and recall score
K = len(server_desc.keys())
top_1_recall = HI_ob.recall_rate(K, topKbest=1, perc_desc=perc_descr)
top_5_recall = HI_ob.recall_rate(K, topKbest=5, perc_desc=perc_descr)

print(f'Top-1 recall rate: {top_1_recall} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')
print(f'Top-5 recall rate: {top_5_recall} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')

# %% Building the Vocabulary Tree using b=5 and depth=7
depth = 7
HI_ob = HI(b, depth)
HI_ob.build_tree(data=get_descr_list(server_desc))

# Compute TFIDF weights 
HI_ob.get_server_TFIDF_weights(server_desc)

# Querying and recall score
K = len(server_desc.keys())
top_1_recall = HI_ob.recall_rate(K, topKbest=1, perc_desc=1.0)
top_5_recall = HI_ob.recall_rate(K, topKbest=5, perc_desc=1.0)

print(f'Top-1 recall rate: {top_1_recall} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')
print(f'Top-5 recall rate: {top_5_recall} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')

top_1_recall_90_perc = HI_ob.recall_rate(K, topKbest=1, perc_desc=0.9)
top_1_recall_70_perc = HI_ob.recall_rate(K, topKbest=1, perc_desc=0.7)
top_1_recall_50_perc = HI_ob.recall_rate(K, topKbest=1, perc_desc=0.5)

top_5_recall_90_perc = HI_ob.recall_rate(K, topKbest=5, perc_desc=0.9)
top_5_recall_70_perc = HI_ob.recall_rate(K, topKbest=5, perc_desc=0.7)
top_5_recall_50_perc = HI_ob.recall_rate(K, topKbest=5, perc_desc=0.5)

print(f'Top-1 recall rate: {top_1_recall_90_perc} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 90% of desc per query')
print(f'Top-1 recall rate: {top_1_recall_70_perc} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 70% of desc per query')
print(f'Top-1 recall rate: {top_1_recall_50_perc} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 50% of desc per query')

print(f'Top-5 recall rate: {top_5_recall_90_perc} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 90% of desc per query')
print(f'Top-5 recall rate: {top_5_recall_70_perc} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 70% of desc per query')
print(f'Top-5 recall rate: {top_5_recall_50_perc} using b = {b} and depth = {depth}, {K} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 50% of desc per query')

# %%
