# %% Project imports and functions
import cv2 
import numpy as np
from glob import glob
from tqdm import tqdm, trange

from sklearn.cluster import KMeans

# for typing
from numpy import ndarray

DATA_FOLDER = 'data2'
OUT_FOLDER = 'output2'
PLAY_FOLDER = 'playground2'
RANDOM_STATE = 3

#%% Functions

def load_images(img_folder: str, max_n: int = 50) -> tuple:
    '''Loads all the images from img folder and stores them into 2 separate lists (client and server)'''
    assert max_n <= 50, 'there is at most 50 images to load'

    client = [
        [cv2.imread(f'{img_folder}/client/obj{n}_t1.JPG')]
        for n in trange(1, max_n+1, desc=f'Loading {max_n} client images')
    ]
    
    server = [
        [
            cv2.imread(img_path) for img_path in glob(f'{img_folder}/server/obj{n}_[0-9].JPG')
        ] for n in trange(1, max_n+1, desc=f'Loading {max_n} server images')
    ]
    return client, server


def get_desc_list(obj_desc_list: list) -> ndarray:
    '''
    Returns a flattened list of all the descriptors in the obj_desc_list.
    :param desc: list containing all the descriptors for each object
    :return: flattened array of all the descriptors
    '''
    return np.array([desc for obj_desc in obj_desc_list for desc in obj_desc])


def extract_desc(obj_imgs_list):
    obj_desc_list = [[] for _ in obj_imgs_list]
    n_desc = 0
    n_imgs = 0

    for i, obj_imgs in tqdm(list(enumerate(obj_imgs_list))):
        for img in obj_imgs:
            _, desc = sift.detectAndCompute(img, None)
            obj_desc_list[i].extend(desc)

            n_imgs += 1
            n_desc += len(desc)
    return obj_desc_list, n_desc, n_imgs


class HI:
    def __init__(self, b: int, depth: int, random_state: int = RANDOM_STATE) -> None:
        self.counter = 0
        self.b = b
        self.max_depth = depth
        self.random_state = random_state

    def build_tree(self, data: ndarray) -> None:
        self.counter = 0
        self.tree = self.hi_kmeans(data, self.b)

    def hi_kmeans(self, data: ndarray, b: int, depth: int = 0)  -> dict:
        '''
        Builds a hierarchical tree using the given keypoints.
        :return: hierarchical tree as a dictionary
        '''
        tree_dict = {i: {} for i in range(b)}

        KM = KMeans(
            n_clusters=b, 
            random_state=self.random_state, 
            n_init=4
        )
        KM.fit(X=data)

        centroids = KM.cluster_centers_
        labels = KM.labels_

        for centroid_index, centroid in enumerate(centroids):
            tree_dict[centroid_index]['centroid'] = centroid
            chosen_idxs = np.where(labels == centroid_index)[0]

            if len(chosen_idxs) <= b or self.max_depth == depth + 1:
                tree_dict[centroid_index]['is_leaf'] = True
                tree_dict[centroid_index]['vis_word_index'] = self.counter
                self.counter += 1
                print(f'Leaf {self.counter}/{np.power(b, self.max_depth)}', end='\r')
            else:
                tree_dict[centroid_index]['is_leaf'] = False
                selected_pts = data[chosen_idxs]
                tree_dict[centroid_index]['subtree'] = self.hi_kmeans(selected_pts, b, depth + 1)
                
        return tree_dict

    def tree_pass(self, query_desc: ndarray, tree: dict) -> int:
        '''
        Passes down the tree using a certain descriptor vector and returns the key of the leaf.
        :param query_desc: vector containing a descriptor for an img
        :param tree: hierarchical tree you want to descend from the root till a leaf node
        :return: index of the visual word the descriptor belongs to
        '''
        best_dist = +np.inf
        best_cluster_index = -1
        
        for cluster_index in range(self.b):
            dist = np.linalg.norm(query_desc - tree[cluster_index]['centroid'])
            if dist < best_dist:
                best_dist = dist
                best_cluster_index = cluster_index      
        if tree[best_cluster_index]['is_leaf']:
            return tree[best_cluster_index]['vis_word_index']
        return self.tree_pass(query_desc, tree[best_cluster_index]['subtree'])

    def get_query_tf_vector(self, descs: ndarray) -> ndarray:
        '''
        Returns the tf vector for a query image.
        :param img: query image
        :param perc_desc: percentage of descriptors to use
        :return: tf vector with shape (n_vis_words, 1)
        '''
        # sift feature extraction
        vis_words = np.zeros(self.counter)
        for desc in descs:
            vis_words[self.tree_pass(desc, self.tree)] += 1
        tf = vis_words / np.sum(vis_words)  # shape (n_vis_words, )
        return tf[:, None] # shape (n_vis_words, 1)

    def recall_rate(self, obj_desc_list: list, topKbest: int = 1, perc_desc: float = 1.0, sim: str = 'l1') -> float:
        '''
        Returns the recall rate for the given parameters.
        :param K: number of images to consider for the query
        :param topKbest: number of best predictions to consider for the recall rate
        :param perc_desc: percentage of descriptors to consider for the query
        :return: recall rate
        '''
        recall_t = 0
        for object_index, object_descs in enumerate(object_desc_list):
            perc_index = int(perc_desc * len(object_descs))
            tf = self.get_query_tf_vector(object_descs[:perc_index])

            query_tfidf = tf * self.idf  # shape (n_vis_words, 1)
            server_tfidf = self.weights  # shape (n_vis_words, n_objects)
            
            if sim == 'l1':
                scores = HI.l1(query_tfidf, server_tfidf)
            elif sim == 'l2':
                scores = HI.l2(query_tfidf, server_tfidf)
            elif sim == 'cos':
                scores = HI.cosine(query_tfidf, server_tfidf)
            else:
                raise ValueError('Invalid similarity function')
            
            if sim == 'cos':
                preds = np.argsort(scores)[::-1][:topKbest].tolist()
            else:
                preds = np.argsort(scores)[:topKbest].tolist()

            if object_index in preds:
                recall_t += 1
        
        return recall_t / len(obj_desc_list)

    @staticmethod
    def l1(x: ndarray, y: ndarray) -> float:
        return np.sum(np.abs(x - y), axis=0)
    
    @staticmethod
    def l2(x: ndarray, y: ndarray) -> float:
        return np.linalg.norm(x - y, axis=0)
    
    @staticmethod
    def cosine(x: ndarray, y: ndarray) -> float:
        return np.tensordot(x, y, ([0],[0])) / (np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0))

    def get_server_TFIDF_weights(self, server_object_desc):
        '''
        Returns the TFIDF weights for the server images.
        :param server_desc: dictionary containing all the descriptors for each image
        :return weights: TFIDF weights for the server images
        :return idf: inverse document frequency vector
        '''
        n_objects = len(server_object_desc)
        n_vis_words = self.counter

        vis_words_count = np.zeros((n_vis_words, n_objects))  # shape (n_vis_words, n_objects)

        for obj_index, obj_desc in enumerate(server_object_desc):
            for desc in obj_desc:
                desc_vis_word_index = self.tree_pass(desc, self.tree)
                vis_words_count[desc_vis_word_index][obj_index] += 1
        
        tf = vis_words_count / np.sum(vis_words_count, axis=0, keepdims=True)  # shape (n_vis_words, n_objects)

        idf = np.log(n_objects / np.sum(np.array(vis_words_count, dtype=bool), axis=1, keepdims=True))  # shape (n_vis_words, 1)
        
        weights = tf * idf  # shape (n_vis_words, n_objects)

        self.weights, self.idf = weights, idf

    
# %% Data loading and feature extraction

# Load client and server images
client_object_imgs, server_object_imgs = load_images(DATA_FOLDER, max_n=10)

# Notice that for the server images:
# - object 26 and 38 only have 2 images
# - object 37 has 4 images
# - all other objects have 3 images

# %% Create SIFT detector
# have to choose the parameters!
sim = 'l1'
edgeT = 0.2
contrastT = 0.1
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT)

# %% Extract descriptor for client images
client_obj_desc, n_client_desc, n_client_imgs = extract_desc(client_object_imgs)

# %% Extract descriptor for server images
server_obj_desc, n_server_desc, n_server_imgs = extract_desc(server_object_imgs)

# %% Print average number of features
print('The average number of features is:'
    +f'\n server images: {int(n_server_desc / n_server_imgs)}'
    +f'\n client images: {int(n_client_desc / n_client_imgs)}'
)

# %% Building the Vocabulary Tree using b = 4 and depth = 3
b = 4
depth = 3

perc_descr = 1.0
HI_ob = HI(b, depth)
HI_ob.build_tree(data=get_desc_list(server_obj_desc))

# %% Compute TFIDF weights 
HI_ob.get_server_TFIDF_weights(server_obj_desc)

# %% Querying and recall score
top_1_recall = HI_ob.recall_rate(client_obj_desc, topKbest=1, perc_desc=perc_descr, sim=sim)
top_5_recall = HI_ob.recall_rate(client_obj_desc, topKbest=5, perc_desc=perc_descr, sim=sim)

print(f'Top-1 recall rate: {top_1_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} descriptors ({perc_descr * 100}% of them)')
print(f'Top-5 recall rate: {top_5_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} descriptors ({perc_descr * 100}% of them)')

# %% Building the Vocabulary Tree using b=4 and depth=5
b = 4
depth = 5
perc_descr = 1.0
HI_ob = HI(b, depth)
HI_ob.build_tree(data=get_desc_list(server_obj_desc))

# Compute TFIDF weights 
HI_ob.get_server_TFIDF_weights(server_obj_desc)

# Querying and recall score
top_1_recall = HI_ob.recall_rate(client_obj_desc, topKbest=1, perc_desc=perc_descr, sim=sim)
top_5_recall = HI_ob.recall_rate(client_obj_desc, topKbest=5, perc_desc=perc_descr, sim=sim)

print(f'Top-1 recall rate: {top_1_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')
print(f'Top-5 recall rate: {top_5_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')

# %% Building the Vocabulary Tree using b=5 and depth=7
depth = 7
b = 5
HI_ob = HI(b, depth)
HI_ob.build_tree(data=get_desc_list(server_obj_desc))

# Compute TFIDF weights 
HI_ob.get_server_TFIDF_weights(server_obj_desc)

# Querying and recall score
top_1_recall = HI_ob.recall_rate(client_obj_desc, topKbest=1, perc_desc=1.0, sim=sim)
top_5_recall = HI_ob.recall_rate(client_obj_desc, topKbest=5, perc_desc=1.0, sim=sim)

print(f'Top-1 recall rate: {top_1_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')
print(f'Top-5 recall rate: {top_5_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, {perc_descr * 100}% of desc per query')

top_1_recall_90_perc = HI_ob.recall_rate(client_obj_desc, topKbest=1, perc_desc=0.9, sim=sim)
top_1_recall_70_perc = HI_ob.recall_rate(client_obj_desc, topKbest=1, perc_desc=0.7, sim=sim)
top_1_recall_50_perc = HI_ob.recall_rate(client_obj_desc, topKbest=1, perc_desc=0.5, sim=sim)

top_5_recall_90_perc = HI_ob.recall_rate(client_obj_desc, topKbest=5, perc_desc=0.9, sim=sim)
top_5_recall_70_perc = HI_ob.recall_rate(client_obj_desc, topKbest=5, perc_desc=0.7, sim=sim)
top_5_recall_50_perc = HI_ob.recall_rate(client_obj_desc, topKbest=5, perc_desc=0.5, sim=sim)

print(f'Top-1 recall rate: {top_1_recall_90_perc} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 90% of desc per query')
print(f'Top-1 recall rate: {top_1_recall_70_perc} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 70% of desc per query')
print(f'Top-1 recall rate: {top_1_recall_50_perc} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 50% of desc per query')

print(f'Top-5 recall rate: {top_5_recall_90_perc} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 90% of desc per query')
print(f'Top-5 recall rate: {top_5_recall_70_perc} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 70% of desc per query')
print(f'Top-5 recall rate: {top_5_recall_50_perc} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {n_server_desc} training descriptors, 50% of desc per query')

# %%
