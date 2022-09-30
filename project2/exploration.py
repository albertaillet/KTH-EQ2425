# %% Project imports and functions
import cv2 
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm, trange
from sklearn.decomposition import PCA

# for typing
from numpy import ndarray
from typing import Union, List, Dict, Optional

DATA_FOLDER = 'data2'
OUT_FOLDER = 'output2'
RANDOM_STATE = 1

#%% Functions

def load_images(img_folder: str, max_n: int = 50) -> tuple:
    '''Loads all the images from img folder and stores them into 3 separate lists (server, client and new_client)'''
    assert max_n <= 50, 'there is at most 50 images to load'

    server = [
        [
            cv2.imread(img_path) for img_path in glob(f'{img_folder}/server/obj{n}_[0-9].JPG')
        ]
        for n in trange(1, max_n+1, desc=f'Loading {max_n} server images')
    ]

    client = [
        [
            cv2.imread(f'{img_folder}/client/obj{n}_t1.JPG')
        ]
        for n in trange(1, max_n+1, desc=f'Loading {max_n} client images')
    ]
    

    new_client = {
        n-1 : [
            cv2.imread(f'{img_folder}/new_client/obj{n}_t1.JPG')
        ]
        for n in tqdm([42, 43, 44], desc='Loading new client images')
    }

    return server, client, new_client


def get_desc_list(obj_desc_list: list) -> ndarray:
    '''Returns a flattened list of all the descriptors in the obj_desc_list.
    :param desc: list containing all the descriptors for each object
    :return: flattened array of all the descriptors
    '''
    return np.array([desc for obj_desc in obj_desc_list for desc in obj_desc])


def extract_desc(obj_imgs_list: Union[list, dict], detector: cv2.xfeatures2d_SIFT):
    '''Extracts the descriptors from the images in the list or dictionary of object images.
    :param obj_imgs_list: list or dict of shape (n_objects, n_object_images, height, width, channels)
    :param detector: SIFT detector object.
    :return: list or dict of descriptors for each object, total number of descriptors, total number of images.
    '''
    if isinstance(obj_imgs_list, list):
        obj_desc_list = [[] for _ in obj_imgs_list]
        iterator = list(enumerate(obj_imgs_list))
    elif isinstance(obj_imgs_list, dict):
        obj_desc_list = {i: [] for i in obj_imgs_list.keys()}
        iterator = obj_imgs_list.items()
    else:
        raise TypeError('obj_imgs_list must be a list or a dict')
    
    n_desc = 0
    n_imgs = 0

    for i, obj_imgs in tqdm(iterator):
        for img in obj_imgs:
            _, desc = detector.detectAndCompute(img, None)
            obj_desc_list[i].extend(desc)

            n_imgs += 1
            n_desc += len(desc)
    return obj_desc_list, n_desc, n_imgs


class HI:
    def __init__(self, b: int, depth: int, random_state: int=RANDOM_STATE) -> None:
        self.counter = 0
        self.b = b
        self.max_depth = depth
        self.random_state = random_state

    def build_tree(self, data: ndarray, n_components: Union[int, float]=0) -> None:
        self.n_components = n_components
        if n_components != 0:
            self.pca = PCA(n_components)
            reduced_data = self.pca.fit_transform(data)
        else:
            reduced_data = data
        
        self.counter = 0
        self.tree = self.hi_kmeans(reduced_data, self.b)

    def hi_kmeans(self, data: ndarray, b: int, depth: int = 0)  -> dict:
        '''
        Builds a hierarchical tree using the given keypoints.
        :return: hierarchical tree as a dictionary
        '''
        cv2.setRNGSeed(self.random_state)
        _, labels, centroids = cv2.kmeans(data, b, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

        subtree = [{} for _ in range(b)]

        for centroid_index, centroid in enumerate(centroids):
            subtree[centroid_index]['centroid'] = centroid
            chosen_idxs = np.where(labels == centroid_index)[0]

            if len(chosen_idxs) <= b or self.max_depth == depth + 1:
                subtree[centroid_index]['is_leaf'] = True
                subtree[centroid_index]['vis_word_index'] = self.counter
                self.counter += 1
                print(f'Leaf {self.counter}/{np.power(b, self.max_depth)}', end='\r')
            else:
                subtree[centroid_index]['is_leaf'] = False
                selected_pts = data[chosen_idxs]
                subtree[centroid_index]['subtree'] = self.hi_kmeans(selected_pts, b, depth + 1)
                
        return subtree

    def tree_pass(self, query_desc: ndarray, tree: list, transform: bool=False) -> int:
        '''
        Passes down the tree using a certain descriptor vector and returns the key of the leaf.
        :param query_desc: vector containing a descriptor for an img
        :param tree: hierarchical tree you want to descend from the root till a leaf node
        :return: index of the visual word the descriptor belongs to
        '''
        if self.n_components != 0 and transform:
            query_desc = self.pca.transform(query_desc.reshape(1, -1)).flatten()

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
            vis_words[self.tree_pass(desc, self.tree, True)] += 1
        tf = vis_words / np.sum(vis_words)  # shape (n_vis_words, )
        return tf[:, None] # shape (n_vis_words, 1)

    def recall_rate(
        self, 
        obj_desc_list: Union[list, dict],
        topKbest: int = 1,
        perc_desc: float = 1.0,
        sim: str = 'l1',
    ) -> float:
        '''
        Returns the recall rate for the given parameters.
        :param obj_desc_list: list or dictionary containing all the descriptors for each object
        :param topKbest: number of best predictions to consider for the recall rate
        :param perc_desc: percentage of descriptors to consider for the query
        :param sim: the similarity function to use
        :return: recall rate
        '''
        iterator = enumerate(obj_desc_list) if isinstance(obj_desc_list, list) else obj_desc_list.items()
        recall_t = 0
        for object_index, object_descs in iterator:
            perc_index = int(perc_desc * len(object_descs))
            tf = self.get_query_tf_vector(object_descs[:perc_index])

            query_tfidf = tf * self.idf  # shape (n_vis_words, 1)
            server_tfidf = self.weights  # shape (n_vis_words, n_objects)
            
            if sim == 'l1':
                scores = HI.l_n(query_tfidf, server_tfidf, 1)
            elif sim == 'l2':
                scores = HI.l_n(query_tfidf, server_tfidf, 2)
            elif sim == 'cos':
                scores = HI.cosine(query_tfidf, server_tfidf)
            else:
                raise ValueError('Invalid similarity function')
            
            preds = np.argsort(scores)[:topKbest].tolist()

            if object_index in preds:
                recall_t += 1
        
        return recall_t / len(obj_desc_list)

    @staticmethod
    def l_n(x: ndarray, y: ndarray, n: int) -> ndarray:
        return np.linalg.norm((x / np.linalg.norm(x, ord=n, axis=0)) - (y / np.linalg.norm(y, ord=n, axis=0)), ord=n, axis=0)
    
    @staticmethod
    def cosine(x: ndarray, y: ndarray) -> ndarray:
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
                desc_vis_word_index = self.tree_pass(desc, self.tree, True)
                vis_words_count[desc_vis_word_index][obj_index] += 1
        
        tf = vis_words_count / np.sum(vis_words_count, axis=0, keepdims=True)  # shape (n_vis_words, n_objects)

        idf = np.log2(n_objects / (1e-3 + np.sum(np.array(vis_words_count, dtype=bool), axis=1, keepdims=True)))  # shape (n_vis_words, 1)
        
        weights = tf * idf  # shape (n_vis_words, n_objects)

        self.weights, self.idf = weights, idf

def tree_recall(
    server_obj_desc: List[List[ndarray]], 
    client_obj_desc: Union[List[List[ndarray]], Dict[int, List[ndarray]]], 
    b: int, 
    depth: int,
    perc_descr: List[float] = [1], 
    n_components: List[float] = [0], 
    sims: List[str] = ['l1'],
    random_state = 42
) -> None:
    HI_ob = HI(b, depth, random_state)
    for n_comp in n_components:
        HI_ob.build_tree(data=get_desc_list(server_obj_desc), n_components=n_comp)

        # Compute TFIDF weights
        HI_ob.get_server_TFIDF_weights(server_obj_desc)

        for perc in perc_descr:
            for sim in sims:
                # Querying and recall score
                top_1_recall = HI_ob.recall_rate(client_obj_desc, topKbest=1, perc_desc=perc, sim=sim)
                top_5_recall = HI_ob.recall_rate(client_obj_desc, topKbest=5, perc_desc=perc, sim=sim)

                print(f'Top-1 recall rate: {top_1_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {perc * 100}% of desc per query, number of components = {n_comp}, similarity = {sim}')
                print(f'Top-5 recall rate: {top_5_recall} using b = {b} and depth = {depth}, {len(client_obj_desc)} images, {np.power(b, depth)} visual words, {perc * 100}% of desc per query, number of components = {n_comp}, similarity = {sim}')



# %% Data loading and feature extraction

# Load client and server images
server_object_imgs, client_object_imgs, new_client_object_imgs = load_images(DATA_FOLDER, max_n=50)

# Notice that for the server images:
# - object 26 and 38 only have 2 images
# - object 37 has 4 images
# - all other objects have 3 images

# %% Create SIFT detector
# have to choose the parameters!

edgeT = 0.2
contrastT = 0.1
sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT)

# %% Extract descriptor for server images
server_obj_desc, n_server_desc, n_server_imgs = extract_desc(server_object_imgs, sift)
print(f'Number of server descriptors: {n_server_desc}, number of server images: {n_server_imgs}')

# %% Extract descriptor for client images
client_obj_desc, n_client_desc, n_client_imgs = extract_desc(client_object_imgs, sift)
print(f'Number of client descriptors: {n_client_desc}, number of client images: {n_client_imgs}')

# %% Extract descriptor for new client images
new_client_obj_desc, n_new_client_desc, n_new_client_imgs = extract_desc(new_client_object_imgs, sift)
print(f'Number of new client descriptors: {n_new_client_desc}, number of new client images: {n_new_client_imgs}')

# %% Print average number of features
print('The average number of features is:'
    +f'\n server images: {int(n_server_desc / n_server_imgs)}'
    +f'\n client images: {int(n_client_desc / n_client_imgs)}'
)

# %% Save descriptors as pickle files
with open(f'{OUT_FOLDER}/server_desc.pkl', 'wb') as f:
    pickle.dump(server_obj_desc, f)

with open(f'{OUT_FOLDER}/client_desc.pkl', 'wb') as f:
    pickle.dump(client_obj_desc, f)

with open(f'{OUT_FOLDER}/new_client_desc.pkl', 'wb') as f:
    pickle.dump(new_client_obj_desc, f)

# %% Load descriptors from pickle files
with open(f'{OUT_FOLDER}/server_desc.pkl', 'rb') as f:
    server_obj_desc = pickle.load(f)

with open(f'{OUT_FOLDER}/client_desc.pkl', 'rb') as f:
    client_obj_desc = pickle.load(f)

with open(f'{OUT_FOLDER}/new_client_desc.pkl', 'rb') as f:
    new_client_obj_desc = pickle.load(f)

# %% Building the Vocabulary Tree using b = 4 and depth = 3
b = 4
depth = 3
tree_recall(server_obj_desc, client_obj_desc, b, depth, random_state=RANDOM_STATE)

# %% Building the Vocabulary Tree using b=4 and depth=5
b = 4
depth = 5
tree_recall(server_obj_desc, client_obj_desc, b, depth, random_state=RANDOM_STATE)

# %% Building the Vocabulary Tree using b=5 and depth=7
b = 5
depth = 7
perc_descr = [1.0, 0.9, 0.7, 0.5]

tree_recall(server_obj_desc, client_obj_desc, b, depth, perc_descr, random_state=RANDOM_STATE)

# %% Building the Vocabulary Tree using b = 4 and depth = 3 using PCA
b = 4
depth = 3
n_components = [0.8]
tree_recall(server_obj_desc, client_obj_desc, b, depth, n_components=n_components, random_state=RANDOM_STATE)

# %% Building the Vocabulary Tree using b=4 and depth=5 using PCA
b = 4
depth = 5
n_components = [0.8]
tree_recall(server_obj_desc, client_obj_desc, b, depth, n_components=n_components, random_state=RANDOM_STATE)
# %% Building the Vocabulary Tree using b=5 and depth=7 using PCA
b = 5
depth = 7
n_components = [0.8]
tree_recall(server_obj_desc, client_obj_desc, b, depth, n_components=n_components, random_state=RANDOM_STATE)
# %%
