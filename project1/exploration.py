# %%
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import pickle

# For typing
from numpy import ndarray
from typing import Union
detector_type = Union[cv.xfeatures2d_SIFT, cv.xfeatures2d_SURF]

IMG_FOLDER = 'data1'
OUT_FOLDER = 'output1'
PLAY_FOLDER = 'playground1'


# %% Functions
def save_img(img: ndarray, name: str, kp=None):
    'Saves the image to disk, if given adds keypoints'
    if kp is not None:
        colors=[(0,255,0), (255,0,0)]
        for idx, k in enumerate(kp):
            img = cv.drawKeypoints(img, keypoints=k, outImage=None, color=colors[idx])
    # save image using cv
    cv.imwrite(f'{PLAY_FOLDER}/{name}.jpg', img)


def load_imgs(img_folder: str) -> dict:
    '''Loads all images in the given folder and returns them in a dictionary.'''
    imgs = {}
    for img in os.listdir(img_folder):
        img_location = f'{img_folder}/{img}'
        imgs[img] = cv.imread(img_location)
    return imgs


def rotate_points(
    points: list, 
    dim: tuple, 
    theta: float=0, 
    rot_center: tuple=(0, 0)
) -> list:
    '''
    Function taken online to rotate points around a center.
    :param points: List of Keypoint Objects
    :param theta: rotation angle in deg
    :param rot_center: tuple with rotation center
    :return: List of rotated points
    '''
    points_copy = np.copy(points)

    points_coords = np.array([point.pt for point in points_copy]).T
    
    height, width = dim
    rot_mtx = cv.getRotationMatrix2D(rot_center, theta, 1)
    abs_cos = abs(rot_mtx[0,0]) 
    abs_sin = abs(rot_mtx[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rot_mtx[0, 2] += bound_w/2 - rot_center[0]
    rot_mtx[1, 2] += bound_h/2 - rot_center[1]
    
    rot_points_coords = np.matmul(rot_mtx, np.vstack((points_coords, np.ones(points_coords.shape[1]))))

    for point, (x, y) in zip(points_copy, rot_points_coords.T):
        point.pt = (x, y)

    return points_copy


def rotate_image(
    img: ndarray, 
    theta: float=0, 
    rot_center=(0, 0)
) -> ndarray:
    '''
    Rotates image around a rotation center.
    :param img: image to rotate
    :param theta: rotation angle in deg
    :param rot_center: tuple with rotation center
    :return: Image rotated of a specific angle.
    '''
    height, width = img.shape[:-1]
    rot_mtx = cv.getRotationMatrix2D(rot_center, theta, 1)
    abs_cos = abs(rot_mtx[0,0]) 
    abs_sin = abs(rot_mtx[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rot_mtx[0, 2] += bound_w/2 - rot_center[0]
    rot_mtx[1, 2] += bound_h/2 - rot_center[1]

    transf_img = cv.warpAffine(src=img, M=rot_mtx, dsize=(bound_w, bound_h))
    return transf_img


def scale_points(points: list, scale_factor: float) -> list:
    points_copy = np.copy(points)
    for idx in range(len(points)):
        point = points_copy[idx]
        scaled = tuple(scale_factor * np.array(point.pt))
        points_copy[idx].pt = (int(scaled[0]), int(scaled[1]))
    return points_copy


def scale_image(img: ndarray, scale_factor: float) -> ndarray:
    new_dims = tuple(scale_factor * np.array(img.shape[:-1]))
    resized = cv.resize(img, dsize=(int(new_dims[1]), int(new_dims[0])))
    return resized

def single_repeatability(p0, p1) -> bool:
    '''
    Check if the 2 points are in the vicinities. For the repeatability factor.
    :param p0: Point detected in the modified image
    :param p1: Predicted point in the modified image, given point in original image.
    :return: bool if the points are close (see 2.1 Introduction)
    '''
    x0, y0 = p0.pt
    x1, y1 = p1.pt
    x_diff = np.abs(x0 - x1)
    y_diff = np.abs(y0 - y1)
    return (x_diff <= 2) and (y_diff <= 2)


def repeatability_score(original_kp, new_kp) -> float:
    '''
    Returns the repeatability measure defined in the course text.
    :param original_kp: list of kp detected in the original image.Transformation must have already been applied.
    :param new_kp: list of points detected in the transformed image.
    :return: repeatability score (defined in project material).
    '''
    den = len(original_kp)
    num = 0
    for orig_pt in original_kp:
        for new_pt in new_kp:
            if single_repeatability(new_pt, orig_pt):
                num += 1
                break
    return num / den


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


def test_rotation(img: ndarray, theta: float, detector: detector_type, save_imgs: bool=False) -> float:
    '''
    Test the rotation repeatability for a certain angle rotation.
    :param img: image to be used for the test.
    :param theta: rotation angle in degrees.
    :param detector: detector to be used (SIFT or SURF).
    :param save_imgs: bool to save the images or not.
    :return: repeatability score for the rotation.
    '''
    orig_kp = detect_keypoints(img, detector=detector)

    rot_center = tuple(reversed(np.floor(np.array(img.shape[:-1]) / 2)))
    rotated_points = rotate_points(orig_kp, dim=img.shape[:-1], theta=theta, rot_center=rot_center)

    rotated_img = rotate_image(img, theta=theta, rot_center=rot_center)

    new_points = detect_keypoints(rotated_img, detector=detector)

    rep_score = repeatability_score(rotated_points, new_points)

    if save_imgs:
        save_img(rotated_img, f'points_on_original_img_theta_{theta}_rep_score_{rep_score}', kp=[rotated_points])
        save_img(rotated_img, f'new_points_rotated_img_theta_{theta}_rep_score_{rep_score}', kp=[new_points])

    return rep_score


def test_scale(img: ndarray, scale_f: float, detector: detector_type, save_imgs: bool=False) -> float:
    '''
    Test the scale repeatability for a certain scale factor.
    :param img: image to be used for the test.
    :param scale_f: scale factor.
    :param detector: detector to be used (SIFT or SURF).
    :param save_imgs: bool to save the images or not.
    :return: repeatability score for the scale.
    '''
    orig_kp = detect_keypoints(img, detector=detector)

    scaled_points = scale_points(orig_kp, scale_f)

    scaled_img = scale_image(img, scale_f)

    new_points = detect_keypoints(scaled_img, detector=detector)

    rep_score = repeatability_score(scaled_points, new_points)

    if save_imgs:
        save_img(scaled_img, f'points_on_original_img_scale_{scale_f}_rep_score_{rep_score}', kp=[scaled_points])
        save_img(scaled_img, f'new_points_scaled_img_scale_{scale_f}_rep_score_{rep_score}', kp=[new_points])

    return rep_score


def test_illumination(img: ndarray, illumination_f: float, detector: detector_type, save_imgs: bool=False):
    '''
    Test the illumination repeatability for a certain scale factor.
    :param img: image to be used for the test.
    :param illumination_factor: illumination factor that scales the image.
    :param detector: detector to be used (SIFT or SURF).
    :param save_imgs: bool to save the images or not.
    :return: repeatability score for the scale.
    '''
    orig_kp = detect_keypoints(img, detector=detector)

    dimmed_image = np.clip(img * illumination_f, 0, 255).astype(np.uint8)

    new_points = detect_keypoints(dimmed_image, detector=detector)

    rep_score = repeatability_score(orig_kp, new_points)

    if save_imgs:
        save_img(img, f'points_on_original_img_illum_{illumination_f}_rep_score_{rep_score}', kp=[orig_kp])
        save_img(dimmed_image, f'new_points_illum_img_illum_{illumination_f}_rep_score_{rep_score}', kp=[new_points])
    
    return rep_score


# %% File loading and detector initialization
# Load images
imgs = load_imgs(IMG_FOLDER)
obj1_5, obj1_t1 = imgs['obj1_5.JPG'], imgs['obj1_t1.JPG']


# Create sift detector
edgeT = 10
contrastT = 0.165
sift = cv.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT)

# Create surf detector
featureT = 5000
surf = cv.xfeatures2d.SURF_create(featureT)

# Create matcher 
bf = cv.BFMatcher()

# Set the parameters for the tests
rotations = np.arange(0, 361, 15)
m = 1.2
scale_factors = [np.power(m, exp) for exp in np.arange(0, 9)]
illumination_factors = np.arange(0.5, 2, 0.1)

# %% Test rotation repeatability for SIFT
rotation_scores_sift = []
for theta in rotations:
    score = test_rotation(obj1_5, theta, detector=sift)
    rotation_scores_sift.append(score)
    print(f'Rotation: {theta}, score: {score}')
# %% Plot rotation repeatability for SIFT
plot(
    rotations, 
    rotation_scores_sift,
    'Rotation angle',
    'Repeatability score',
    'Repeatability score for different rotation angles using the SIFT algorithm',
    (0, 360),
    f"{OUT_FOLDER}/rotation_graph_sift.png"
)
# %% Test scale repeatability for SIFT
scale_scores_sift = []
for scale in scale_factors:
    score = test_scale(obj1_5, scale_f=scale, detector=sift)
    scale_scores_sift.append(score)
    print(f'Scale: {scale}, score: {score}')
# %% Plot scale repeatability for SIFT
plot(
    scale_factors, 
    scale_scores_sift,
    'Scale factor',
    'Repeatability score',
    'Repeatability score for different scale factors using the SIFT algorithm',
    filename=f"{OUT_FOLDER}/scale_graph_sift.png"
)
# %% Test illumination repeatability for SIFT
illumination_scores_sift = []
for illumination_factor in illumination_factors:
    score = test_illumination(obj1_5, illumination_factor, detector=sift)
    illumination_scores_sift.append(score)
    print(f'Illumination factor: {illumination_factor}, score: {score}')
# %% Plot illumination repeatability for SIFT
plot(
    illumination_factors, 
    illumination_scores_sift,
    'Illumination factor',
    'Repeatability score',
    'Repeatability score for different illumination factors using the SIFT algorithm',
    filename=f"{OUT_FOLDER}/illumination_graph_sift.png"
)
# %% Test rotation repeatability for SURF
rotation_scores_surf = []
for theta in rotations:
    score = test_rotation(obj1_5, theta, detector=surf)
    rotation_scores_surf.append(score)
    print(f'Rotation: {theta}, score: {score}')
# %% Plot rotation repeatability for SURF
plot(
    rotations, 
    rotation_scores_surf,
    'Rotation angle',
    'Repeatability score',
    'Repeatability score for different rotation angles using the SURF algorithm',
    (0, 360),
    f"{OUT_FOLDER}/rotation_graph_surf.png"
)
# %% Test scale repeatability for SURF
scale_scores_surf = []
for scale in scale_factors:
    score = test_scale(obj1_5, scale_f=scale, detector=surf)
    scale_scores_surf.append(score)
    print(f'Scale: {scale}, score: {score}')
# %% Plot scale repeatability for SURF
plot(
    scale_factors, 
    scale_scores_surf,
    'Scale factor',
    'Repeatability score',
    'Repeatability score for different scale factors using the SURF algorithm',
    filename=f"{OUT_FOLDER}/scale_graph_surf.png"
)
# %% Test illumination repeatability for SURF
illumination_scores_surf = []
for illumination_factor in illumination_factors:
    score = test_illumination(obj1_5, illumination_factor, detector=surf)
    illumination_scores_surf.append(score)
    print(f'Illumination factor: {illumination_factor}, score: {score}')
# %% Plot illumination repeatability for SURF
plot(
    illumination_factors, 
    illumination_scores_surf,
    'Illumination factor',
    'Repeatability score',
    'Repeatability score for different illumination factors using the SURF algorithm',
    filename=f"{OUT_FOLDER}/illumination_graph_surf.png"
)
# %%
with open(f'{OUT_FOLDER}/scale_scores_sift.pkl','wb') as f:
    pickle.dump(scale_scores_sift, f)
# %%
with open(f'{OUT_FOLDER}/rotation_scores_sift.pkl','wb') as f:
    pickle.dump(rotation_scores_sift, f)
# %%
with open(f'{OUT_FOLDER}/scale_scores_surf.pkl','wb') as f:
    pickle.dump(scale_scores_surf, f)
# %%
with open(f'{OUT_FOLDER}/rotation_scores_surf.pkl','wb') as f:
    pickle.dump(rotation_scores_surf, f)
# %%
with open(f'{OUT_FOLDER}/illumination_scores_sift.pkl','wb') as f:
    pickle.dump(illumination_scores_sift, f)
# %%
with open(f'{OUT_FOLDER}/illumination_scores_surf.pkl','wb') as f:
    pickle.dump(illumination_scores_surf, f)

# %% Function to draw matches
def draw_matches(img1, kp1, img2, kp2, matches, title='', filename=None):
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    matches_image = cv.drawMatchesKnn(
        img1, kp1, img2, kp2, matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        #matchColor=(0, 255, 0)
    )
    if filename:
        plt.imshow(matches_image)
        plt.title(title, {'fontsize': 7})
        plt.axis('off')
        plt.savefig(
            f'{PLAY_FOLDER}/{filename}.png', 
            dpi=400,
            bbox_inches='tight'
        )
        plt.close()
    else:
        return matches_image


# %% Getting the keypoints and descriptors using SIFT
sift_kp_obj1_5, sift_desc_obj1_5 = sift.detectAndCompute(obj1_5, None)
sift_kp_obj1_t1, sift_desc_obj1_t1 = sift.detectAndCompute(obj1_t1, None)
sift_matches = bf.knnMatch(
    sift_desc_obj1_5,
    sift_desc_obj1_t1,
    k=2
)

# %% Fixed Threshold matching
for FT_threshold in [160, 200]: # np.arange(150, 201, 10)
    FT_matches = [[d] for d ,_ in sift_matches if d.distance < FT_threshold]
    draw_matches(
        obj1_5,
        sift_kp_obj1_5,
        obj1_t1,
        sift_kp_obj1_t1,
        FT_matches,
        f'Feature matching of SIFT descriptors using a Fixed Threshold of {FT_threshold}',
        f'SIFT_FT_matches_T{FT_threshold}'
    )
    print(f'FT threshold: {FT_threshold}, number of matches: {len(FT_matches)}')

# %% Nearest Neighbor matching
NN_matches = [[d] for d ,_ in sift_matches]
draw_matches(
    obj1_5,
    sift_kp_obj1_5,
    obj1_t1,
    sift_kp_obj1_t1,
    NN_matches,
    'Feature matching of SIFT descriptors using Nearest Neighbor',
    'SIFT_NN_matches'
)

# %% Nearest Neighbor Distance Ratio matching
for NNDR_threshold in [0.7, 0.8, 0.9]:
    NNDR_matches = [[d1] for d1, d2 in sift_matches if d1.distance / d2.distance < NNDR_threshold]
    draw_matches(
        obj1_5,
        sift_kp_obj1_5,
        obj1_t1,
        sift_kp_obj1_t1,
        NNDR_matches,
        f'Feature matching of SIFT descriptors using Nearest Neighbor Distance Ratio with a threshold of {NNDR_threshold}',
        f'SIFT_NNDR_matches_T{NNDR_threshold}'
    )

# %% Getting the keypoints and descriptors using SURF
surf_kp_obj1_5, surf_desc_obj1_5 = surf.detectAndCompute(obj1_5, None)
surf_kp_obj1_t1, surf_desc_obj1_t1 = surf.detectAndCompute(obj1_t1, None)
surf_matches = bf.knnMatch(
    surf_desc_obj1_5,
    surf_desc_obj1_t1,
    k=2
)

# %% Nearest Neighbor Distance Ratio matching using the SURF matchese
for NNDR_threshold in [0.7, 0.8, 0.9]:
    NNDR_matches = [[d1] for d1, d2 in surf_matches if d1.distance / d2.distance < NNDR_threshold]
    draw_matches(
        obj1_5,
        sift_kp_obj1_5,
        obj1_t1,
        sift_kp_obj1_t1,
        NNDR_matches,
        f'Feature matching of SURF descriptors using Nearest Neighbor Distance Ratio with a threshold of {NNDR_threshold}',
        f'SURF_NNDR_matches_T{NNDR_threshold}'
    )
# %%
