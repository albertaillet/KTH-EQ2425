# %%
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import pickle

# For typing
from numpy import ndarray

IMG_FOLDER = 'data1'
OUT_FOLDER = 'output1'
PLAY_FOLDER = 'playground1'


# %%
# functions
def save_img(img: ndarray, name: str, kp=None):
    'Saves the image to disk, if given adds keypoints'
    if kp is not None:
        colors=[(0,255,0), (255,0,0)]
        for idx, k in enumerate(kp):
            img = cv.drawKeypoints(img, keypoints=k, outImage=None, color=colors[idx])
    # save image using cv
    cv.imwrite(f'{PLAY_FOLDER}/{name}.jpg', img)


def load_imgs(img_folder):
    imgs = []
    for img in os.listdir(img_folder):
        img_location = f'{img_folder}/{img}'
        imgs.append(cv.imread(img_location))
    return imgs


def rotate_points(points, dim, theta=0, rot_center=(0, 0)):
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
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mtx[0,0]) 
    abs_sin = abs(rot_mtx[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rot_mtx[0, 2] += bound_w/2 - rot_center[0]
    rot_mtx[1, 2] += bound_h/2 - rot_center[1]
    
    rot_points_coords = np.matmul(rot_mtx, np.vstack((points_coords, np.ones(points_coords.shape[1]))))

    for point, (x, y) in zip(points_copy, rot_points_coords.T):
        point.pt = (x, y)

    return points_copy


def rotate_image(img, theta=0, rot_center=(0, 0)):
    '''
    Rotates image around a rotation center.
    :param img: image to rotate
    :param theta: rotation angle in deg
    :param rot_center: tuple with rotation center
    :return: Image rotated of a specific angle.
    '''
    height, width = img.shape[:-1]
    rot_mtx = cv.getRotationMatrix2D(rot_center, theta, 1)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rot_mtx[0,0]) 
    abs_sin = abs(rot_mtx[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rot_mtx[0, 2] += bound_w/2 - rot_center[0]
    rot_mtx[1, 2] += bound_h/2 - rot_center[1]

    transf_img = cv.warpAffine(src=img, M=rot_mtx, dsize=(bound_w, bound_h))
    return transf_img


def scale_points(points, scale_factor):
    points_copy = np.copy(points)
    for idx in range(len(points)):
        point = points_copy[idx]
        scaled = tuple(scale_factor * np.array(point.pt))
        points_copy[idx].pt = (int(scaled[0]), int(scaled[1]))
    return points_copy


def scale_image(img, scale_factor):
    new_dims = tuple(scale_factor * np.array(img.shape[:-1]))
    resized = cv.resize(img, dsize=(int(new_dims[1]), int(new_dims[0])))
    return resized

def single_repeatability(p0, p1):
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


def repeatability_score(original_kp, new_kp):
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

def compute_distances(original_kp, new_kp):
    n1 = len(original_kp)
    n2 = len(new_kp)
    dist_mtx = np.zeros(shape=(n1, n2))
    for idx_orig in range(len(original_kp)):
        for idx_new in range(len(new_kp)):
            idx_orig

def detect_keypoints(img, detector):
    kp, _ = detector.detectAndCompute(img, None)
    return kp 


def test_rotation(img, theta, detector):
    orig_kp = detect_keypoints(img, detector=detector)

    rot_center = tuple(reversed(np.floor(np.array(img.shape[:-1]) / 2)))
    rotated_points = rotate_points(orig_kp, dim=img.shape[:-1], theta=theta, rot_center=rot_center)

    rotated_img = rotate_image(img, theta=theta, rot_center=rot_center)

    new_points = detect_keypoints(rotated_img, detector=detector)

    rep_score = repeatability_score(rotated_points, new_points)

    save_img(rotated_img, f'points_on_original_img_theta_{theta}_rep_score_{rep_score}', kp=[rotated_points])
    save_img(rotated_img, f'new_points_rotated_img_theta_{theta}_rep_score_{rep_score}', kp=[new_points])
    
    return rep_score


def test_scale(img, scale_f, detector):
    orig_kp = detect_keypoints(img, detector=detector)

    scaled_points = scale_points(orig_kp, scale_f)

    scaled_img = scale_image(img, scale_f)

    new_points = detect_keypoints(scaled_img, detector=detector)

    rep_score = repeatability_score(scaled_points, new_points)

    save_img(scaled_img, f'points_on_original_img_scale_{scale_f}_rep_score_{rep_score}', kp=[scaled_points])
    save_img(scaled_img, f'new_points_scaled_img_scale_{scale_f}_rep_score_{rep_score}', kp=[new_points])

    return rep_score


def test_illumination(img, illumination_factor, detector):
    orig_kp = detect_keypoints(img, detector=detector)

    dimmed_image = np.clip(img * illumination_factor, 0, 255).astype(np.uint8)

    new_points = detect_keypoints(dimmed_image, detector=detector)

    rep_score = repeatability_score(orig_kp, new_points)

    save_img(img, f'points_on_original_img_illum_{illumination_factor}_rep_score_{rep_score}', kp=[orig_kp])
    save_img(dimmed_image, f'new_points_illum_img_illum_{illumination_factor}_rep_score_{rep_score}', kp=[new_points])
    
    return rep_score


# %%
imgs = load_imgs(IMG_FOLDER)
img = imgs[0]

edgeT = 10
contrastT = 0.165

featureT = 5000

# create sift detector
sift = cv.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT )
sift_kp = sift.detect(img)

# create surf detector
surf = cv.xfeatures2d.SURF_create(featureT)
surf_kp, _ = surf.detectAndCompute(img, None)

rotations = np.arange(0, 361, 15)
m = 1.2
scale_factors = [np.power(m, exp) for exp in np.arange(0, 9)]
illumination_factors = np.arange(0.5, 1.5, 0.1)

# %%
rotation_scores_sift = []
for theta in rotations:
    score = test_rotation(img, theta, detector=sift)
    rotation_scores_sift.append(score)
    print(f'Rotation: {theta}, score: {score}')
# %%
plot(
    rotations, 
    rotation_scores_sift,
    'Rotation angle',
    'Repeatability score',
    'Repeatability score for different rotation angles using the SIFT algorithm',
    (0, 360),
    f"{OUT_FOLDER}/rotation_graph_sift.png"
)
# %%
scale_scores_sift = []
for scale in scale_factors:
    score = test_scale(img, scale_f=scale, detector=sift)
    scale_scores_sift.append(score)
    print(f'Scale: {scale}, score: {score}')
# %%
plot(
    scale_factors, 
    scale_scores_sift,
    'Scale factor',
    'Repeatability score',
    'Repeatability score for different scale factors using the SIFT algorithm',
    filename=f"{OUT_FOLDER}/scale_graph_sift.png"
)
# %%
rotation_scores_surf = []
for theta in rotations:
    score = test_rotation(img, theta, detector=surf)
    rotation_scores_surf.append(score)
    print(f'Rotation: {theta}, score: {score}')
# %%
plot(
    rotations, 
    rotation_scores_surf,
    'Rotation angle',
    'Repeatability score',
    'Repeatability score for different rotation angles using the SURF algorithm',
    (0, 360),
    f"{OUT_FOLDER}/rotation_graph_surf.png"
)
# %%
scale_scores_surf = []
for scale in scale_factors:
    score = test_scale(img, scale_f=scale, detector=surf)
    scale_scores_surf.append(score)
    print(f'Scale: {scale}, score: {score}')
# %%
plot(
    scale_factors, 
    scale_scores_surf,
    'Scale factor',
    'Repeatability score',
    'Repeatability score for different scale factors using the SURF algorithm',
    filename=f"{OUT_FOLDER}/scale_graph_surf.png"
)
# %%
illumination_scores_sift = []
for illumination_factor in illumination_factors:
    score = test_illumination(img, illumination_factor, detector=sift)
    illumination_scores_sift.append(score)
    print(f'Illumination factor: {illumination_factor}, score: {score}')
# %%
plot(
    illumination_factors, 
    illumination_scores_sift,
    'Illumination factor',
    'Repeatability score',
    'Repeatability score for different illumination factors using the SIFT algorithm',
    filename=f"{OUT_FOLDER}/illumination_graph_sift.png"
)
# %%
illumination_scores_surf = []
for illumination_factor in illumination_factors:
    score = test_illumination(img, illumination_factor, detector=surf)
    illumination_scores_surf.append(score)
    print(f'Illumination factor: {illumination_factor}, score: {score}')
# %%
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
sift_kp_obj1_5, sift_desc_obj1_5 = sift.detectAndCompute(imgs[0], None)
sift_kp_obj1_t5, sift_desc_obj1_t5 = sift.detectAndCompute(imgs[1], None)

# %% Fixed Threshold matching
bf = cv.BFMatcher()
matches = bf.knnMatch(
    sift_desc_obj1_5,
    sift_desc_obj1_t5,
    k=1
)
FT_threshold = 300
FT_matches = []
for (d,) in matches:
    if d.distance < FT_threshold:
        FT_matches.append([d])

#%%
# in_bet = [x for x in np.arange(150, 201, 10)]
for FT_threshold in [160, 200]:
    FT_matches = []
    for (d,) in matches:
        if d.distance < FT_threshold:
            FT_matches.append([d])
    FT_matches_image = cv.drawMatchesKnn(
        imgs[0],
        sift_kp_obj1_5,
        imgs[1],
        sift_kp_obj1_t5,
        FT_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # plt.figure(figsize=(30,20))
    plt.imshow(FT_matches_image)
    plt.title(f'Feature matching using a Fixed Threshold of {FT_threshold}')
    plt.axis('off')
    plt.savefig(f'{PLAY_FOLDER}/FT_matches_T{FT_threshold}.png', dpi=400,bbox_inches='tight')
    plt.close()
    print(f'FT threshold: {FT_threshold}, number of matches: {len(FT_matches)}')

# %% NN matching
bf = cv.BFMatcher()
NN_matches = bf.knnMatch(
    sift_desc_obj1_5,
    sift_desc_obj1_t5,
    k=1
)
# %%
NN_matches_image = cv.drawMatchesKnn(
    imgs[0],
    sift_kp_obj1_5,
    imgs[1],
    sift_kp_obj1_t5,
    NN_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(45,20))
plt.imshow(NN_matches_image)
plt.axis('off')
plt.title(f'Feature matching using Nearest Neighbor')
plt.savefig(f'{PLAY_FOLDER}/NN_matches.png', dpi=400,bbox_inches='tight')
plt.close()


# %% NNDR matching
bf = cv.BFMatcher()
matches = bf.knnMatch(
    sift_desc_obj1_5,
    sift_desc_obj1_t5,
    k=2
)
for NNDR_threshold in [0.7]:
    NNDR_matches = []
    for d1, d2 in matches:
        if d1.distance / d2.distance < NNDR_threshold:
            NNDR_matches.append([d1])

    NNDR_matches_image = cv.drawMatchesKnn(
        imgs[0],
        sift_kp_obj1_5,
        imgs[1],
        sift_kp_obj1_t5,
        NNDR_matches,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(45,20))
    plt.imshow(NNDR_matches_image)
    plt.axis('off')
    plt.title(f'Feature matching using Nearest Neighbor Distance Ratio of {NNDR_threshold}')
    plt.savefig(f'{PLAY_FOLDER}/NNDR_matches_T{NNDR_threshold}.png', dpi=400,bbox_inches='tight')
    plt.close()

# %%
