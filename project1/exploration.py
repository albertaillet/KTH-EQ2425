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

# not good, when plotting the points on top of the rescaled image they don't match the location
def scale_points(points, scale_factor):
    points_copy = np.copy(points)
    for idx in range(len(points)):
        point = points_copy[idx]
        scaled = tuple(scale_factor * np.array(point.pt))
        points_copy[idx].pt = (int(scaled[0]), int(scaled[1]))
    return points_copy


def scale_img(img, scale_factor):
    new_dims = tuple(scale_factor * np.array(img.shape[:-1]))
    resized = cv.resize(img, dsize=(int(new_dims[1]), int(new_dims[0])))
    return resized

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

# don't like the name but I'll keep it for now
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

def remove_outside(points, img_dims):
    '''
    Keeps all the points in the original image that can still be in the transformed one.
    Since after a rotation some points might disappear from the picture.
    :param points: List of all the points found in the original picture and transformed.
    :param img_dims: dimensions of the transformed picture.
    :return: list of points that are still inside the transformed picture.
    '''
    points = [point for point in points if 0 <= point.pt[0] <= img_dims[0] and 0 <= point.pt[1] <= img_dims[1]]
    return points

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

def detect_keypoints(img, detector, sift=True):
    if sift:
        kp, _ = detector.detectAndCompute(img, None)
    else:
        kp, _ = detector.detectAndCompute(img, None)
    return kp 

def test_rotation(img, theta, detector, sift=True):
    orig_kp = detect_keypoints(img, detector=detector, sift=sift)

    rot_center = tuple(reversed(np.floor(np.array(img.shape[:-1]) / 2)))
    rotated_points = rotate_points(orig_kp, dim=img.shape[:-1], theta=theta, rot_center=rot_center)

    rotated_img = rotate_image(img, theta=theta, rot_center=rot_center)

    new_points = detect_keypoints(rotated_img, detector=detector, sift=sift)

    rep_score = repeatability_score(rotated_points, new_points)

    save_img(rotated_img, f'points_on_original_img_theta_{theta}_rep_score_{rep_score}', kp=[rotated_points])
    save_img(rotated_img, f'new_points_rotated_img_theta_{theta}_rep_score_{rep_score}', kp=[new_points])
    
    return rep_score

def test_scale(img, scale_f, detector, sift=True):
    orig_kp = detect_keypoints(img, detector=detector, sift=sift)

    scaled_points = scale_points(orig_kp, scale_f)

    scaled_img = scale_img(img, scale_f)

    new_points = detect_keypoints(scaled_img, detector=detector, sift=sift)

    rep_score = repeatability_score(scaled_points, new_points)

    save_img(scaled_img, f'points_on_original_img_scale_{scale_f}_rep_score_{rep_score}', kp=[scaled_points])
    save_img(scaled_img, f'new_points_rotated_img_scale_{scale_f}_rep_score_{rep_score}', kp=[new_points])

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

# %%
rotation_scores_sift = []
for theta in rotations:
    score = test_rotation(img, theta, surf, True)
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
    score = test_scale(img, scale_f=scale, detector=sift, sift=True)
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

with open(f'{OUT_FOLDER}/scale_scores_sift.pkl','wb') as f:
    pickle.dump(scale_scores_sift, f)

with open(f'{OUT_FOLDER}/rotation_scores_sift.pkl','wb') as f:
    pickle.dump(rotation_scores_sift, f)



# %%
rotation_scores_surf = []
for theta in rotations:
    score = test_rotation(img, theta, surf, False)
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
    score = test_scale(img, scale_f=scale, detector=surf, sift=False)
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

with open(f'{OUT_FOLDER}/scale_scores_surf.pkl','wb') as f:
    pickle.dump(scale_scores_surf, f)

with open(f'{OUT_FOLDER}/rotation_scores_surf.pkl','wb') as f:
    pickle.dump(rotation_scores_surf, f)

# %%

save_img(img, 'trial', [surf_kp, sift_kp])