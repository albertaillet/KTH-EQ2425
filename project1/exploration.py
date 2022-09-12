# %%
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

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
        img = cv.drawKeypoints(img, keypoints=kp, outImage=None, color=(0,255,0))
    # save image using cv
    cv.imwrite(f'{PLAY_FOLDER}/{name}.jpg', img)



def load_imgs(img_folder):
    imgs = []
    for img in os.listdir(img_folder):
        img_location = f'{img_folder}/{img}'
        imgs.append(cv.imread(img_location))
    return imgs


def rotate_points(points, theta=0, rot_center=(0, 0)):
    '''
    Function taken online to rotate points around a center.
    :param points: List of Keypoint Objects
    :param theta: rotation angle in deg
    :param rot_center: tuple with rotation center
    :return: List of rotated points
    '''
    points_copy = np.copy(points)
    for idx in range(len(points)):
        x, y = points_copy[idx].pt
        ox, oy = rot_center

        qx = ox + np.cos(np.deg2rad(theta)) * (x - ox) + np.sin(np.deg2rad(theta)) * (y - oy)
        qy = oy + -np.sin(np.deg2rad(theta)) * (x - ox) + np.cos(np.deg2rad(theta)) * (y - oy)

        points_copy[idx].pt = (qx, qy)
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
    rot_mtx = cv.getRotationMatrix2D(rot_center, theta, 1)
    transf_img = cv.warpAffine(src=img, M=rot_mtx, dsize=img.shape[:-1])
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

def plot(x, y, x_label, y_label, title):
    fig = plt.figure(figsize=(15, 15))
    plt.plot(x, y)
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.title = title
    plt.show()

def compute_distances(original_kp, new_kp):
    n1 = len(original_kp)
    n2 = len(new_kp)
    dist_mtx = np.zeros(shape=(n1, n2))
    for idx_orig in range(len(original_kp)):
        for idx_new in range(len(new_kp)):
            idx_orig


def SIFT(img):
    pass


def SURF(img):
    pass


def main():
    imgs = load_imgs(IMG_FOLDER)
    img = imgs[0]

    edgeT = 10
    contrastT = 0.165

    featureT = 5000

    # create sift detector
    sift = cv.xfeatures2d.SIFT_create(edgeThreshold=edgeT, contrastThreshold=contrastT )
    sift_kp = sift.detect(img)

    # create surf detector
    # surf = cv.xfeatures2d.SURF_create(featureT)
    # surf_kp, _ = surf.detectAndCompute(img, None)

    n_pts = len(sift_kp)
    save_img(img, f'sift_{n_pts}_{edgeT}_{contrastT}', kp=sift_kp)
    # save_img(img, f'surf_{n_pts}_{featureT}', kp=surf_kp)




if __name__ == '__main__':
    main()

# %%