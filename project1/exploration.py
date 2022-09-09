# %%
import numpy as np
import cv2 as cv
import os

# %%
img1 = cv.imread('data1/obj1_5.jpg')
img2 = cv.imread('data1/obj1_t1.jpg')
img_folder = 'data1'


# %%
# functions
def show_img(img, name='', kp=None):
    'Plots the img, if given adds keypoints'
    img_kp = cv.drawKeypoints(img, keypoints=kp, outImage=None)
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img_kp)
    cv.waitKey(0)



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
    resized = cv.resize(img, dsize=(int(new_dims[0]), int(new_dims[1])), interpolation=cv.INTER_LINEAR)
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
def repeatability(p0, p1):
    '''
    Check if the 2 points are in the vicinities. For the repeatability factor
    :param p0: Point detected in the modified image
    :param p1: Predicted point in the modified image, given point in original imag
    :return: bool if the points are close (see 2.1 Introduction)
    '''
    x0, y0 = p0
    x1, y1 = p1
    x_diff = np.abs(x0 - x1)
    y_diff = np.abs(y0 - y1)
    return (x_diff <= 2) and (y_diff <= 2)


def SIFT(img):
    pass


def SURF(img):
    pass


def main():
    imgs = load_imgs(img_folder)
    img = imgs[0]

    # create sift detector
    sift = cv.xfeatures2d.SIFT_create()
    sift_kp = sift.detect(img)

    # create surf detector
    surf = cv.xfeatures2d.SURF_create(400)
    surf_kp, surf_descr = surf.detectAndCompute(img, None)

    theta = 45
    rot_center = tuple(reversed(np.floor(np.array(img.shape[:-1]) / 2)))

    # rot_img = rotate_image(img, theta, rot_center=rot_center)
    # transf_points = rotate_points(surf_kp, theta, rot_center=rot_center)

    scaled_img = scale_img(img, 1.5)
    scaled_pts = scale_points(surf_kp, 1.5)

    show_img(scaled_img, kp=scaled_pts)


if __name__ == '__main__':
    main()
