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
def show_img(img, name=''):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def load_imgs(img_folder):
    imgs = []
    for img in os.listdir(img_folder):
        img_location = f'{img_folder}/{img}'
        imgs.append(cv.imread(img_location))
    return imgs

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





def main():
    imgs = load_imgs(img_folder)
    # show_img(imgs[0])

if __name__ == '__main__':
    main()