import cv2
import time
import numpy as np
from helper import get_model, output_img, show_img, gray, drawpts


def ms():
    """
    Arguments: None
    Description: Returns time in millisecond
    Returns: Time in millisecond
    """

    return time.time_ns() / (10 ** 6)


def dist(a, b):
    """
        Arguments:
            a, b---> Two pixel points

        Description: Calculates Euclidean distance between two points

        Returns:
            Length of line connecting a and b
    """

    return np.sqrt(np.sum(np.square(a - b)))


def extremeud(img):
    """
        Arguments:
            img---> Input image (output of segmentation model with outline enabled)

        Description: Finds the top-most and bottom-most white points in the image

        Returns:
            upt, dpt---> Extreme vertical white points in (width,height) format
    """

    h = img.shape[0]
    w = img.shape[1]
    upt, dpt = [0, 0], [0, 0]
    ustate = 1

    for i in range(h):
        for j in range(w):
            ele = img[i][j]
            if ele >= 200:
                if ustate == 1:
                    upt = [j, i]
                    ustate = 0
                dpt = [j, i]
    dpt = [upt[0], dpt[1]]
    return upt, dpt


def extremeud_alt(img):
    """
    3X more efficient than previous implementation

        Arguments:
            img---> Input image (output of segmentation model with outline enabled)

        Description: Finds the top-most and bottom-most white points in the image

        Returns:
            upt, dpt---> Extreme vertical white points in (width,height) format
    """

    h = img.shape[0]
    w = img.shape[1]
    upt, dpt = [0, 0], [0, 0]
    ustate, dstate = 1, 1

    for i in range(h):
        for j in range(w):
            ele_up = img[i][j]
            ele_down = img[h - 1 - i][w - 1 - j]
            if ustate == 1:
                if ele_up >= 220:
                    upt = [j, i]
                    ustate = 0
            if dstate == 1:
                if ele_down >= 220:
                    dpt = [w - 1 - j, h - 1 - i]
                    dstate = 0
    dpt = [upt[0], dpt[1]]
    return upt, dpt


def extremelr(img, top, bot):
    """
    Made robust by restricting search area for extreme points

        Arguments:
            img---> Input image (output of segmentation model with outline enabled)
            top---> Top-most point of torso
            bot---> Bottom-most point of torso

        Description: Finds the left-most and right-most white points in the image

        Returns:
            lpt, rpt---> Extreme horizontal white points in (width,height) format
    """

    # h = img.shape[0]
    w = img.shape[1]
    precision = 4
    h_range = np.ceil((bot[1] - top[1]) / precision).astype(int).item()
    min = w - 1
    max = 0
    lpt, rpt = [0, 0], [0, 0]

    for i in range(top[1], top[1] + h_range):
        for j in range(w):
            ele = img[i][j]
            if ele >= 220:
                if j < min:
                    min = j
                    lpt = [j, i]
                    """(j,i) because cv2.circle takes (width,height) format in points whereas i is height and j is width"""
                elif j > max:
                    max = j
                    rpt = [j, i]
    return lpt, rpt


def get_shoulders(img_src, model, fullbody, u, d, threshold=0.8, stats=0, show=0, hght=160):  # height in cm
    """
        Arguments:
            img_src---> Absolute path of input image
            model---> BodyPix model
            fullbody---> Full body segmented image
            u---> Top-most point of fullbody
            d---> Bottom-most point of fullbody
            threshold---> Model predictions confidence threshold, i.e. predicts points with [confidence >= thresh], 0.8 by default
            stats---> If enabled, displays prediction process in more detail, disabled by default
            show---> If enabled, displays predicted images, disabled by default
            hght---> The actual height of the person in image, input by user, 160cm by default

        Description: Gets shoulder measurement of person given height

        Returns:
            shmeasurement---> Shoulder width measurement
    """

    parts = ['torso_front']

    img = cv2.imread(img_src)
    torso = output_img(src=img_src, model=model, thresh=threshold, part_flag=1, parts=parts, outline=1)
    # fullbody = output_img(src=img_src, model=model, thresh=threshold, part_flag=0, parts=parts, outline=1)
    assert fullbody.shape == torso.shape

    # u, d = np.array(extremeud_alt(gray(fullbody)))
    top_torso, bot_torso = np.array(extremeud_alt(gray(torso)))
    l, r = np.array(extremelr(gray(torso), top=top_torso, bot=bot_torso))

    shlen = dist(l, r)
    height = dist(u, d)
    ratio = shlen / height
    shmeasurement = np.round(ratio * hght, 2)

    if stats == 1:
        print(f'SHAPE: {torso.shape[1]} px wide,{torso.shape[0]} px tall', '\nLEFT: ', l, '\nRIGHT: ', r, '\nTOP: ', u,
              '\nBOTTOM: ', d)
        print('\nShoulder Length(px): ', shlen, '\nHeight(px): ', height, '\nRatio: ', ratio)
        print(f'Approx. Shoulder Measurement: {shmeasurement}cm for height {hght}cm')

    if show == 1:
        out1 = drawpts(img, u, d, l, r)
        out2 = drawpts(fullbody.copy(), u, d, l, r)
        # show_img(torso)
        show_img(out1, 'original')
        show_img(out2, 'outline')

    return shmeasurement


if __name__ == '__main__':
    img_source = 'D:\Python Projects\BodyPix\TA Poses\\normal2.jpg'
    shmeasure = get_shoulders(img_source, model=get_model(), threshold=0.8, stats=0, show=1)
