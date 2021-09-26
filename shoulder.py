import numpy as np
import cv2
from helper import get_model, output_img, show_img, gray, drawpts


def dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def extremelr(img):
    h = img.shape[0]
    w = img.shape[1]
    min = w - 1
    max = 0
    lpt, rpt = [0, 0], [0, 0]

    for i in range(h):
        for j in range(w):
            ele = img[i][j]
            if ele >= 200:
                if j < min:
                    min = j
                    lpt = [j, i]
                    """(j,i) because cv2.circle takes (width,height) format in points whereas i is height and j is width"""
                elif j > max:
                    max = j
                    rpt = [j, i]
    return lpt, rpt


def extremeud(img):
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


def get_shoulders(img_src, model, threshold=0.8, stats=1, show=0, hght=160):  # height in cm
    # model = get_model()
    parts = ['torso_front']

    img = cv2.imread(img_src)
    torso = output_img(src=img_src, model=model, thresh=threshold, part_flag=1, parts=parts, outline=1)
    fullbody = output_img(src=img_src, model=model, thresh=threshold, part_flag=0, parts=parts, outline=1)
    assert fullbody.shape == torso.shape

    l, r = np.array(extremelr(gray(torso)))
    u, d = np.array(extremeud(gray(fullbody)))

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
        out2 = drawpts(fullbody, u, d, l, r)
        show_img(out1, 'original')
        show_img(out2, 'outline')

    return shmeasurement


if __name__ == '__main__':
    img_source = 'D:\Python Projects\BodyPix\OG Poses\\pose4.jpg'
    shmeasure = get_shoulders(img_source, model=get_model(), threshold=0.8, stats=1, show=1)