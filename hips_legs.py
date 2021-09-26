import numpy as np
import cv2
from helper import get_model, output_img, show_img, gray, drawpts
from shoulder import extremeud, dist


def get_hips(img_src, model, threshold=0.8, stats=1, show=0, hght=160):
    # model = get_model()
    right = ['right_upper_leg_front', 'right_upper_leg_back', 'right_lower_leg_front', 'right_lower_leg_back',
             'right_feet']
    left = ['left_upper_leg_front', 'left_upper_leg_back', 'left_lower_leg_front', 'left_lower_leg_back', 'left_feet']

    img = cv2.imread(img_src)
    left_leg = output_img(src=img_src, model=model, thresh=threshold, part_flag=1, parts=left, outline=1,
                          show=0)
    right_leg = output_img(src=img_src, model=model, thresh=threshold, part_flag=1, parts=right, outline=1,
                           show=0)
    fullbody = output_img(src=img_src, model=model, thresh=threshold, part_flag=0, outline=1)

    top_left, bl = np.array(extremeud(gray(left_leg)))
    top_right, br = np.array(extremeud(gray(right_leg)))
    tmax, bmax = np.array(extremeud(gray(fullbody)))

    hiplen = dist(top_left, top_right)
    height = dist(tmax, bmax)
    ratio = hiplen / height
    hipwidth = np.round(ratio * hght, 2)

    if stats == 1:
        print(f'SHAPE: {fullbody.shape[1]} px wide,{fullbody.shape[0]} px tall', '\nLEFT: ', top_left, '\nRIGHT: ',
              top_right, '\nTOP: ', tmax, '\nBOTTOM: ', bmax)
        print('\nHip Length(px): ', hiplen, '\nHeight(px): ', height, '\nRatio: ', ratio)
        print(f'Approx. Hip Measurement: {hipwidth}cm for height {hght}cm')

    if show == 1:
        out1 = drawpts(img, tmax, bmax, top_left, top_right)
        out2 = drawpts(fullbody, tmax, bmax, top_left, top_right)
        show_img(out1, 'original')
        show_img(out2, 'outline')

    return hipwidth


def get_legs(img_src, model, threshold=0.8, stats=1, show=0, hght=160):
    # model = get_model()
    right = {'Ankle': 16, 'Hip': 12}
    left = {'Ankle': 15, 'Hip': 11}

    img = cv2.imread(img_src)
    poseposn = output_img(src=img_src, model=model, thresh=threshold, pose_flag=1)
    fullbody = output_img(src=img_src, model=model, thresh=threshold, outline=1)

    rank = np.array(poseposn[right['Ankle']])
    rhip = np.array(poseposn[right['Hip']])
    lank = np.array(poseposn[left['Ankle']])
    lhip = np.array(poseposn[left['Hip']])
    tmax, bmax = np.array(extremeud(gray(fullbody)))

    height = dist(tmax, bmax)
    ratio_left = dist(lank, lhip) / height
    ratio_right = dist(rank, rhip) / height
    ratio = (ratio_right + ratio_left) / 2
    leglength = np.round(ratio * hght, 2)

    if stats == 1:
        print(f'SHAPE: {fullbody.shape[1]} px wide,{fullbody.shape[0]} px tall', '\nLEFT HIP: ', lhip, '\nRIGHT HIP: ',
              rhip, '\nLEFT ANKLE: ', lank, '\nRIGHT ANKLE: ', rank, '\nTOP: ', tmax, '\nBOTTOM: ', bmax)
        print('\nLeft Leg Length Ratio: ', ratio_left, '\nRight Leg Length Ratio: ', ratio_right,
              '\nAverage Length Ratio: ', ratio)
        print(f'Approx. Leg Length Measurement: {leglength}cm for height {hght}cm')

    if show == 1:
        img = drawpts(img, u=tmax, d=bmax, color=1)
        img = drawpts(img, u=rank, d=rhip, l=lank, r=lhip)
        show_img(img, 'img')

    return leglength


if __name__ == '__main__':
    img_source = 'D:\Python Projects\BodyPix\TA Poses\\normal3.jpg'
    hipswidth = get_hips(img_source, model=get_model(), threshold=0.8, stats=1, show=1)
    leglen = get_legs(img_source, model=get_model(), threshold=0.8, stats=1, show=1)
