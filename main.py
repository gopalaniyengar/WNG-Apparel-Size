import cv2
import numpy as np
import time

from helper import get_model, show_img
from hips_legs import get_hips, get_legs
from shoulder import get_shoulders


def ms():
    """
    Arguments: None
    Description: Returns time in millisecond
    Returns: Time in millisecond
    """

    return time.time_ns() / (10 ** 6)


def get_measurements(src, height, time_flag=0):
    """
        Arguments:
            src---> Absolute path of input image
            height---> Height of person in image
            time_flag---> If enabled, calculates execution time of function

        Description: Calculates shoulder, hip and leg measurements if time_flag disabled, execution time if enabled

        Returns:
            time_flag == 0: shoulder_measure, hip_measure, leg_measure
            time_flag == 1: model call time, shoulder exec. time, hip exec. time, leg exec. time
    """

    tm1 = ms()
    bodypix = get_model()
    tm2 = ms()

    t1 = ms()
    shoulder_measure = get_shoulders(img_src=src, model=bodypix, threshold=0.8, stats=0, show=0, hght=height)
    t2 = ms()
    hip_measure = get_hips(img_src=src, model=bodypix, threshold=0.8, stats=0, show=0, hght=height)
    t3 = ms()
    leg_measure = get_legs(img_src=src, model=bodypix, threshold=0.8, stats=0, show=0, hght=height)
    t4 = ms()

    if time_flag == 0:
        print(f'Approx. Shoulder Measurement: {shoulder_measure}cm for height {height}cm')
        print(f'Approx. Leg Length Measurement: {leg_measure}cm for height {height}cm')
        print(f'Approx. Hip Measurement: {hip_measure}cm for height {height}cm')
        return shoulder_measure, hip_measure, leg_measure, None

    else:
        print(f'Model Loading Time: {tm2 - tm1}ms')
        print(f'Shoulder Processing Time: {t2 - t1}ms')
        print(f'Hip Processing Time: {t3 - t2}ms')
        print(f'Leg Processing Time: {t4 - t3}ms\n\n')
        return tm2 - tm1, t2 - t1, t3 - t2, t4 - t3


if __name__ == '__main__':
    img_source = 'D:\Python Projects\BodyPix\TA Poses\\normal3.jpg'
    ht = 168
    # show_img(cv2.imread(img_source), 'Original Image')

    """avg = np.zeros(4)
    iters = 5
    for i in range(iters):
        ts = np.array(get_measurements(src=img_source, height=ht, time_flag=1))
        avg = avg + ts

    avg = avg / iters
    print(np.sum(avg))"""

    ts = np.array(get_measurements(src=img_source, height=ht, time_flag=0))
