# import tensorflow as tf
# from matplotlib import pyplot as plt
from tf_bodypix.api import download_model, load_model  # , BodyPixModelPaths
from tf_bodypix import draw
import cv2
import numpy as np


def show_img(img, label):
    im2 = cv2.resize(img, (300, 700))
    # im2 = img
    cv2.imshow(label, im2)
    cv2.waitKey(0)


def get_model():
    bodypix_model = load_model(download_model(
        "https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/float/model-stride16.json"))
    return bodypix_model


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def output_img(src, model, thresh=0.8, part_flag=0, parts=None, outline=0, show=0, pose_flag=0):
    if parts is None:
        parts = []
    img = cv2.imread(src)

    res = model.predict_single(img)
    mask = res.get_mask(thresh).numpy().astype(np.uint8)
    if part_flag == 1:
        mask = res.get_part_mask(mask, parts)

    part = cv2.bitwise_and(img, img, mask=mask)
    white = 255 * np.ones_like(part)
    if outline == 1:
        part = cv2.bitwise_or(part, white, mask=mask)

    if show == 1:
        show_img(img, 'image')
        show_img(part, 'prediction')

    if pose_flag == 1:
        poses = res.get_poses()
        max_keypoint_pose = 0
        max_score_pose = 0

        if len(poses) != 1:
            max_score_pose = np.argmax([poses[0].score, poses[1].score])
            max_keypoint_pose = np.argmax([len(poses[0].keypoints), len(poses[1].keypoints)])

        if show == 1:
            op = draw.draw_poses(image=img.copy(), poses=[poses[max_keypoint_pose]], min_score=0.5,
                                 keypoints_color=[255, 0, 0],
                                 skeleton_color=[0, 0, 255])

            show_img(op, 'poses')

        partpos = []
        for i in range(len(poses[max_keypoint_pose].keypoints)):
            s = poses[max_keypoint_pose].keypoints[i].position
            partpos.append([int(np.round(s.x)), int(np.round(s.y))])

        return partpos

    return part


def drawpts(img, u=None, d=None, l=None, r=None, color=0):
    res = img
    orng = [0, 69, 255]
    grn = [34, 139, 34]
    blk = [0, 0, 0]
    if color == 0:
        colors = [orng, orng, grn, grn]
    else:
        colors = [blk, blk, blk, blk]

    for i, a in enumerate([u, d, l, r]):
        if a is not None:
            res = cv2.circle(res, a, 3, colors[i], -1)

    if u is not None and d is not None:
        res = cv2.line(res, u, d, colors[-1], 1)
    if l is not None and r is not None:
        res = cv2.line(res, l, r, colors[0], 1)

    return res


if __name__ == '__main__':
    img_source = "D:\Python Projects\BodyPix\TA Poses\\normal5.jpg"

    _ = output_img(src=img_source, model=get_model(), thresh=0.8, part_flag=0, parts=[], outline=0, show=1,
                   pose_flag=1)
    print(_, '\n', len(_))

    pass
