# import tensorflow as tf
# from matplotlib import pyplot as plt
from tf_bodypix.api import download_model, load_model  # , BodyPixModelPaths
from tf_bodypix import draw
import cv2
import numpy as np


def show_img(img, label='default label'):
    """
        Arguments:
            img---> Input image
            label---> Corresponding label to be shown

        Description: Displays image in '300 width x 700 height' frame

        Returns: None
    """

    im2 = cv2.resize(img, (300, 700))
    # im2 = img
    cv2.imshow(label, im2)
    cv2.waitKey(0)


def get_model():
    """
        Arguments: None

        Description: Downloads BodyPix model Python version

        Returns: bodypix_model
    """

    bodypix_model = load_model(download_model(
        "https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/float/model-stride16.json"))
    return bodypix_model


def gray(img):
    """
        Arguments:
           img---> Input image

        Description: Color to grayscale conversion of image

        Returns: Grayscale image version of input
    """

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def output_img(src, model, thresh=0.8, part_flag=0, parts=None, outline=0, show=0, pose_flag=0):
    """
        Arguments:
            src---> Absolute path of input image
            model---> BodyPix model
            thresh---> Model predictions confidence threshold, i.e. predicts points with [confidence >= thresh], 0.8 by default
            part_flag---> If enabled, predicts locations of only those parts given in 'parts', disabled by default
            parts---> List of parts to predict {see 'part_names.txt'}, set to 'None' by default
            outline---> If enabled, gives white filled areas corresponding to predicted parts, disabled by default
            show---> If enabled, displays the predicted output, be it part segmentations, or the detected pose, disabled by default
            pose_flag---> If enabled, gets model's predicted poses on the image

        Description: Gets various types of outputs out of model on an input image

        Returns:
            pose_flag == 0: part---> Output image containing required body parts, white-filled if 'outline' is enabled
            pose_flag == 1: partpos---> List of pixels corresponding to the locations of the key-points predicted by model {see 'part_names.txt'}
    """

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

    if pose_flag != 1 and show == 1:
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
    """
        Arguments:
            img---> Input image
            u, d, l, r---> Upper, Lower, Left, Right points
            color---> If enabled, draws black non-contrasting line, disabled by default

        Description: Draws two pairs of points(u,d and l,r), and lines connecting those pairs

        Returns:
            res---> Output image
    """

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
    foo = output_img(src=img_source, model=get_model(), thresh=0.8, part_flag=0, parts=[], outline=0, show=1,
                     pose_flag=1)
