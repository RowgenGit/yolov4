import random
import numpy as np
import cv2


def random_horizontal_flip(image, bboxes, probability):
    if random.random() < probability:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes


def random_crop(image, bboxes, probability):
    if random.random() < probability:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes


def random_translate(image, bboxes, probability):
    if random.random() < probability:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes


def random_bright_contrast_modif(image, bboxes, probability):
    if random.random() < probability:
        alpha = 0.05 + 2 * random.random()
        beta = 40 * random.random()
        image = cv2.convertScaleAbs(image.copy(), alpha=alpha, beta=beta)
    return image, bboxes


def random_grey(image, bboxes, probability):
    if random.random() < probability:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image, bboxes


def data_augmentation(image, bboxes, flip_proba, crop_proba, trans_proba, grey_proba, bright_proba):
    image, bboxes = random_horizontal_flip(np.copy(image), np.copy(bboxes), flip_proba)
    image, bboxes = random_crop(np.copy(image), np.copy(bboxes), crop_proba)
    image, bboxes = random_translate(np.copy(image), np.copy(bboxes), trans_proba)
    image, bboxes = random_grey(np.copy(image), np.copy(bboxes), grey_proba)
    image, bboxes = random_bright_contrast_modif(np.copy(image), np.copy(bboxes), bright_proba)
    return image, bboxes
