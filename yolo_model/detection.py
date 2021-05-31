import os
import cv2
import colorsys
import numpy as np

from data_extraction.load_data import read_classes
from yolo_model.preprocessing import image_preprocess
from yolo_model.postprocessing import postprocess_boxes


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to score order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def draw_bbox(image, bboxes, classes, show_label=True, show_confidence=True, text_colors=(255, 255, 0)):
    classes_dict = read_classes(classes)
    num_classes = len(classes_dict)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            try:
                label = "{}".format(list(classes_dict.keys())[class_ind]) + score_str
            except KeyError:
                print(classes_dict)
                print(class_ind)
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")
                label = "Error in code"

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline),
                          bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image


def detect_frame(model, image, size, score_thresh, iou_thresh, classes):
    image_data = image_preprocess(np.copy(image), [size, size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    if len(pred_bbox) == 6:
        pred_bbox = [pred_bbox[i] for i in [1, 3, 5]]

    bboxes = postprocess_boxes(pred_bbox, image, size, score_thresh)
    bboxes = nms(bboxes, iou_thresh, method='nms')

    image = draw_bbox(image, bboxes, classes)

    return image, bboxes


def detect_image(model, path, output_path, size, score_thresh, iou_thresh, classes):
    image = cv2.imread(path)
    detection_image, _ = detect_frame(model, image, size, score_thresh, iou_thresh, classes)
    cv2.imwrite(output_path, detection_image)


def detect_video(model, path, output_path, size, score_thresh, iou_thresh, classes):
    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        raise IOError("Couldn't open video: " + path)
    video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    while vid.isOpened():
        return_value, frame = vid.read()
        if not return_value:
            break
        detection_image, _ = detect_frame(model, frame, size, score_thresh, iou_thresh, classes)
        out.write(detection_image)
    vid.release()
    out.release()


def write_txt(path, bboxes):
    file = open(path, "x")
    for i, bbox in enumerate(bboxes):
        file.write(str(i) + " ")
        file.write(str(int(bbox[-1])) + " ")

        x0 = round(bbox[0], 1)
        y0 = round(bbox[1], 1)
        w = round(bbox[2] - x0, 1)
        h = round(bbox[3] - y0, 1)
        file.write(str(x0) + " ")
        file.write(str(y0) + " ")
        file.write(str(w) + " ")
        file.write(str(h) + " ")

        file.write('\n')
    file.close()


def detect_directory(model, path, output_path, size, score_thresh, iou_thresh, classes):
    os.mkdir(output_path)
    files = os.listdir(path)
    for file in files:
        if not file.endswith((".jpg", ".pdf", ".png", "jpeg")):
            continue
        full_path = os.path.join(path, file)
        image = cv2.imread(full_path)
        detection_image, bboxes = detect_frame(model, image, size, score_thresh, iou_thresh, classes)
        img_output_path = os.path.join(output_path, file)
        cv2.imwrite(img_output_path, detection_image)
        output_file = file[:-4] + ".txt"
        output_full_path = os.path.join(output_path, output_file)
        write_txt(output_full_path, bboxes)
