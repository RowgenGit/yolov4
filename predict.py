import os
import argparse
import sys
from tensorflow.keras.models import load_model
import tensorflow as tf
from time import time
from threading import Thread
from yolo_model.detection import detect_image, detect_video, detect_directory


def get_model():
    return load_model("/builds/detectbot/yolov4/custom_model/trained_model")


def prepare_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "-s",
        "--size",
        default=416,
        type=int,
        help="Input size. Default is 416. Must be a multiple of 32.",
    )
    parser.add_argument(
        "-a",
        "--toto",
        default=False,
        type=bool,
        help="First call",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        default=0.5,
        type=float,
        help="Threshold to filter predictions. Default is 0.5",
    )
    parser.add_argument(
        "-it",
        "--iou_thresh",
        default=0.5,
        type=float,
        help="IOU Treshold to filter predictions in NMS. Default is 0.6.",
    )
    parser.add_argument(
        "-pf",
        "--postfix",
        default="_processed",
        type=str,
        help="Postfix to add to the files names. Default is processed",
    )
    return parser.parse_args()


def limit_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(gpus)
        try:
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        except RuntimeError as e:
            print(e)


def run_in_thread():
    model_loading_time = time()
    model = get_model()
    print("model loaded in: " + str(round(time() - model_loading_time, 2)) + " secs")
    prediction_data_path = "test_data/"
    results_path = "results/"
    try:
        os.mkdir(results_path)
    except FileExistsError:
        print("file " + results_path + " already exists")
    files = os.listdir(prediction_data_path)
    pred_time = time()
    for file in files:
        if file.endswith((".jpg", ".pdf", ".png", "jpeg")):
            file_pred_time = time()
            path = os.path.join(prediction_data_path, file)
            new_file = postfix.join(os.path.splitext(os.path.basename(file)))
            output_path = os.path.join(results_path, new_file)
            detect_image(model, path, output_path, size, thresh, iou_thresh, classes_path)
            print("image processed in: " + str(round(time() - file_pred_time, 2)) + " secs")

        elif file.endswith('.mp4'):
            file_pred_time = time()
            path = os.path.join(prediction_data_path, file)
            new_file = postfix.join(os.path.splitext(os.path.basename(file)))
            output_path = os.path.join(results_path, new_file)
            detect_video(model, path, output_path, size, thresh, iou_thresh, classes_path)
            print("video processed in: " + str(round(time() - file_pred_time, 2)) + " secs")

        elif os.path.isdir(os.path.join(prediction_data_path, file)):
            dir_pred_time = time()
            path = os.path.join(prediction_data_path, file)
            output_path = os.path.join(results_path, file)
            detect_directory(model, path, output_path, size, thresh, iou_thresh, classes_path)
            print("directory processed in: " + str(round(time() - dir_pred_time, 2)) + " secs")
    print("all data processed in: " + str(round(time() - pred_time, 2)) + " secs")


if __name__ == "__main__":
    print('IN MAIN')
    args = prepare_args()
    print(args)

    if args.toto:
        print('toto!')
        sys.exit(0)

    limit_gpu_usage()
    print("gpu config done")

    # prediction parameters
    size = args.size
    thresh = args.thresh
    iou_thresh = args.iou_thresh
    postfix = args.postfix

    classes_path = "custom_model/classes.txt"
    run_in_thread()
    thread = Thread(target = run_in_thread)
    thread.start()
    thread.join()
