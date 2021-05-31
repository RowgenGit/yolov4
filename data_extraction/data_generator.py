import tensorflow as tf

from data_extraction.load_data import *
from yolo_model.preprocessing import image_preprocess, preprocess_true_boxes
from data_extraction.data_augmentation import data_augmentation


default_data_augm_proba = [0.25, 0.25, 0.25, 0.25, 0.25]


class Data_Generator(object):
    def __init__(self, data, data_path, nb_classes, strides, anchors, input_size=416, batch_size=32,
                 data_augm_probabilities=default_data_augm_proba, anchors_per_scale=3, max_bbox_per_scale=100):
        self.data = data
        self.data_path = data_path
        self.sample_nb = self.data.shape[0]
        self.num_classes = nb_classes

        self.input_size = input_size
        self.batch_size = batch_size
        self.batch_nb = np.ceil(self.sample_nb / batch_size)
        self.strides = strides
        self.anchors = anchors
        self.anchors_per_scale = anchors_per_scale
        self.max_bbox_per_scale = max_bbox_per_scale
        self.train_output_sizes = self.input_size // self.strides

        self.data_augm_probabilities = data_augm_probabilities

        self.batch_count = 0

    def __len__(self):
        return int(self.batch_nb)

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):

            batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchors_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.batch_nb:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.sample_nb:
                        index -= self.sample_nb
                    image, bboxes = get_sample(self.data_path, self.data, index)
                    image, bboxes = data_augmentation(image, bboxes, *self.data_augm_probabilities)
                    image, bboxes = image_preprocess(np.copy(image), [self.input_size, self.input_size], np.copy(bboxes))
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = preprocess_true_boxes(bboxes, self.train_output_sizes, self.anchors, self.anchors_per_scale, self.num_classes, self.max_bbox_per_scale, self.strides)
                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1

                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                self.data = self.data.sample(frac=1).reset_index(drop=True)
                raise StopIteration
