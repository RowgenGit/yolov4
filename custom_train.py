from yolo_model.create_model import load_model_from_pretrained, unzip_and_load
from data_extraction.load_data import load_csv_file, read_classes
from data_extraction.data_generator import Data_Generator
from yolo_model.training import train_step, validate_step, freeze_layers

import tensorflow as tf
import numpy as np
import os
import argparse
import struct

class WeightReader:
	def __init__(self, weight_file):
		with open(weight_file, 'rb') as w_f:
			major,	= struct.unpack('i', w_f.read(4))
			minor,	= struct.unpack('i', w_f.read(4))
			revision, = struct.unpack('i', w_f.read(4))
			if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
				w_f.read(8)
			else:
				w_f.read(4)
			transpose = (major > 1000) or (minor > 1000)
			binary = w_f.read()
		self.offset = 0
		self.all_weights = np.frombuffer(binary, dtype='float32')
 
	def read_bytes(self, size):
		self.offset = self.offset + size
		return self.all_weights[self.offset-size:self.offset]
 
	def load_weights(self, model):
		for i in range(106):
			try:
				conv_layer = model.get_layer('conv_' + str(i))
				print("loading weights of convolution #" + str(i))
				if i not in [81, 93, 105]:
					norm_layer = model.get_layer('bnorm_' + str(i))
					size = np.prod(norm_layer.get_weights()[0].shape)
					beta  = self.read_bytes(size) # bias
					gamma = self.read_bytes(size) # scale
					mean  = self.read_bytes(size) # mean
					var   = self.read_bytes(size) # variance
					weights = norm_layer.set_weights([gamma, beta, mean, var])
				if len(conv_layer.get_weights()) > 1:
					bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel, bias])
				else:
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel])
			except ValueError:
				print("no convolution #" + str(i))
 
	def reset(self):
		self.offset = 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "-nm",
        "--new_model",
        default="NO",
        type=str,
        help="If the model must be created before training. Default is False.",
    )
    parser.add_argument(
        "-t",
        "--tiny",
        default="NO",
        type=str,
        help="If the model is tiny",
    )
    parser.add_argument(
        "-s",
        "--size",
        default=416,
        type=int,
        help="Input size. Default is 416. Must be a multiple of 32.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=2,
        type=int,
        help="Size of batches. Default is 32",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs. Default is 50",
    )
    parser.add_argument(
        "-we",
        "--warmup_epochs",
        default=2,
        type=int,
        help="Number of warmup epochs. Default is 2",
    )
    parser.add_argument(
        "--lr_init",
        default=1e-4,
        type=float,
        help="Initial learning rate. Default is 1e-4",
    )
    parser.add_argument(
        "--lr_end",
        default=1e-6,
        type=float,
        help="Final learning rate. Default is 1e-6",
    )
    parser.add_argument(
        "--val_prop",
        default=0.2,
        type=float,
        help="Validation proportion. Default is 0.2",
    )
    args = parser.parse_args()

    # limit GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        except RuntimeError as e:
            print(e)

    new_model = args.new_model == "YES"
    is_tiny = args.tiny == "YES"

    # Model parameters
    input_size = args.size
    lr_init = args.lr_init
    lr_end = args.lr_end
    epochs = args.epochs
    batch_size = args.batch_size
    warmup_epochs = args.warmup_epochs
    val_prop = args.val_prop

    pretrained_weights = "model/yolov42.weights"
    pretrained_classes_path = "model/classes.txt"
    pretrained_classes = read_classes(pretrained_classes_path)
    nb_pretrained_classes = len(pretrained_classes)
    print(nb_pretrained_classes)

    if is_tiny:
        strides = np.array([16, 32, 64])
        anchors = (np.array([[[10, 14], [23, 27], [37, 58]],
                            [[81, 82], [135, 169], [344, 319]],
                            [[0, 0], [0, 0], [0, 0]]]).T / strides).T
    else:
        strides = np.array([8, 16, 32])
        anchors = (np.array([[[12, 16], [19, 36], [40, 28]],
                            [[36, 75], [76, 55], [72, 146]],
                            [[142, 110], [192, 243], [459, 401]]]).T/strides).T

    training_data_path = "training_data/"
    csv_path = training_data_path + "Annotations.csv"
    model_path = "custom_model/"
    classes_path = model_path + "classes.txt"
    classes = read_classes(classes_path)
    nb_classes = len(classes)
    checkpoint_path = model_path + "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    print("loading data")
    data = load_csv_file(csv_path, classes)
    data = data.sample(frac=1).reset_index(drop=True)
    train_nb = int(data.shape[0] * (1 - val_prop))
    train_data, val_data = data.iloc[:train_nb], data.iloc[train_nb:]

    train_dataset = Data_Generator(train_data, training_data_path, nb_classes, input_size=input_size,
                                   batch_size=batch_size, anchors=anchors, strides=strides)
    val_dataset = Data_Generator(val_data, training_data_path, nb_classes, input_size=input_size,
                                 batch_size=batch_size, anchors=anchors, strides=strides)
    print("training and validation data loaded and created")

    steps_per_epoch = len(train_dataset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    print("loading model")
    if new_model:
        print("creating a new model")
        yolo = load_model_from_pretrained(input_size, nb_classes, pretrained_weights, nb_pretrained_classes, is_tiny,
                                          anchors=anchors, strides=strides)
    else:
        print("loading existing model")
        yolo = unzip_and_load("custom_model", "model.zip")
    print("model loaded successfully")

    optimizer = tf.keras.optimizers.Adam()

    validate_writer = tf.summary.create_file_writer(checkpoint_path)
    train_writer = tf.summary.create_file_writer(checkpoint_path)

    best_val_loss = 100000  # should be large at start
    print("start training")

    # Freezing layers
    if not new_model:
        freeze_layers(yolo, 510)

    for epoch in range(epochs):
        if (not new_model) & (epoch == epochs//2):
            for idx_layer in range(len(yolo.layers)):
                yolo.layers[idx_layer].trainable = True
            print('all layers unfrozen')
        print("starting epoch: " + str(epoch + 1) + " / " + str(epochs))
        print(str(len(yolo.trainable_variables)) + " trainable variables")
        for image_data, target in train_dataset:
            results = train_step(yolo, image_data, target, nb_classes, optimizer, global_steps, warmup_steps,
                                 total_steps, lr_init, lr_end, train_writer, is_tiny, strides)
            cur_step = results[0] % steps_per_epoch
            print(
                "epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                    .format(epoch + 1, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4],
                            results[5]))

        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in val_dataset:
            results = validate_step(yolo, image_data, target, nb_classes, is_tiny, strides)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val / count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)
        validate_writer.flush()

        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val / count, conf_val / count, prob_val / count, total_val / count))

        if best_val_loss > total_val / count:
            save_directory = os.path.join(model_path + "trained_model")
            tf.keras.models.save_model(yolo, save_directory)
            best_val_loss = total_val / count

    print("training complete")
