import os
import cv2
import glob
import numpy as np

# set data path to your own dataset
dataset_path = "/workspace/data/VOCdevkit/VOC2007/JPEGImages" 

# set input size 
inputsize = {'h': 608, 'c': 3, 'w': 608}

# set input node name
#input_node = "yolov2/net1"
#input_node = "yolov4-rds/net1"
#input_node = "yolov3-tiny/net1"
input_node = "yolov3/net1"
calib_batch_size = 10

def convertimage(img, w, h, c):
    new_img = np.zeros((w, h, c))
    for idx in range(c):
        resize_img = img[:, :, idx]
        resize_img = cv2.resize(resize_img, (w, h), cv2.INTER_AREA)
        new_img[:, :, idx] = resize_img
    return new_img


# This function reads all images in dataset and return all images with the name of inputnode
def calib_input(iter):
    images = []
    line = glob.glob(dataset_path + "/*.j*") # either .jpg or .jpeg
    for index in range(0, calib_batch_size):
        curline = line[iter * calib_batch_size + index]
        calib_image_name = curline.strip()
        image = cv2.imread(calib_image_name)
        image = convertimage(image, inputsize["w"], inputsize["h"], inputsize["c"])
        image = image / 255.0
        images.append(image)
    return {input_node: images}  # first layer
