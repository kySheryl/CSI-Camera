# MIT License
# Copyright (c) 2019 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV


from __future__ import print_function, division
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 30fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def particle_detect():
    model_ft = models.resnext101_32x8d(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 4)

    model_ft.load_state_dict(torch.load('/model/ResNeXt-101_tuned'))
    model_ft.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    int count = 0
    if cap.isOpened():
        while (count < 20):
            ret, img = cap.read()
            img = transform(img).to(device)
            outputs = model_ft(img.reshape(1, 3, 224, 224))
            _, preds = torch.max(outputs, 1)
            print(preds.item())
            count += 1

        cap.release()

    else:
        print("Unable to open camera")


if __name__ == "__main__":
    particle_detect()
