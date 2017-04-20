#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import os
from PIL import Image

IMAGE_PATH = "../../images_2016_08/validation/"
for filename in os.listdir(IMAGE_PATH):
    try:
        im = Image.open(os.path.join(IMAGE_PATH, filename))
        width, height = im.size
        if width < 640 or height < 480:
            print("Image " + filename + " is too small. Deleting...")
            os.remove(os.path.join(IMAGE_PATH, filename))
        print("Image " + filename + " is OK.")
    except OSError:
        print("Image " + filename + " is broken. Deleting...")
        os.remove(os.path.join(IMAGE_PATH, filename))
