#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from network import Network
from PIL import Image
import os


def size(im, size=(240, 320)):
    im = im.crop((0, 0, min(im.size[0], round(im.size[1] / 240 * 320)), min(im.size[1], round(im.size[0] / 320 * 240))))
    return im.resize(size)


if __name__ == '__main__':
    image_size = [320, 240]
    batch_size = 10
    network = Network(image_size, batch_size)
    images = []
    for filename in os.listdir("images/"):
        im = Image.open(os.path.join("images", filename))
        print("Resizing " + filename)
        im = size(im)
        images.append(im)
        if filename == "pexels-photo-28340.jpg":
            im.show()
    network.train_step(images)
