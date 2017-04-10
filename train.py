#!/usr/bin/python3
# -*- coding: utf-8 -*-
from network import Network
from PIL import Image
from scale import size
import numpy as np
import os
import sys

if __name__ == "__main__":
    image_size = (320, 240)
    net = Network(dimensions=(320, 240), batch_size=4)
    batches = []
    images = []
    for filename in os.listdir("images/"):
        im = size(Image.open(os.path.join("images", filename)), (image_size[0] * 2, image_size[1] * 2))
        images.append(np.array(im))
        if len(images) >= net.get_batch_size():
            batches.append(images)
            images = []
    for epoch in range(1):
        print("Training epoch " + str(epoch + 1) + " ...")
        for batch_no, batch in enumerate(batches):
            net.train_step(batch)
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('=' * int((batch_no + 1) / len(batches) * 50),
                                               int((batch_no + 1) / len(batches) * 100)))
            sys.stdout.flush()
