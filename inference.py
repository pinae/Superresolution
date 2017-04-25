#!/usr/bin/python3
# -*- coding: utf-8 -*-
from network import Network
from PIL import Image
from scale import size
import numpy as np
import os

if __name__ == "__main__":
    image_size = (320, 240)
    net = Network(dimensions=(320, 240), batch_size=1, initialize_loss=False)
    net.load()
    im = size(Image.open(os.path.join("images", "pexels-photo-25953.jpg")), (image_size[0], image_size[1]))
    output = net.inference(images=[np.array(im)])
    Image.fromarray(np.clip(output[0], 0.0, 255.0).astype(np.uint8)).show()
