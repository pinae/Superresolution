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
    net = Network(dimensions=(320, 240), batch_size=10)
    net.initialize()
    if os.path.exists(os.path.abspath("network_params.index")):
        net.load("network_params")
    batches = []
    images = []
    for filename in os.listdir("../../images_2016_08/train/"):
        try:
            im = size(Image.open(os.path.join("../../images_2016_08/train/", filename)),
                      (image_size[0] * net.get_scale_factor(), image_size[1] * net.get_scale_factor()))
            try:
                if np.array(im).shape[2] != 3:
                    print("No color file: " + filename)
                    continue
            except IndexError:
                print("No color file: " + filename)
                continue
        except ZeroDivisionError or OSError:
            print("Errorous file: " + filename)
            continue
        images.append(np.array(im))
        if len(images) >= net.get_batch_size():
            batches.append(images)
            images = []
            print(str(len(batches)) + " batches scaled.")
        if len(batches) >= 100:
            break
    test_batches = []
    test_images = []
    for filename in os.listdir("../../images_2016_08/validation/"):
        try:
            im = size(Image.open(os.path.join("../../images_2016_08/validation/", filename)),
                      (image_size[0] * net.get_scale_factor(), image_size[1] * net.get_scale_factor()))
            try:
                if np.array(im).shape[2] != 3:
                    print("No color file: " + filename)
                    continue
            except IndexError:
                print("No color file: " + filename)
                continue
        except ZeroDivisionError or OSError:
            print("Errorous file: " + filename)
            continue
        test_images.append(np.array(im))
        if len(test_images) >= net.get_batch_size():
            test_batches.append(test_images)
            test_images = []
            print(str(len(test_batches)) + " validation batches scaled.")
        if len(test_batches) >= 10:
            break
    output = net.inference(images=[np.array(
        size(Image.open(os.path.join("images", "pexels-photo-25953.jpg")),
             (image_size[0], image_size[1]))) for _ in range(net.get_batch_size())])
    o = output[0]
    print(o[100][100])
    Image.fromarray(o.astype(np.uint8)).save("test_image_epoch_" + str(0) + ".png")
    for epoch in range(20000):
        print("Training epoch " + str(epoch + 1) + " ...")
        losses = []
        for batch_no, batch in enumerate(batches):
            loss, lr = net.train_step(batch, epoch=epoch)
            losses.append(loss)
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('=' * int((batch_no + 1) / len(batches) * 50),
                                               int((batch_no + 1) / len(batches) * 100)))
            sys.stdout.flush()
        print("\nLearning rate: " + str(lr))
        print("Loss: " + str(losses))
        print("\nValidation:")
        losses = []
        for batch_no, batch in enumerate(test_batches):
            loss = net.validation_step(batch)[0]
            losses.append(loss)
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d%%" % ('=' * int((batch_no + 1) / len(test_batches) * 50),
                                               int((batch_no + 1) / len(test_batches) * 100)))
            sys.stdout.flush()
        print("\nValidation loss: " + str(sum(losses)/len(losses)))
        print("\nParams saved: " + net.save() + "\n")
        im = size(Image.open(os.path.join("images", "pexels-photo-25953.jpg")), (image_size[0], image_size[1]))
        output = net.inference(images=[np.array(im) for _ in range(net.get_batch_size())])
        Image.fromarray(np.clip(output[0], 0.0, 255.0).astype(np.uint8)).save("test_image_epoch_"+str(epoch+1)+".png")
