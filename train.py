#!/usr/bin/python3
# -*- coding: utf-8 -*-
from network import Network
from PIL import Image
from scale import size
import numpy as np
import os
import sys


def training(training_batches):
    training_losses = []
    lr = 0.0
    for batch_no, batch in enumerate(training_batches):
        loss, lr = net.train_step(batch, epoch=epoch)
        training_losses.append(loss)
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * int((batch_no + 1) / len(training_batches) * 50),
                                           int((batch_no + 1) / len(training_batches) * 100)))
        sys.stdout.flush()
    print("\nLearning rate: " + str(lr))
    return training_losses


def validation(validation_batches):
    print("Validation:")
    validation_losses = []
    for validation_batch_no, validation_batch in enumerate(validation_batches):
        validation_loss = net.validation_step(validation_batch)[0]
        validation_losses.append(validation_loss)
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * int((validation_batch_no + 1) / len(validation_batches) * 50),
                                           int((validation_batch_no + 1) / len(validation_batches) * 100)))
        sys.stdout.flush()
    print("\n")
    return validation_losses


def log_losses(losses, validation_losses):
    with open("losses.csv", 'a') as csv:
        csv.write(", ".join(
            [str(loss_value) for loss_value in [sum(validation_losses) / len(validation_losses)] + losses]) + "\n")


def inference_and_save_example_image(index):
    example_image = size(Image.open(os.path.join("images", "pexels-photo-25953.jpg")), (image_size[0], image_size[1]))
    inference_output = net.inference(images=[np.array(example_image) for _ in range(net.get_batch_size())])
    Image.fromarray(np.clip(inference_output[0], 0.0, 255.0).astype(np.uint8)).save(
        "test_image_epoch_" + str(index) + ".png")


if __name__ == "__main__":
    image_size = (320, 240)
    net = Network(dimensions=(320, 240), batch_size=5)
    net.initialize()
    if os.path.exists(os.path.abspath("nnetwork_params.index")):
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
        if len(batches) >= 200:
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
        if len(test_batches) >= 20:
            break
    val_losses = validation(test_batches)
    best_validation_loss = sum(val_losses) / len(val_losses)
    print("Validation loss: " + str(best_validation_loss))
    print("Params saved: " + net.save())
    output = net.inference(images=[np.array(
        size(Image.open(os.path.join("images", "pexels-photo-25953.jpg")),
             (image_size[0], image_size[1]))) for _ in range(net.get_batch_size())])
    Image.fromarray(np.clip(output[0], 0.0, 255.0).astype(np.uint8)).save("test_image_epoch_" + str(0) + ".png")
    for epoch in range(20000):
        print("\nTraining epoch " + str(epoch + 1) + " ...")
        train_losses = training(batches)
        print("Loss: " + str(train_losses) + "\n")
        val_losses = validation(test_batches)
        avg_validation_loss = sum(val_losses) / len(val_losses)
        print("Validation loss: " + str(avg_validation_loss))
        log_losses(train_losses, val_losses)
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            print("Params saved: " + net.save())
        else:
            print("Discarding the epoch because the validation accuracy did not improve.")
            net.load("network_params")
        inference_and_save_example_image(epoch + 1)
