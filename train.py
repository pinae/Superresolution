#!/usr/bin/python3
# -*- coding: utf-8 -*-
from network import Network
from PIL import Image
from scale import size, load_batches
import numpy as np
import os
import sys


def training(input_training_batches, target_training_batches, net):
    training_losses = []
    lr = 0.0
    for batch_no, batch in enumerate(input_training_batches):
        loss, lr = net.train_step(batch, target_training_batches[batch_no], epoch=epoch)
        training_losses.append(loss)
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * int((batch_no + 1) / len(input_training_batches) * 50),
                                           int((batch_no + 1) / len(input_training_batches) * 100)))
        sys.stdout.flush()
    print("\nLearning rate: " + str(lr))
    return training_losses


def validation(input_validation_batches, target_validation_batches, net):
    print("Validation:")
    validation_losses = []
    for validation_batch_no, validation_batch in enumerate(input_validation_batches):
        validation_loss = net.validation_step(validation_batch, target_validation_batches[validation_batch_no])[0]
        validation_losses.append(validation_loss)
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('=' * int((validation_batch_no + 1) / len(input_validation_batches) * 50),
                                           int((validation_batch_no + 1) / len(input_validation_batches) * 100)))
        sys.stdout.flush()
    print("")
    return validation_losses


def log_losses(losses, validation_losses):
    with open("losses.csv", 'a') as csv:
        csv.write(", ".join(
            [str(loss_value) for loss_value in [sum(validation_losses) / len(validation_losses)] + losses]) + "\n")


def inference_and_save_example_image(index, net):
    example_image = Image.open(os.path.join("scaled_images", "validation", "small",
                                            os.listdir(os.path.join("scaled_images", "validation", "small"))[0]))
    inference_output = net.inference(images=[np.array(example_image) for _ in range(network.get_batch_size())])
    Image.fromarray(np.clip(inference_output[0], 0.0, 255.0).astype(np.uint8)).save(
        "test_image_epoch_" + str(index) + ".png")


if __name__ == "__main__":
    image_size = (320, 240)
    network = Network(dimensions=(320, 240), batch_size=5)
    network.initialize()
    if os.path.exists(os.path.abspath("network_params.index")):
        network.load("network_params")
    input_batches, target_batches = load_batches(os.path.join("scaled_images", "training"), network, 500)
    input_batches_test, target_batches_test = load_batches(os.path.join("scaled_images", "validation"), network, 20)
    val_losses = validation(input_batches_test, target_batches_test, network)
    best_validation_loss = sum(val_losses) / len(val_losses)
    print("Validation loss: " + str(best_validation_loss))
    print("Params saved: " + network.save())
    inference_and_save_example_image(0, network)
    for epoch in range(20000):
        print("\nTraining epoch " + str(epoch + 1) + " ...")
        train_losses = training(input_batches, target_batches, network)
        print("Loss: " + str(train_losses) + "\n")
        val_losses = validation(input_batches_test, target_batches_test, network)
        avg_validation_loss = sum(val_losses) / len(val_losses)
        print("Validation loss: " + str(avg_validation_loss))
        log_losses(train_losses, val_losses)
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            print("Params saved: " + network.save())
        else:
            print("Discarding the epoch because the validation accuracy did not improve.")
            network.load("network_params")
        inference_and_save_example_image(epoch + 1, network)
