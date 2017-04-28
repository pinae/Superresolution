#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PIL import Image
import numpy as np
import os
from random import shuffle


def size(im, size=(240, 320)):
    im = im.crop((0, 0, min(im.size[0], round(im.size[1] / 240 * 320)), min(im.size[1], round(im.size[0] / 320 * 240))))
    return im.resize(size, resample=Image.BICUBIC)


def load_batches(directory, net, batch_count):
    input_batches = []
    target_batches = []
    current_input_batch = []
    current_target_batch = []
    filenames = os.listdir(os.path.join(directory, "big"))
    shuffle(filenames)
    for filename in filenames:
        try:
            big_img = np.array(Image.open(os.path.join(directory, "big", filename)))
            small_img = np.array(Image.open(os.path.join(directory, "small", filename)))
            try:
                if big_img.shape[2] != 3:
                    print("No color file: " + filename)
                    continue
            except IndexError:
                print("No color file: " + filename)
                continue
        except ZeroDivisionError or OSError:
            print("Errorous file: " + filename)
            continue
        current_input_batch.append(small_img)
        current_target_batch.append(big_img)
        if len(current_input_batch) >= net.get_batch_size():
            input_batches.append(current_input_batch)
            target_batches.append(current_target_batch)
            current_input_batch = []
            current_target_batch = []
            print(str(len(input_batches)) + " batches from " + directory + " read.")
        if len(input_batches) >= batch_count:
            break
    return input_batches, target_batches


def create_or_clear_dir(directory):
    if not os.path.isdir(directory):
        print("Creating folder: " + directory)
        os.mkdir(directory)
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            os.remove(os.path.join(directory, filename))


if __name__ == '__main__':
    input_image_size = [320, 240]
    scale_factor = 2
    images_folder = "images"
    validation_images_folder = os.path.join(images_folder, "validation")
    scaled_images_folder = "scaled_images"
    create_or_clear_dir(scaled_images_folder)
    create_or_clear_dir(os.path.join(scaled_images_folder, "training"))
    create_or_clear_dir(os.path.join(scaled_images_folder, "training", "big"))
    create_or_clear_dir(os.path.join(scaled_images_folder, "training", "small"))
    create_or_clear_dir(os.path.join(scaled_images_folder, "validation"))
    create_or_clear_dir(os.path.join(scaled_images_folder, "validation", "big"))
    create_or_clear_dir(os.path.join(scaled_images_folder, "validation", "small"))
    for filename in os.listdir(images_folder):
        if os.path.isfile(os.path.join(images_folder, filename)):
            try:
                original_image = Image.open(os.path.join(images_folder, filename))
                np_image = np.array(original_image)
                try:
                    if np_image.shape[1] < input_image_size[0] * scale_factor or \
                       np_image.shape[0] < input_image_size[1] * scale_factor:
                        print("Image is too small: " + filename)
                        continue
                    if np_image.shape[2] != 3:
                        print("No color file: " + filename)
                        continue
                except IndexError:
                    print("No color file: " + filename)
                    continue
                print("Resizing " + filename)
                big_image = size(original_image, (input_image_size[0] * scale_factor,
                                                  input_image_size[1] * scale_factor))
                small_image = size(original_image, (input_image_size[0], input_image_size[1]))
                big_image.save(os.path.join(scaled_images_folder, "training", "big", filename),
                               'jpeg', quality=95)
                small_image.save(os.path.join(scaled_images_folder, "training", "small", filename),
                                 'jpeg', quality=95)
            except ZeroDivisionError or OSError:
                print("Erroneous file: " + filename)
                continue
    if os.path.isdir(validation_images_folder):
        for filename in os.listdir(validation_images_folder):
            if os.path.isfile(os.path.join(validation_images_folder, filename)):
                try:
                    original_image = Image.open(os.path.join(validation_images_folder, filename))
                    np_image = np.array(original_image)
                    try:
                        if np_image.shape[1] < input_image_size[0] * scale_factor or \
                           np_image.shape[0] < input_image_size[1] * scale_factor:
                            print("Image is too small: " + filename)
                            continue
                        if np_image.shape[2] != 3:
                            print("No color file: " + filename)
                            continue
                    except IndexError:
                        print("No color file: " + filename)
                        continue
                    print("Resizing " + filename)
                    big_image = size(original_image, (input_image_size[0] * scale_factor,
                                                      input_image_size[1] * scale_factor))
                    small_image = size(original_image, (input_image_size[0], input_image_size[1]))
                    big_image.save(os.path.join(scaled_images_folder, "validation", "big", filename),
                                   'jpeg', quality=95)
                    small_image.save(os.path.join(scaled_images_folder, "validation", "small", filename),
                                     'jpeg', quality=95)
                except ZeroDivisionError or OSError:
                    print("Erroneous file: " + filename)
                    continue
    else:
        print("There is no " + validation_images_folder + " folder.")
