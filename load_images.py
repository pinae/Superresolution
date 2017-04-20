#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import os

with open("~/images_2016_08/validation/images.csv", 'r') as f:
    while True:
        line = f.readline()
        url = line.split(',')[2]
        if url.startswith("http"):
            os.system("wget " + url + " --show-progress --directory-prefix=~/images_2016_08/validation/")
            print(url)
