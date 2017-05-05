# Superresolution
Double the size of images using a convolutional ANN.

## Installation
```
git clone https://github.com/pinae/Superresolution.git
cd Superresolution
pyvenv env
pip install --upgrade pip
pip install -r requirements.txt
```
If you have a Nvidia GPU and CUDA installed:
```
pip install tensorflow-gpu
```

## Usage
1. Collect training images in the `images` folder and some validation 
images in `images/validation`. 
2. Scale all training and validation images: `python scale.py`
3. If you want to start with an itialized network skip this Step: `rm checkpoint network_params.*`
4. Train network: `python train.py`
5. For inferencing change `inference.py` according to your needs. After that run: `python inference.py`

## Examples
Some Videos on YouTube:
* [unsuccessful learning](https://youtu.be/QpeQ4vZyUOk)
* [unsuccessful learning with slower learning rate](https://youtu.be/6N8e936fj0Y)
* [nearly converging but too deep network](https://youtu.be/P5iRfjeTl4c)
* [optimization towards a black image as a local minima](https://youtu.be/TyZCkg3CExw)
* [simple reproduction of the input image with a 1-layer-testnet](https://youtu.be/Is197jjGFTE)
* [successful training of the network in this repository](https://youtu.be/ZwvH3Pxk8UM)