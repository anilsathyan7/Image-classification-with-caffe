# Image-classification-with-caffe
It is a python program for classifying images using a deep learning tool called Caffe.
The system accepts images from local storage or from webcam in real-time.It finally outputs the
predicted class labels with corresponding probabilities.It has an easy to use GUI for selecting images
through webcam or local filesystem.The caffe model was trained on over 2 lakh images with the help of 
a GPU and consists of 50 different classes.

## Getting Started

Recommended basic knowledge of Python, Image Processing and Linux.

## Prerequisites

PC with Ubuntu 16.04 and NVIDIA GPU (optional).

### Installing

See:-

https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide

http://caffe.berkeleyvision.org/installation.html

https://github.com/NVIDIA/DIGITS

Additionally install the following libraries from terminal

Easygui : pip install --upgrade easygui

SimpleCV : https://github.com/sightmachine/SimpleCV

Also install Numpy and Pyttsx

If you are using NVIDIA GPU, ensure proper CUDA drivers are installed.

### Running the tests

First, make sure the caffe and related libraries are installed properly.Also, make sure your webcam is working properly.
Keep some images for testing purpose.See the 'labels.txt' file for getting an idea about image classes and labels.

Replce the file path as required for the inputs.

In terminal:-

Firstly,make  sure caffe path is properly set and execute the following commands:-

python fin_imclassify.py

NB: Python v2.7, Pyttsx gives background speech prompts (i.e. Audio... Just 4 Fun !!!)

Now follow the prompts and test the system with any suitable images.
In the case of webcam just click on the webcam window to capture the image.

## Screenshots

### Choose Network
![alt text](https://sevenshinestudios.files.wordpress.com/2018/04/1choosenetwork.png)

### Classify Image

Select | Result
------------ | -------------
![alt text](https://i1.wp.com/sevenshinestudios.files.wordpress.com/2018/04/2selectimage.png?ssl=1&w=450) | ![alt text](https://i0.wp.com/sevenshinestudios.files.wordpress.com/2018/04/3fileexplore.png?ssl=1&w=450)
![alt text](https://i2.wp.com/sevenshinestudios.files.wordpress.com/2018/04/4imageconfirm.png?ssl=1&w=450) | ![alt text](https://i0.wp.com/sevenshinestudios.files.wordpress.com/2018/04/5result.png?ssl=1&w=450)

 ### Webcam
<p align="center">
  <img  src="https://i0.wp.com/sevenshinestudios.files.wordpress.com/2018/04/6camwindow.png?ssl=1&w=450">
</p>


## Versioning

Version 1.0

## Authors

Anil Sathyan
## License

Free to use, share or modify!! ... Copyleft!! :sparkles: :boom:

## Acknowledgments
* "http://shengshuyang.github.io/A-step-by-step-guide-to-Caffe.html"
* "http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/"
* "http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html"
* "https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md"
* "http://caffe.berkeleyvision.org/"
* "https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb" 
