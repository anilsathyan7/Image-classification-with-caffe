import caffe
import numpy as np
import os
import timeit
import pyttsx
from SimpleCV import *
from easygui import *

#Configure the tts engine
engine = pyttsx.init()
engine.setProperty('voice', 'english+f3')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-50)



MODEL_FILE = '/home/anilsathyan7/caffe-caffe-0.16/python/alexnet_classify/deploy.prototxt'
WEIGHT_CAFFEMODEL = '/home/anilsathyan7/caffe-caffe-0.16/python/alexnet_classify/snapshot_iter_110880.caffemodel'
MEAN = '/home/anilsathyan7/caffe-caffe-0.16/python/alexnet_classify/out.npy'
WEBCAM = '/home/anilsathyan7/caffe-caffe-0.16/python/image.png'
LABELS = '/home/anilsathyan7/caffe-caffe-0.16/python/alexnet_classify/labels.txt'

#Using Alexnet
#For standard input tomato-109.jpg : GPU_EX_TIME = 0.0415749549866 ; CPU_EX_TIME = 0.291295051575 (i.e time for net.forward()-exclusive)
# 7x speed-up observed !!

caffe.set_mode_gpu() #Use GPU
caffe.set_device(0)

#Choose a Convnet
engine.say('Good morning, choose the required network')
engine.runAndWait()
engine.runAndWait()  #keep running tts

msg ="Choose a neural network"
title = "Convnet"
choices = ["Alexnet", "GoogLeNet", "Caffenet"]
choice = choicebox(msg, title, choices)

if choice == 'Alexnet':
	im_size=227
	engine.say('Initializing Alexnet ...')
elif choice == 'GoogLeNet':
	im_size=224
	engine.say('Initializing GoogLeNet ...')
	MODEL_FILE = '/home/anilsathyan7/caffe-caffe-0.16/python/googlenet_classify/deploy.prototxt'
	WEIGHT_CAFFEMODEL = '/home/anilsathyan7/caffe-caffe-0.16/python/googlenet_classify/snapshot_iter_137970.caffemodel'


# Neural Networkl Initialization ...
net = caffe.Net(MODEL_FILE,WEIGHT_CAFFEMODEL, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load(MEAN).mean(1).mean(1))

transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1,3,im_size,im_size)


#Choose Image for Classification

engine.say('Now, choose an image to classify')

choice=indexbox(msg="Select an image to classify", title="Classification",choices=['Local','Camera'])


if choice == 0:
	engine.say('Choose the image from the directory') #speak
	IMAGE = fileopenbox("Choose the required image") # '/media/anilsathyan7/work/imdb/47/tomato-109.jpg'
	
elif choice == 1:
	engine.say('Click on the window to take a picture') #speak
        cam = Camera()
	disp = Display()

	while disp.isNotDone():
       		img = cam.getImage()
        	if disp.mouseLeft:
                	break
        	img.save(disp)
		img.save("image.png")
	IMAGE = WEBCAM

img = caffe.io.load_image(IMAGE)

engine.say('Click yes to continue') #speak
image = IMAGE
msg = "Selected image is shown below. Press 'Yes' to start classification !"
choices = ["Yes","No"]
reply = buttonbox(msg, image=image, choices=choices)


if reply == 'Yes':

	net.blobs['data'].data[...] = transformer.preprocess('data', img)

	start_time = timeit.default_timer()
	output = net.forward()
	elapsed = timeit.default_timer() - start_time

	print output['softmax'].argmax()
	Class = output['softmax'].argmax()
        output_prob = output['softmax'][0]   #probabilities
	

	label_mapping = np.loadtxt(LABELS, str, delimiter='\t')
	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	Output = " \nIt looks like a " + label_mapping[top_inds][0][1]  # Output class name
	Heading = "\n\nLabels and Probabilities :- \n\n"                        # Heading

        
        Values=''                                   # Labels and probabilities string
	for i in   range(0,5):
		Values = Values + str(label_mapping[top_inds][i]) + ' - ' + str(output_prob[top_inds][i])+'\n'


	Result = Output + Heading + Values + "\n\nTime taken: "+ str(elapsed)
	engine.say(Output) #speak
	print Result
	textbox("Result :-","Classification",str(Result))
