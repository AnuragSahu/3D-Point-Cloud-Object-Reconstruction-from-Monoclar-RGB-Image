import sys
import tflearn
import cv2 as cv
import numpy as np
import tensorflow as tf
import cPickle as pickle

debug = False

def run_image(model,image,image_mask):
	image_filtered = image*(1-image_mask[:,:,None])+191*image_mask[:,:,None]
	if(debug):
		cv.imwrite(sys.argv[1][:-4]+"_filtered.jpg", image_filtered)
	sess, imageVector, x = model
	input_image = np.dstack([image_filtered.astype('float32')/255,
		                     image_mask[:,:,None]])
	if(input_image.shape!=(h,w,4)):
		print("Mismatch in input, make sure that the input image and masks are of size 256,192")
		return

	(ret,),=sess.run([x],feed_dict={imageVector:input_image[None,:,:,:]})
	return ret

def getWeights(filePath):
	loaddict={}
	weightPath = open(filePath,'rb')
	while True:
		try:
			key, value = pickle.load(weightPath)
			loaddict[key] = value
		except (EOFError):
			break
		
	weightPath.close()
	return loaddict

def getFullyConnectedLayer(x, win, activation='relu', weight_decay=1e-2, regularizer = 'L2'):
	return tflearn.layers.core.fully_connected(x, win, activation = activation,
											   weight_decay = weight_decay, 
											   regularizer = regularizer)

def getConvolution2dLayer(x,  win = 16, window = (3, 3), strides = 1, activation = 'relu', weight_decay = 1e-5, regularizer = 'L2'):
	return tflearn.layers.conv.conv_2d(x,  win, window, 
									   strides = strides, 
									   activation = activation, 
									   weight_decay = weight_decay, 
									   regularizer = regularizer)


def getConv2dTranspose(x,win = 256, window = [5,5], win1 = [6,8],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2'):
	return tflearn.layers.conv.conv_2d_transpose(x,win,window,win1,strides=strides,activation=activation,weight_decay=weight_decay,regularizer=regularizer)

def buildGraph(weightsfile, batchSize, h, w):
	with tf.device('/cpu'):
		x = imageVector=tf.placeholder(tf.float32,name='imageVector',shape=(batchSize,h,w,4))
		#Start Building the graph from here
		#192 256
		x = getConvolution2dLayer(x)
		x = getConvolution2dLayer(x)
		x0 = x
		x = getConvolution2dLayer(x, win = 32, strides = 2)
		#96 128
		x = getConvolution2dLayer(x, win = 32)
		x = getConvolution2dLayer(x, win = 32)
		x1 = x
		x = getConvolution2dLayer(x, win = 64, strides = 2)
		#48 64
		x = getConvolution2dLayer(x, win = 64)
		x = getConvolution2dLayer(x, win = 64)
		x2 = x
		x = getConvolution2dLayer(x, win = 128, strides = 2)
		#24 32
		x = getConvolution2dLayer(x, win = 128)
		x = getConvolution2dLayer(x, win = 128)
		x3 = x
		x = getConvolution2dLayer(x, win = 256, strides = 2)
		#12 16
		x = getConvolution2dLayer(x, win = 256)
		x = getConvolution2dLayer(x, win = 256)
		x4 = x
		x = getConvolution2dLayer(x, win = 512, strides = 2)
		#6 8
		x = getConvolution2dLayer(x, win = 512)
		x = getConvolution2dLayer(x, win = 512)
		x = getConvolution2dLayer(x, win = 512)
		x5 = x

		x = getConvolution2dLayer(x, win = 512, window = (5,5), strides = 2)
		x_add = getFullyConnectedLayer(x, 2048)
		x_add = getFullyConnectedLayer(x_add, 1024)
		x_add = getFullyConnectedLayer(x_add, 256*3, activation='linear')
		# Reshape the final array to get the desired 3d points
		x_add = tf.reshape(x_add, (batchSize, 256, 3))
	
		x = getConv2dTranspose(x)
		x5 = getConvolution2dLayer(x5, win = 256,activation='linear')
		x = tf.nn.relu(tf.add(x,x5))
		x = getConvolution2dLayer(x, win = 256)
		x = getConv2dTranspose(x,win=128,win1=[12,16])	

		x4 = getConvolution2dLayer(x4, win = 128 ,activation='linear')
		x = tf.nn.relu(tf.add(x,x4))
		x = getConvolution2dLayer(x, win = 128)
		x = getConv2dTranspose(x,win=64, win1=[24,32])	

		x3 = getConvolution2dLayer(x3, win = 64, activation='linear')
		x = tf.nn.relu(tf.add(x,x3))
		x = getConvolution2dLayer(x, win = 64, activation='relu')
		x = getConvolution2dLayer(x, win = 64, activation='relu')
		x = getConvolution2dLayer(x, win = 3, activation='linear')

		x = tf.reshape(x, (batchSize, 32*24,3))
		x = tf.concat([x_add, x],axis=1)
		x = tf.reshape(x,(batchSize, 1024,3))

	sess=tf.Session('')
	sess.run(tf.global_variables_initializer())
	loaddict = getWeights(weightsfile)
	for t in tf.trainable_variables():
		sess.run(t.assign(loaddict[t.name]))
		del loaddict[t.name] # remove the used weight to free memory
	return (sess, imageVector, x)


if __name__=='__main__':
	batchSize=1
	h=192
	w=256
	image = cv.imread(sys.argv[1])
	image_mask = cv.imread(sys.argv[2],0)!=0
	model = buildGraph(sys.argv[3], batchSize, h, w)
	points = run_image(model,image,image_mask)
	coordinates_file = open(sys.argv[1][:-4]+'.txt','w')
	for x,y,z in points:
		print >>coordinates_file,x,y,z
