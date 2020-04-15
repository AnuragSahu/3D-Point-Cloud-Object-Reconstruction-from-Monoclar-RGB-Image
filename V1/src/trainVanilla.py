import os
import tflearn
import numpy as np
#import tf_nndistance
import tensorflow as tf
#import cPickle as pickle
from BatchFetcher import * # File to import Data Files

batchSize = 32
h = 192
w = 256
pclSize = 16384

lastbatch = None  # Initialization
lastconsumed = FETCH_BATCH_SIZE # Defined in BatchFetcher
OUTPUTPOINTS = 1024

from tensorflow.python.framework import ops
nn_distance_module=tf.load_op_library('./tf_nndistance_so.so')

def nn_distance(xyz1,xyz2):
	'''
Computes the distance of nearest neighbors for a pair of point clouds
input: xyz1: (batch_size,#points_1,3)  the first point cloud
input: xyz2: (batch_size,#points_2,3)  the second point cloud
output: dist1: (batch_size,#point_1)   distance from first to second
output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
output: dist2: (batch_size,#point_2)   distance from second to first
output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
	'''
	return nn_distance_module.nn_distance(xyz1,xyz2)
#@tf.RegisterShape('NnDistance')
#def _nn_distance_shape(op):
	#shape1=op.inputs[0].get_shape().with_rank(3)
	#shape2=op.inputs[1].get_shape().with_rank(3)
	#return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
		#tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
	xyz1=op.inputs[0]
	xyz2=op.inputs[1]
	idx1=op.outputs[1]
	idx2=op.outputs[3]
	return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)


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

def getConfig():
	config=tf.ConfigProto() # For session configuration Dimag mat lagao bahut options h
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	return config, tf.train.Saver() # File pointer to save the Training Values/Weights

def buildGraph(resourceid): # Make the Model Architecture Here
# input : resourceid, it is the GPU number which we want to Use
# TODO remove the resourceid and CUDA deps

	#with tf.device('/gpu:%d'%resourceid):
	with tf.device('/cpu'):
		tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=0.9,soft_placement=True)
		x = img_inp=tf.placeholder(tf.float32,shape=(batchSize,h,w,4),name='img_inp') # Equivalent to np.empty for images
		
		#Start Building the graph from here
		#192 256
		x = getConvolution2dLayer(x)
		x = getConvolution2dLayer(x)
		x = getConvolution2dLayer(x, win = 32, strides = 2)
		#96 128
		x = getConvolution2dLayer(x, win = 32)
		x = getConvolution2dLayer(x, win = 32)
		x = getConvolution2dLayer(x, win = 64, strides = 2)
		#48 64
		x = getConvolution2dLayer(x, win = 64)
		x = getConvolution2dLayer(x, win = 64)
		x = getConvolution2dLayer(x, win = 128, strides = 2)
		#24 32
		x = getConvolution2dLayer(x, win = 128)
		x = getConvolution2dLayer(x, win = 128)
		x = getConvolution2dLayer(x, win = 256, strides = 2)
		#12 16
		x = getConvolution2dLayer(x, win = 256)
		x = getConvolution2dLayer(x, win = 256)
		x = getConvolution2dLayer(x, win = 512, strides = 2)
		#6 8
		x = getConvolution2dLayer(x, win = 512)
		x = getConvolution2dLayer(x, win = 512)
		x = getConvolution2dLayer(x, win = 512)

		x = getConvolution2dLayer(x, win = 512, window = (5,5), strides = 2)
		x = getFullyConnectedLayer(x, 2048)
		x = getFullyConnectedLayer(x, 1024)
		x = getFullyConnectedLayer(x, 256*3, activation='linear')
		# Reshape the final array to get the desired 3d points
		x = tf.reshape(x, (batchSize, 256, 3))

		point_ground_truth = tf.placeholder(tf.float32,shape=(batchSize,POINTCLOUDSIZE,3),name='pt_gt')
		error_forward, _, error_backward, _ = nn_distance(point_ground_truth, x) # point_ground_truth is the ground truth with size 32*16384*3 and x is 32*256*3,
																			#its only 256 not 1024 beacuse its vanilla version
		loss_nodecay = (tf.reduce_mean(error_forward)+tf.reduce_mean(error_backward)/2.0)*10000
		regularizerTerm = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		regularizerTerm *= 0.1
		loss=loss_nodecay+regularizerTerm
		batchno = tf.Variable(0, dtype=tf.int32) # tensorflow Variable having Inital value Zero
		mindist=error_forward
		
		
	return [img_inp,x,point_ground_truth,loss,tf.train.AdamOptimizer(3e-5*batchSize/FETCH_BATCH_SIZE).minimize(loss,global_step=batchno),
		   batchno,batchno.assign(batchno+1),mindist,loss_nodecay,tf.reduce_mean(error_forward),tf.reduce_mean(error_backward),mindist[0,:]]
	# Returns place holder for input image, predicated_3d point cloud, ground truth point cloud, optimizer which is a function
	# TODO
	# loss_nodecay?
	# batchnoinc, batchno


# This is the main training Loop
def main(resourceid,keyname):
	global lastbatch,lastconsumed
	config, chkFile = getConfig()
	img_inp,x,point_ground_truth,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,error_forward,error_backward,dist0=buildGraph(resourceid)
	# Gets the place holder for All the things
	
	with tf.Session(config=config) as sess, open('%s/%s.log'%(dumpdir,keyname),'a') as fout:   # Session Defn

		sess.run(tf.global_variables_initializer())
		bno=sess.run(batchno) # gets the initial value for batch number ie gradient step
		fetchworker.bno = bno//(FETCH_BATCH_SIZE/batchSize) # fetch worker is an instance of Class BatchFetcher
		fetchworker.start()
		trainloss_accs=[0,0,0]
		trainloss_acc0=1e-9 # Training Loss
		while bno<300000:
			#data,ptcloud,validating = fetch_batch() # Gets the Input data, Ground Truth, Validation Bit
			if lastbatch is None or lastconsumed+batchSize > FETCH_BATCH_SIZE:
				lastbatch = fetchworker.fetch() # fetch worker is an instance of BatchFetcher Class
				lastconsumed = 0

			data, ptcloud, validating = [ i[lastconsumed:lastconsumed+batchSize] for i in lastbatch] # Gets Images, Pointclouds, validation bits
			lastconsumed += batchSize
				
				# Trainloss = loss_nodecay
				# total_loss = loss
				# trainloss1 = distance_forward
				# trainloss2 = distanse_backward
				# distmap_0 = dist0
			iterationOutput = sess.run([optimizer,x,loss,loss_nodecay,error_forward,error_backward,dist0],
						  feed_dict={img_inp:data,point_ground_truth:ptcloud}) # Getting all the Training Values
				
			#_,_,_,_,_,_,_ = iterationOutput
			trainloss_accs=[trainloss_accs[0]*0.99+iterationOutput[3], trainloss_accs[1]*0.99+iterationOutput[4], trainloss_accs[2]*0.99+iterationOutput[5]]
			trainloss_acc0=trainloss_acc0*0.99+1

			bno=sess.run(batchno) # again retreiving the gradient step
			x_loss = trainloss_accs[0]/trainloss_acc0
			y_loss = trainloss_accs[1]/trainloss_acc0
			z_loss = trainloss_accs[2]/trainloss_acc0
			print >>fout, bno,x_loss, y_loss, z_loss, iterationOutput[2]-iterationOutput[3]
			# Printed all the values in fout file
			print bno,'t',trainloss_accs[0]/trainloss_acc0, iterationOutput[2]-iterationOutput[3],fetchworker.queue.qsize()
		chkFile.save(sess,'%s/'%dumpdir+keyname+".ckpt") # after 3 lakh gradient steps on one batch of 32 images

if __name__=='__main__':
	resourceid = 0
	datadir,dumpdir,cmd,valnum="data","dump","predict",3
	datadir = "../../data/"
	dumpdir = "../output/"
	num = 2
	os.system("mkdir -p %s"%dumpdir)
	fetchworker=BatchFetcher(datadir)

	keyname=os.path.basename(__file__).rstrip('.py')
	main(resourceid,keyname)
	fetchworker.shutdown()
	#stop_fetcher()
