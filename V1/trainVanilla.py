import sys
import tensorflow as tf
import numpy as np
import cv2
import tflearn
import random
import math
import os
#os.system("chmod +w /unsullied/sharefs/wangmengdi/wangmengdi")
import time
import zlib
import socket
import threading
import Queue
import tf_nndistance
import cPickle as pickle

OUTPUTPOINTS=1024

from BatchFetcher import * # File to import Data Files


lastbatch = None  # Initialization
lastconsumed = FETCH_BATCH_SIZE # Defined in BatchFetcher

def fetch_batch():  # Functions to fetch one batch of Data of Size 32
	global lastbatch,lastconsumed
	if lastbatch is None or lastconsumed+BATCH_SIZE > FETCH_BATCH_SIZE:
		lastbatch = fetchworker.fetch() # fetch worker is an instance of BatchFetcher Class
		lastconsumed = 0

	ret = [ i[lastconsumed:lastconsumed+BATCH_SIZE] for i in lastbatch] # Gets Images, Pointclouds, validation bits
	lastconsumed += BATCH_SIZE
	return ret

def stop_fetcher():
	fetchworker.shutdown()

def build_graph(resourceid): # Make the Model Architecture Here
# input : resourceid, it is the GPU number which we want to Use
# TODO remove the resourceid and CUDA deps

	with tf.device('/gpu:%d'%resourceid):
		tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=0.9,soft_placement=True)
		img_inp=tf.placeholder(tf.float32,shape=(BATCH_SIZE,HEIGHT,WIDTH,4),name='img_inp') # Equivalent to np.empty for images
		pt_gt=tf.placeholder(tf.float32,shape=(BATCH_SIZE,POINTCLOUDSIZE,3),name='pt_gt')

		x=img_inp # size of Image : 192 256
		x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x0=x
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x1=x
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x2=x
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x3=x
		x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x4=x
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
		x5=x
		x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
		##################################################### ENCODER DONE ################################################################################
		x_additional=tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
		###################################################### DECODER HERE############################################################################
		x_additional=tflearn.layers.core.fully_connected(x_additional,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
		x_additional=tflearn.layers.core.fully_connected(x_additional,256*3,activation='linear',weight_decay=1e-3,regularizer='L2')
		x_additional=tf.reshape(x_additional,(BATCH_SIZE,256,3))
		# x=tflearn.layers.conv.conv_2d(x,3,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
		# x=tf.reshape(x,(BATCH_SIZE,32*24,3))
		# x=tf.concat([x_additional,x],1)
		# x=tf.reshape(x,(BATCH_SIZE,OUTPUTPOINTS,3))
		x = x_additional		#Addi
		dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(pt_gt,x) # pt_gt is the ground truth with size 32*16384*3 and x is 32*256*3,
																			#its only 256 not 1024 beacuse its vanilla version
		mindist=dists_forward
		dist0=mindist[0,:]
		dists_forward=tf.reduce_mean(dists_forward)  # Calculates Mean like np.mean
		dists_backward=tf.reduce_mean(dists_backward)
		loss_nodecay = (dists_forward+dists_backward/2.0)*10000
		loss=loss_nodecay+tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1
		batchno = tf.Variable(0, dtype=tf.int32) # tensorflow Variable having Inital value Zero
		optimizer = tf.train.AdamOptimizer(3e-5*BATCH_SIZE/FETCH_BATCH_SIZE).minimize(loss,global_step=batchno)#  global_step: Optional `Variable` to increment by one after the
        																									   #   variables have been updated.
		batchnoinc=batchno.assign(batchno+1)
	return img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0
	# Returns place holder for input image, predicated_3d point cloud, ground truth point cloud, optimizer which is a function
	# TODO
	# loss_nodecay?
	# batchnoinc, batchno


# This is the main training Loop
def main(resourceid,keyname):
	if not os.path.exists(dumpdir): # creating a directory to store weights for further usage
		os.system("mkdir -p %s"%dumpdir)

	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid)
	# Gets the place holder for All the things
	config=tf.ConfigProto() # For session configuration Dimag mat lagao bahut options h
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver = tf.train.Saver() # File pointer to save the Training Values/Weights
	with tf.Session(config=config) as sess,\
				open('%s/%s.log'%(dumpdir,keyname),'a') as fout:   # Session Defn

		sess.run(tf.global_variables_initializer())
		trainloss_accs=[0,0,0]
		trainloss_acc0=1e-9 # Training Loss
		validloss_accs=[0,0,0]
		validloss_acc0=1e-9 # validation Loss
		lastsave=time.time()
		bno=sess.run(batchno) # gets the initial value for batch number ie gradient step
		fetchworker.bno = bno//(FETCH_BATCH_SIZE/BATCH_SIZE) # fetch worker is an instance of Class BatchFetcher
		fetchworker.start()
		while bno<300000:
			t0=time.time()
			data,ptcloud,validating = fetch_batch() # Gets the Input data, Ground Truth, Validation Bit
			t1=time.time() # start time
			validating = validating[0]!=0
			if not validating: # For Training
				_,pred,total_loss,trainloss,trainloss1,trainloss2,distmap_0=sess.run([optimizer,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
					feed_dict={img_inp:data,pt_gt:ptcloud}) # Getting all the Training Values
				# Trainloss = loss_nodecay
				# total_loss = loss
				# trainloss1 = distance_forward
				# trainloss2 = distanse_backward
				# distmap_0 = dist0

				trainloss_accs[0]=trainloss_accs[0]*0.99+trainloss
				trainloss_accs[1]=trainloss_accs[1]*0.99+trainloss1
				trainloss_accs[2]=trainloss_accs[2]*0.99+trainloss2
				trainloss_acc0=trainloss_acc0*0.99+1
			else:
				_,pred,total_loss,validloss,validloss1,validloss2,distmap_0=sess.run([batchnoinc,x,loss,loss_nodecay,dists_forward,dists_backward,dist0],
					feed_dict={img_inp:data,pt_gt:ptcloud})
				validloss_accs[0]=validloss_accs[0]*0.997+validloss
				validloss_accs[1]=validloss_accs[1]*0.997+validloss1
				validloss_accs[2]=validloss_accs[2]*0.997+validloss2
				validloss_acc0=validloss_acc0*0.997+1
			t2=time.time() # end timer
			down=2

			bno=sess.run(batchno) # again retreiving the gradient step
			if not validating:
				showloss=trainloss
				showloss1=trainloss1
				showloss2=trainloss2
			else:
				showloss=validloss
				showloss1=validloss1
				showloss2=validloss2
			print >>fout,bno,trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,showloss,showloss1,showloss2,validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss
			# Printed all the values in fout file
			if bno%128==0:
				fout.flush()
			if time.time()-lastsave>900: # inevery time interval 900 seconds or apprx 15 mins create checkpoints
				saver.save(sess,'%s/'%dumpdir+keyname+".ckpt")
				lastsave=time.time()
			print bno,'t',trainloss_accs[0]/trainloss_acc0,trainloss_accs[1]/trainloss_acc0,trainloss_accs[2]/trainloss_acc0,'v',validloss_accs[0]/validloss_acc0,validloss_accs[1]/validloss_acc0,validloss_accs[2]/validloss_acc0,total_loss-showloss,t1-t0,t2-t1,time.time()-t0,fetchworker.queue.qsize()
		saver.save(sess,'%s/'%dumpdir+keyname+".ckpt") # after 3 lakh gradient steps on one batch of 32 images

def dumppredictions(resourceid,keyname,valnum):
	img_inp,x,pt_gt,loss,optimizer,batchno,batchnoinc,mindist,loss_nodecay,dists_forward,dists_backward,dist0=build_graph(resourceid)
	config=tf.ConfigProto()
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	saver=tf.train.Saver()
	fout = open("%s/%s.v.pkl"%(dumpdir,keyname),'wb') # open the existing model
	with tf.Session(config=config) as sess:
		#sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())
		saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))
		fetchworker.bno=0
		fetchworker.start()
		cnt=0
		for i in xrange(0,300000):
			t0=time.time()
			data,ptcloud,validating=fetch_batch()
			validating=validating[0]!=0
			if not validating:
				continue
			cnt+=1
			pred,distmap=sess.run([x,mindist],feed_dict={img_inp:data,pt_gt:ptcloud})
			pickle.dump((i,data,ptcloud,pred,distmap),fout,protocol=-1)
			print i,'time',time.time()-t0,cnt
			if cnt>=valnum:
				break
	fout.close()

if __name__=='__main__':
	resourceid = 0
	datadir,dumpdir,cmd,valnum="data","dump","predict",3
	for pt in sys.argv[1:]:
		if pt[:5]=="data=":
			datadir = pt[5:]
		elif pt[:5]=="dump=":
			dumpdir = pt[5:]
		elif pt[:4]=="num=":
			valnum = int(pt[4:])
		else:
			cmd = pt
	if datadir[-1]=='/':
		datadir = datadir[:-1]
	if dumpdir[-1]=='/':
		dumpdir = dumpdir[:-1]
	assert os.path.exists(datadir),"data dir not exists"
	os.system("mkdir -p %s"%dumpdir)
	fetchworker=BatchFetcher(datadir)
	print "datadir=%s dumpdir=%s num=%d cmd=%s started"%(datadir,dumpdir,valnum,cmd)

	keyname=os.path.basename(__file__).rstrip('.py')
	try:
		if cmd=="train":
			main(resourceid,keyname)
		elif cmd=="predict":
			dumppredictions(resourceid,keyname,valnum)
		else:
			assert False,"format wrong"
	finally:
		stop_fetcher()
