# -*- coding: utf-8 -*-
"""
Created on Mon Feb. 17 2020

this code is used for prediction the biomass.

Note: instead of using the fully_connected(), conv2d() and dropout() functions from the tensorflow.contrib.layers module (as in the book), 
we now use the dense(), conv2d() and dropout() functions (respectively) from the tf.layers module, which did not exist when this chapter was written. 
This is preferable because anything in contrib may change or be deleted without notice, while  tf.layers is part of the official API. As you will see, the code is mostly the same.
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import csv
import os
from sklearn.preprocessing import OneHotEncoder

import pdb

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Clear nodes on graph
tf.reset_default_graph()


input_dim=9
output_dim = 1
maxEpoch=100000
batch_size = 2000
d_learning_rate=1e-4
dropout_rate=0.5
#n_fc1 = 256
#fc1_dropout_rate = 0.5


he_init = tf.contrib.layers.variance_scaling_initializer()
"""
xe means cross-entropy
mse means mean squared error
"""
_srng = np.random.RandomState(np.random.randint(1,2147462579))
                
# initial state
d_loss = 0.0
best_loss = np.infty
max_checks_without_progress = 1000
checks_without_progress = 0

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

        
def data_loader(features, labels, batch_size, shuffle=False): 
        if shuffle:
                indices = np.arange(len(features))
                _srng.shuffle(indices)
        for start_idx in range(0, len(features) - batch_size + 1, batch_size):
                if shuffle:
                        excerpt = indices[start_idx:start_idx + batch_size]
                else:
                        excerpt = slice(start_idx, start_idx + batch_size)
                yield features[excerpt], labels[excerpt]
                

      
# build computation graph of model
in_array = tf.placeholder(tf.float32, shape=[None,input_dim]) #input arrays
out = tf.placeholder(tf.float32, shape=[None,output_dim]) #output categories
training = tf.placeholder_with_default(False, shape=[], name='training')

#array = tf.expand_dims(in_array, -1) 

h0 = tf.layers.dense(in_array,512, kernel_initializer=he_init, name="fcl1")
#h0_drop=tf.layers.dropout(h0, dropout_rate, training=training)
h1 = tf.layers.dense(h0, 512, kernel_initializer=he_init, activation=lrelu, name="fcl2")
#h1_drop=tf.layers.dropout(h1, dropout_rate, training=training)
h2 = tf.layers.dense(h1, 256, kernel_initializer=he_init, activation=lrelu, name="fcl3")
#h2_drop=tf.layers.dropout(h2, dropout_rate, training=training)
h3 = tf.layers.dense(h2, 128, kernel_initializer=he_init, activation=lrelu, name="fcl4")
h3_drop=tf.layers.dropout(h3, dropout_rate, training=training)
#h1 = tf.layers.dense(h0_drop, 32, kernel_initializer=he_init, name="fcl2")


logits = tf.layers.dense(h3_drop, output_dim, kernel_initializer=he_init,  name="logit")
#pdb.set_trace()


d_loss = tf.reduce_mean(tf.squared_difference(logits,out))        

d_vars = tf.trainable_variables()

d_optimizer = tf.train.AdamOptimizer(d_learning_rate)
d_grads = d_optimizer.compute_gradients(d_loss, d_vars)
clip_d_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in d_grads if grad is not None]
d_optimizer = d_optimizer.apply_gradients(clip_d_grads)

#preparing screening sample size
#randomly generating input parameter combinations 
with open('WD_comb_matrix.csv') as csvfile:
    mpg = list(csv.reader(csvfile))
    WD_comb = np.array(mpg).astype("float")

with open('Input_parameter_matrix.csv') as csvfile:
    mpg = list(csv.reader(csvfile))
    test_set_6_params = np.array(mpg).astype("float")

nn=1    
sample_size = np.size(test_set_6_params, 0)
N0=[0.2, 0.4, 0.6, 0.8, 1.0] 
num_WD=np.size(WD_comb,0)  
b_length=len(N0)*num_WD 
input_N0=np.multiply(np.ones((num_WD,1)),N0).flatten('F')

for itera in range(nn):
    print(itera)
    test_set_6_params = np.random.rand(sample_size,6)
    saved_matrix1=np.empty((sample_size,b_length+6))
    saved_matrix2=np.empty((sample_size,b_length+6))
    saved_matrix3=np.empty((sample_size,b_length+6))
    for i in range(2):
        print(i)
        test_data=np.concatenate((np.multiply(np.ones((b_length,1)),test_set_6_params[i,:]),np.expand_dims(input_N0, axis=1), np.tile(WD_comb,(len(N0),1))),axis=1)
        saver = tf.train.Saver()                  
        with tf.Session() as sess:
            saver.restore(sess, "NN1/saved_model/mymodel.ckpt")
            pred_test1=logits.eval(feed_dict={in_array:test_data}) 
        
        with tf.Session() as sess:
            saver.restore(sess, "NN2/saved_model/mymodel.ckpt")
            pred_test2=logits.eval(feed_dict={in_array:test_data}) 
            np.shape(pred_test2)
            
        with tf.Session() as sess:
            saver.restore(sess, "NN3/saved_model/mymodel.ckpt")
            pred_test3=logits.eval(feed_dict={in_array:test_data}) 
        #pdb.set_trace()
        saved_matrix1[i,:]=np.concatenate((np.expand_dims(test_set_6_params[i,:],axis=0),np.transpose(pred_test1)),axis=1)
        saved_matrix2[i,:]=np.concatenate((np.expand_dims(test_set_6_params[i,:],axis=0),np.transpose(pred_test2)),axis=1)
        saved_matrix3[i,:]=np.concatenate((np.expand_dims(test_set_6_params[i,:],axis=0),np.transpose(pred_test3)),axis=1)
    with open('screening_NN1.txt','ab') as ff1:
        np.savetxt(ff1, saved_matrix1)
    with open('screening_NN2.txt','ab') as ff2:
        np.savetxt(ff2, saved_matrix2)  
    with open('screening_NN3.txt','ab') as ff3:
        np.savetxt(ff3, saved_matrix3)      
