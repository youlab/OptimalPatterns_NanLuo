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


# Load data from csv file (the data file needs to be shuffled before loading for best performance)
with open('training_data_normalized.csv') as csvfile:
    mpg = list(csv.reader(csvfile))
    training_data = np.array(mpg).astype("float")
with open('test_data_normalized.csv') as csvfile:
    mpg = list(csv.reader(csvfile))
    test_data = np.array(mpg).astype("float")



train_input = training_data[:,:9]
train_output = training_data[:,9:]
test_input = test_data[:,:9]
test_output = test_data[:,9:]

train_size=len(train_output)
test_size=len(test_output)


model_path = "saved_model"
input_dim=9
output_dim = 1
maxEpoch=100000
batch_size = 2000
d_learning_rate=1e-3
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
max_checks_without_progress = 5000
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
#h3_drop=tf.layers.dropout(h3, dropout_rate, training=training)
#h1 = tf.layers.dense(h0_drop, 32, kernel_initializer=he_init, name="fcl2")

logits = tf.layers.dense(h3, output_dim, kernel_initializer=he_init,  name="logit")
#pdb.set_trace()


d_loss = tf.reduce_mean(tf.squared_difference(logits,out))        

d_vars = tf.trainable_variables()

d_optimizer = tf.train.AdamOptimizer(d_learning_rate)
d_grads = d_optimizer.compute_gradients(d_loss, d_vars)
clip_d_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in d_grads if grad is not None]
d_optimizer = d_optimizer.apply_gradients(clip_d_grads)                

#Train the model  
saver = tf.train.Saver()                  
            

with tf.Session() as sess:
    saver.restore(sess, "NN1/saved_model/mymodel.ckpt")
    loss_train1 = d_loss.eval(feed_dict={in_array:train_input, out: train_output})
    loss_test1 = d_loss.eval(feed_dict={in_array:test_input, out: test_output})   
    pred_train1=logits.eval(feed_dict={in_array:train_input, out: train_output})
    pred_test1=logits.eval(feed_dict={in_array:test_input, out: test_output}) 

with tf.Session() as sess:
    saver.restore(sess, "NN2/saved_model/mymodel.ckpt")
    loss_train2=d_loss.eval(feed_dict={in_array:train_input, out: train_output})
    loss_test2=d_loss.eval(feed_dict={in_array:test_input, out: test_output})   
    pred_train2=logits.eval(feed_dict={in_array:train_input, out: train_output})
    pred_test2=logits.eval(feed_dict={in_array:test_input, out: test_output}) 

with tf.Session() as sess:
    saver.restore(sess, "NN3/saved_model/mymodel.ckpt")
    loss_train3 = d_loss.eval(feed_dict={in_array:train_input, out: train_output})
    loss_test3 = d_loss.eval(feed_dict={in_array:test_input, out: test_output})  
    pred_train3=logits.eval(feed_dict={in_array:train_input, out: train_output})
    pred_test3=logits.eval(feed_dict={in_array:test_input, out: test_output}) 

    #pdb.set_trace()
saved_data_train=np.zeros((len(pred_train1),8)) #gt, pred1X3,indx, pred_ensemble, mdiv_train, error_train
saved_data_test=np.zeros((len(pred_test1),8)) #gt, pred1,pred2,pred3,indx, pred_ensemble, mdiv_train, error_train

for kb in range(train_size):
    pred_train=np.r_[pred_train1[kb],pred_train2[kb],pred_train3[kb]]
    rmse1=rmse(pred_train[0]*np.ones((2)),np.delete(pred_train,0))
    rmse2=rmse(pred_train[1]*np.ones((2)),np.delete(pred_train,1))
    rmse3=rmse(pred_train[2]*np.ones((2)),np.delete(pred_train,2))

    m_train=np.r_[rmse1,rmse2,rmse3]
    mdiv_train=np.mean(m_train) #average disagreements among NNs      
    idx_train=np.argmin(m_train)
    pred_train_ensemble=pred_train[idx_train]
    error_train_ensemble=rmse(train_output[kb], pred_train_ensemble)
    saved_data_train[kb,:]=np.r_[train_output[kb],pred_train,idx_train,pred_train_ensemble,mdiv_train,error_train_ensemble]
       
for kb in range(test_size):  
    pred_test=np.r_[pred_test1[kb],pred_test2[kb],pred_test3[kb]]
    rmse1=rmse(pred_test[0]*np.ones((2)),np.delete(pred_test,0))
    rmse2=rmse(pred_test[1]*np.ones((2)),np.delete(pred_test,1))
    rmse3=rmse(pred_test[2]*np.ones((2)),np.delete(pred_test,2))

    m_test=np.r_[rmse1,rmse2,rmse3]
    mdiv_test=np.mean(m_test) #average disagreements among NNs 
    idx_test=np.argmin(m_test)
    pred_test_ensemble=pred_test[idx_test]
    error_test_ensemble=rmse(test_output[kb], pred_test_ensemble)
    saved_data_test[kb,:]=np.r_[test_output[kb],pred_test,idx_test,pred_test_ensemble,mdiv_test,error_test_ensemble]


loss_train_ensemble= (np.square(saved_data_train[:,0]-saved_data_train[:,5])).mean(axis=None)
loss_test_ensemble= (np.square(saved_data_test[:,0]-saved_data_test[:,5])).mean(axis=None)


print("final train loss:", loss_train_ensemble, "final test loss:", loss_test_ensemble) 
#pdb.set_trace()
with open('final_output_train_ensemble.txt','wb') as ff1:
    np.savetxt(ff1, saved_data_train)
with open('final_output_test_ensemble.txt','wb') as ff2:
    np.savetxt(ff2, saved_data_test)  
