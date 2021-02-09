# -*- coding: utf-8 -*-
"""
Created on Mon Feb. 17 2020
this code is used for prediction the biomass.
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
valid_input = test_data[:,:9]
valid_output = test_data[:,9:]


model_path = "save"
input_dim=9
output_dim = 1
maxEpoch=100000
d_learning_rate=1e-4
dropout_rate=0.5
batch_size = 2000
#n_fc1 = 256
#fc1_dropout_rate = 0.5


he_init = tf.contrib.layers.variance_scaling_initializer()
_srng = np.random.RandomState(np.random.randint(1,2147462579))
                
# initial state
d_loss = 0.0
best_loss = np.infty
max_checks_without_progress = 1000
checks_without_progress = 0

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

                

#Train the model  
init=tf.global_variables_initializer()  #prepare an init node
saver = tf.train.Saver()                  


with tf.Session() as sess:
    
    sess.run(init) #actually initialize all the variables
    for epoch in range(maxEpoch): # range for python3
            
        for xtrain, ytrain in data_loader(train_input, train_output, batch_size, shuffle=True):
               
            #pdb.set_trace()
            _, Ld = sess.run([d_optimizer, d_loss], feed_dict={training: True, in_array: xtrain, out: ytrain})
        loss_train,pred_train = sess.run([d_loss,logits], feed_dict={in_array:train_input, out: train_output})
        if loss_train <= best_loss:
            my_path = os.path.join(model_path, "mymodel.ckpt")
            saver.save(sess, os.path.join(os.getcwd(), my_path))
#            with open('output_train.txt','wb') as ff1:
#                np.savetxt(ff1, np.concatenate((train_output,pred_train),axis=0))
            best_loss = loss_train
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        loss_test,pred_test =sess.run([d_loss,logits],feed_dict={in_array:valid_input, out: valid_output})
        with open('output_test.txt','wb') as ff2:
            np.savetxt(ff2, np.concatenate((valid_output,pred_test),axis=0))
        print(epoch, "Train loss:", loss_train, "Test loss:", loss_test,"Best loss",best_loss, "epoch",checks_without_progress)
        
            

with tf.Session() as sess:
    saver.restore(sess, model_path +"/mymodel.ckpt")
    loss_train = d_loss.eval(feed_dict={in_array:train_input, out: train_output})
    loss_test = d_loss.eval(feed_dict={in_array:valid_input, out: valid_output})  
    pred_train=logits.eval(feed_dict={in_array:train_input, out: train_output})
    pred_test=logits.eval(feed_dict={in_array:valid_input, out: valid_output}) 
    print("final train loss:", loss_train, "final test loss:", loss_test) 
    #pdb.set_trace()
    with open('output_train.txt','wb') as ff1:
        np.savetxt(ff1, np.concatenate((train_output,pred_train),axis=0))
    with open('output_test.txt','wb') as ff2:
        np.savetxt(ff2, np.concatenate((valid_output,pred_test),axis=0))       
