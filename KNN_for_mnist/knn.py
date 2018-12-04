import tensorflow as tf
import numpy as np


def knn(x_train, y_train, x_test):

    predicted_y_test = []
    k = 3
    xtrain_tensor = tf.placeholder(tf.float32, [None, 784])
    xtest_tensor = tf.placeholder(tf.float32, [784])    
    
    #L1-norm to calculate distance
    #distance = tf.reduce_sum(tf.abs(tf.add(xtrain_tensor, tf.negative(xte))), reduction_indices=0)
    #L2 to calculate Euclidean Distance
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(xtrain_tensor,tf.negative(xtest_tensor)),2),reduction_indices=1))
    
    #get the min index for k =1 
    pred = tf.argmin(distance,0)
    
    #get the min idex for k more that 1
    
    #pred = tf.nn.top_k(-distance,k,sorted=True)
    
    #initialize it and ready to start learning from the 
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
      sess.run(init)
      #top_k = np.zeros(k)
      for i in range(len(x_test)):
        #每次循环feed数据，候选Xtr全部，测试集Xte一次循环输入一条
        index = sess.run(pred, feed_dict={xtrain_tensor: x_train, xtest_tensor: x_test[i,:]})
        predicted_y_test.append(y_train[index])
        #top_k = []
#         for j in index[1]:
#           top_k.append(y_train[j])
#           print(np.argmax(np.bincount(top_k)))
#           predicted_y_test.append(np.argmax(np.bincount(top_k)))

        
        
    return predicted_y_test