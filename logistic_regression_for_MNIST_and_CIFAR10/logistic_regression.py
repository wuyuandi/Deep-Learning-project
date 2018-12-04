import tensorflow as tf
import numpy as np


class DatasetIterator:
    def __init__(self, x, y, batch_size):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.b_sz = batch_size
        self.b_pt = 0
        self.d_sz = len(x)
        self.idx = None
        self.randomize()

    def randomize(self):
        self.idx = np.random.permutation(self.d_sz)
        self.b_pt = 0

    def next_batch(self):
        start = self.b_pt
        end = self.b_pt + self.b_sz
        idx = self.idx[start:end]
        x = self.x[idx]
        y = self.y[idx]

        self.b_pt += self.b_sz
        if self.b_pt >= self.d_sz:
            self.randomize()

        return x, y


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])



def logistic_regression(dataset_name, x_train, y_train, x_valid, y_valid, x_test):
  
    if dataset_name == "MNIST":
      image_size = 28 *28
      #tf Graph Input
      x = tf.placeholder(tf.float32, [None,784])
      y = tf.placeholder(tf.float32,[None,10])
      lam_val = 0.00001
        
    elif dataset_name == "CIFAR10":
      x = tf.placeholder(tf.float32)
      y = tf.placeholder(tf.float32)
      image_size = 32*32*3
      x_train = x_train.reshape(40000, 3*32*32)
      x_valid = x_valid.reshape(10000, 3*32*32)
      
      x_test = x_test.reshape(10000, 3*32*32)
      lam_val = 1
      
      #print(y_test.shape)
        
    #setup the parameter
    learning_rate = 0.01
    training_epochs = 100
    batch_size = 100
    display_step = 20
    y_train = one_hot(y_train,10)
    y_valid = one_hot(y_valid,10)    
    
    
    #create model inputs
    Xmean = np.mean(x_train,axis=0)
    Xm = x - Xmean
    b = tf.Variable(tf.zeros([10]))
    
    
    
    #weight
    w = tf.Variable(tf.zeros([image_size,10])) # parameter of the linear model
    
    logits= tf.matmul(Xm,w)+b # scores
    
    yp    = tf.nn.softmax(logits)
    #cost = loss + regulazation
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yp, y))
    if dataset_name == "CIFAR10":
      cost = -tf.reduce_mean(y * tf.log(yp+0.5),reduction_indices=1) + lam_val * tf.reduce_mean(tf.square(w))
      optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)    

    else:
      cost = -tf.reduce_mean(y * tf.log(yp),reduction_indices=1) + lam_val * tf.reduce_mean(tf.square(w))
        
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)    
   
    #initialize it and ready to start learning from the 
    init = tf.global_variables_initializer()
    y1 = tf.placeholder(tf.float32)
    y2 = tf.placeholder(tf.float32)
    acc = 100.0 * tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y1, axis=1), tf.argmax(y2, axis=1)), tf.float32))
    
    with tf.Session() as sess:
      sess.run(init)
      n_batchs = DatasetIterator(x_train, y_train, batch_size)
      total_batch = len(x_train) // batch_size
      for epoch in range(training_epochs):
        avg_cost = 0.
        print(epoch)
        for i in range(total_batch):
          [batch_x,batch_y] = n_batchs.next_batch()       
          #print(batch_x)
         
          #print(batch_x.shape)
          sess.run([optimizer,cost], feed_dict={x:batch_x, y:batch_y})
          
          
          
          #avg_cost += c / total_batch
        if epoch % display_step == 0:
          theta_value = w.eval()
          yp_train = yp.eval(feed_dict={x: x_train, w: theta_value})
          acc_train = acc.eval(feed_dict={y1: y_train, y2: yp_train})
          regerr_train = tf.reduce_mean(tf.square(yp_train - y_train)).eval()

                #prediction on validation set
          yp_validation = yp.eval(feed_dict={x: x_valid, w: theta_value})
          acc_validation = acc.eval(feed_dict={y1: y_valid, y2: yp_validation})
          regerr_validation = tf.reduce_mean(tf.square(yp_validation - y_valid)).eval()
          print(regerr_train, regerr_validation, acc_train, acc_validation)

      theta_value = w.eval()
      yp_test = tf.argmax(yp.eval(feed_dict= {x:x_test.reshape(len(x_test),image_size),w: theta_value}),1).eval()

     

      
      
    
    return yp_test

