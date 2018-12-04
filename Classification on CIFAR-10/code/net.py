from ops import *
import timeit
from cifar10 import Cifar10

"""net part"""
from tensorflow.contrib.layers import flatten
def weight_variable(shape,name,stddev):
    initial = tf.truncated_normal(shape, mean=0,stddev=stddev)
    return tf.Variable(initial,name= name)
def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

def net(input, is_training, dropout_kept_prob):

  
  
  # layer 1 
  w_conv1 = weight_variable([5, 5, 3, 32],'w_conv1',0.1)
  b_conv1 = bias_variable(32)
  w_conv1 = tf.get_variable('w_conv1', shape = [5,5,3,32], initializer = tf.contrib.layers.xavier_initializer())
  conv1 = tf.nn.conv2d(input, w_conv1, strides = [1,1,1,1],padding = 'SAME') + b_conv1 
  #conv1 = tf.layers.batch_normalization(conv1, center=True,scale=True,training=True)
  conv1 = tf.nn.relu(conv1)
  
  conv1 = tf.nn.dropout(conv1,dropout_kept_prob)  
  #conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
  
  
  
  #layer 2   
  w_conv2 = weight_variable([5,5,32,32],'w_conv2',0.1) 
  b_conv2 = bias_variable(32)
  conv2 = tf.nn.conv2d(conv1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2 
  conv2 = tf.layers.batch_normalization(conv2, center=True,scale=True,training=True)
  #conv2 = tf.nn.relu(conv2)
 
  conv2 = tf.nn.dropout(conv2,dropout_kept_prob)
  
  conv2 = tf.concat([conv1,conv2],3)
  #conv2 = conv1 + conv2
  conv2 = tf.nn.relu(conv2)

  conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
  
  #layer3
  w_conv3 = weight_variable([7,7,64,128],'w_conv3',0.1) 
  b_conv3 = bias_variable(128)
  conv3 = tf.nn.conv2d(conv2, w_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3 
  conv3 = tf.layers.batch_normalization(conv3, center=True,scale=True,training=True)
  conv3 = tf.nn.relu(conv3)
  conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
  
  #layer4
  w_conv4 = weight_variable([5,5,128,256],'w_conv4',0.1) 
  b_conv4 = bias_variable(256)
  conv4 = tf.nn.conv2d(conv3, w_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4 
  conv4 = tf.layers.batch_normalization(conv4, center=True,scale=True,training=True)
  
  conv4 = tf.nn.dropout(conv4,dropout_kept_prob)
  conv4 = tf.nn.relu(conv4)
  conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
    
  #layer5
#   w_conv5 = weight_variable([3,3,256,512],'w_conv5',0.1) 
#   b_conv5 = bias_variable(512)
#   conv5 = tf.nn.conv2d(conv4, w_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5 
#   conv5 = tf.layers.batch_normalization(conv5, center=True,scale=True,training=True)
#   conv5 = tf.nn.relu(conv5)
#   conv5 = tf.nn.max_pool(conv5, ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID') 
#   conv5 = tf.nn.dropout(conv5,dropout_kept_prob) 
  #layer6
#   w_conv6 = weight_variable([5,5,512,512],'w_conv6',0.1) 
#   b_conv6 = bias_variable(512)
#   conv6 = tf.nn.conv2d(conv5, w_conv6, strides=[1,1,1,1], padding='SAME') + b_conv6 
#   conv6 = tf.layers.batch_normalization(conv6, center=True,scale=True,training=True)
#   conv6 = tf.nn.relu(conv6)
#   #conv6 = tf.nn.dropout(conv6,0.8)  
#   conv6 = tf.nn.max_pool(conv6, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 
  
  
  #layer 7 fully connect 1   
  fc0 = flatten(conv4)
  # Dropout with probability 0.5
  DropMask = (tf.to_float(tf.random_uniform((1,4096))<dropout_kept_prob))/dropout_kept_prob
  fc0 = tf.cond(is_training, lambda: fc0*DropMask, lambda: fc0)
  fc1_w = weight_variable([4096,512],'fc1_w',0.1)
  fc1_b = bias_variable(512)
  fc1 = tf.matmul(fc0,fc1_w) + fc1_b
  fc1 = tf.layers.batch_normalization(fc1, center=True,scale=True,training=True)

  fc1 = tf.nn.relu(fc1)

  
  
  #layer 8 fully connect 2 
  fc2_w = weight_variable([512,84],'fc2_w',0.1)
  fc2_b = bias_variable(84)
  fc2 = tf.matmul(fc1,fc2_w) + fc2_b
  fc2 = tf.layers.batch_normalization(fc2, center=True,scale=True,training=True)
  fc2 = tf.nn.relu(fc2)

  
  #layer 9 fully connect 3
  fc3_w = weight_variable([84,10],'fc3_w',0.1)
  fc3_b = bias_variable(10)
  logits = tf.matmul(fc2,fc3_w) + fc3_b
  
  return logits
  
  
  
  
def train():
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # TODO: Write your training code here
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  # - Construct your model
  # - Create loss and training op
  # - Run training
  # AS IT WILL TAKE VERY LONG ON CIFAR10 DATASET TO TRAIN
  # YOU SHOULD USE tf.train.Saver() TO SAVE YOUR MODEL AFTER TRAINING
  # AT TEST TIME, LOAD THE MODEL AND RUN TEST ON THE TEST SET
  x = tf.placeholder(tf.float32,[None,32,32,3])
  y = tf.placeholder(tf.float32,(None))
  #one_hot_y = tf.one_hot(y, 10)
  
  cifar10_train = Cifar10(batch_size = 100, one_hot = True, test = False, shuffle = True)
  cifar10_test = Cifar10(batch_size = 100, one_hot = False, test = True, shuffle = False)
  cifar10_test_images, cifar10_test_labels = cifar10_test.images, cifar10_test.labels
  
  learning_rate = 0.001
  logits = net(x,tf.constant(True,dtype=tf.bool),0.6)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels = y)
  loss_operation = tf.reduce_mean(cross_entropy)
  
  optimizer = tf.train.AdamOptimizer(learning_rate)
  
  grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation = optimizer.apply_gradients(grads_and_vars)
  
  
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/histogram", var)

  add_gradient_summaries(grads_and_vars)
  tf.summary.scalar('loss_operation', loss_operation)
  merged_summary_op= tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('logs/')
  
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y,1))
  accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
#   var_list = tf.trainable_variables()
#   g_list = tf.global_variables()
#   bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
#   bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
#   var_list += bn_moving_vars
#   saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
  
  saver = tf.train.Saver(max_to_keep = 5)
  n_epoch = 80
  n_batch = 100
  
  global_step = 0
  with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
      
      for batch in range(n_batch):
        batch_x, batch_y = cifar10_train.get_next_batch() # get the next batch of Cifar10 train set
        _, summaries = sess.run([training_operation, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
        if global_step % 100 == 0:
          _loss = sess.run(loss_operation, feed_dict = {x:batch_x, y:batch_y})
          print("loss: ", _loss, "Step: ", global_step)
        global_step += 1
      
      
      training_accuracy = evaluate(batch_x, batch_y, accuracy_operation, x, y)
      print("epoch: ", epoch)
      print("accuracy_rate: ", training_accuracy * 100, " % ")
      print()
      saver.save(sess, 'ckpt/netCheckpoint', global_step = epoch)
      
    
    
  
  
  
def test(cifar10_test_images):
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()

  # LOAD THE MODEL AND RUN TEST ON THE TEST SET
  x = tf.placeholder(tf.float32,[None,32,32,3])
  
  logits = net(x,tf.constant(False,dtype=tf.bool),1.0)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('ckpt'))
    test = sess.run(logits, feed_dict={x: cifar10_test_images})
    pred_ytest = np.argmax(test,1)
  
  
  return pred_ytest


