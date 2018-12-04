

def AutoEncoder(input_tensor, is_training):

    n_inputs = 28*28
    n_hidden1 = 500
    n_hidden2 = 20
    n_hidden3 = n_hidden1
    n_outputs = n_inputs
    x_init = tf.contrib.layers.xavier_initializer()
    z_init = tf.zeros_initializer()
    
    with tf.name_scope("weights"):
      W1 = tf.get_variable(dtype=tf.float32,shape=(n_inputs,n_hidden1),initializer=x_init,name="W1")
      b1 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden1),initializer=z_init,name="b1")
      W2 = tf.get_variable(dtype=tf.float32,shape=(n_hidden1,n_hidden2),initializer=x_init,name="W2")
      b2 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden2),initializer=z_init,name="b2")
      W3 = tf.get_variable(shape=(n_hidden2,n_hidden3),initializer=x_init,name="W3")
      b3 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden3),initializer=z_init,name="b3")
      W4 = tf.get_variable(shape=(n_hidden3,n_outputs),initializer=x_init,name="W4")
      b4 = tf.get_variable(dtype=tf.float32,shape=(1,n_outputs),initializer=z_init,name="b4")
    with tf.name_scope("AE"):
    # encoding part
      hidden1 = tf.nn.elu(tf.matmul(input_tensor,W1)+b1)
      hidden1 = tf.contrib.layers.batch_norm(hidden1, decay=0.9, center=True,
                                         scale=True, epsilon=1e-8,
                                         updates_collections=None,
                                         is_training=is_training, scope=None)
      hidden2 = tf.nn.elu(tf.matmul(hidden1,W2)+b2)
      hidden2 = tf.contrib.layers.batch_norm(hidden2, decay=0.9, center=True,
                                         scale=True, epsilon=1e-8,
                                         updates_collections=None,
                                         is_training=is_training, scope=None)
    # decoding part
      hidden3 = tf.nn.elu(tf.matmul(hidden2,W3)+b3)
      recon = tf.nn.sigmoid(tf.matmul(hidden3,W4)+b4) # sigmoid limits output within [0,1]
      
      
    with tf.name_scope("fullyconnect"):
      fc = hidden2
      W5 = tf.get_variable(shape=(n_hidden2,10),initializer=x_init,name="W5")

      b5 = tf.get_variable(dtype=tf.float32,shape=(1,10),initializer=z_init,name="b5")
      fully = tf.matmul(fc,W5)+b5
      logits = tf.nn.softmax(fully)
    # return:
    # recon: reconstruction of the image by the autoencoder
    # logits: logits of the classification branch from the bottleneck of the autoencoder
    return recon, logits

def run():
    # General setup    
    EPOCHS = 10
    BATCH_SIZE = 64
    NUM_ITERS = int(55000 / BATCH_SIZE * EPOCHS)

    train_set = MNIST('train', batch_size=BATCH_SIZE)
    valid_set = MNIST('valid')
    test_set = MNIST('test', shuffle=False)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (None, 784))
    y = tf.placeholder(tf.int32, (None, 1))
    is_labeled = tf.placeholder(tf.float32, (None, 1))
    is_training = tf.placeholder(tf.bool, ())
    one_hot_y = tf.one_hot(y, 10)

    # create loss
    rate = 0.001
    recon, logits = AutoEncoder(x, is_training=is_training)
    prediction = tf.argmax(logits, axis=1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y) * is_labeled)
    recon_loss = tf.reduce_mean((recon - x) ** 2)
    loss_operation = cross_entropy + recon_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    def evaluation(images, true_labels):
        eval_batch_size = 100
        predicted_labels = []
        for start_index in range(0, len(images), eval_batch_size):
            end_index = start_index + eval_batch_size
            batch_x = images[start_index: end_index]
            batch_predicted_labels = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
            predicted_labels += list(batch_predicted_labels)
        predicted_labels = np.vstack(predicted_labels).flatten()
        true_labels = true_labels.flatten()
        accuracy = float((predicted_labels == true_labels).astype(np.int32).sum()) / len(images)
        return predicted_labels, accuracy

    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("Training...")
    for i in range(NUM_ITERS):
        batch_x, batch_y, batch_is_labeled = train_set.get_next_batch()
        _ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_labeled: batch_is_labeled, is_training: True})
        if (i + 1) % 1000 == 0 or i == NUM_ITERS - 1:
            _, validation_accuracy = evaluation(valid_set._images, valid_set._labels)
            print("Iter {}: Validation Accuracy = {:.3f}".format(i, validation_accuracy))

    print('Evaluating on test set')
    _, test_accuracy = evaluation(test_set._images, test_set._labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    sess.close()
    return test_accuracy


if __name__ == '__main__':
    run()