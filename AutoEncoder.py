import tensorflow as tf
import numpy as np
from ReadDataset import next_batch
# np.set_printoptions(threshold=np.nan)

from tensorflow.examples.tutorials.mnist import input_data

# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AutoEncoder(object):

    def __init__(self, learning_rate=0.1, epochs=10, epochautoencoder=5, batch_size=200, hiden_size1=150,hiden_size2=90,hiden_size3=60,
                 neorun_size=28 * 28, numOfOutput=36):

        self._learning_rate = learning_rate
        self._epochs = epochs
        self.epochautoencoder = epochautoencoder
        self._batch_size = batch_size
        self._hiden_size1 = hiden_size1
        self._hiden_size2 = hiden_size2
        self._hiden_size3 = hiden_size3
        self._input_size = neorun_size
        self._numOfOutput = numOfOutput
        # declare the training data placeholders
        # input x - for 28 x 28 pixels = 784
        self._X = tf.placeholder(tf.float32, [None, self._input_size], name='X')
        # now declare the output data placeholder - 35 digits
        self._Y = tf.placeholder(tf.float32, [None, self._numOfOutput], name='Y')
        self._weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self._input_size, self._hiden_size1], stddev=0.03)),
            'encoder_h2': tf.Variable(tf.random_normal([self._hiden_size1, self._hiden_size2], stddev=0.03)),
            'encoder_h3': tf.Variable(tf.random_normal([self._hiden_size2, self._hiden_size3], stddev=0.03)),
            'decoder_h1': tf.Variable(tf.random_normal([self._hiden_size1, self._input_size], stddev=0.03)),
            'decoder_h2': tf.Variable(tf.random_normal([self._hiden_size2, self._hiden_size1], stddev=0.03)),
            'decoder_h3': tf.Variable(tf.random_normal([self._hiden_size3, self._hiden_size2], stddev=0.03)),
        }
        self._biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self._hiden_size1], stddev=0.03)),
            'encoder_b2': tf.Variable(tf.random_normal([self._hiden_size2], stddev=0.03)),
            'encoder_b3': tf.Variable(tf.random_normal([self._hiden_size3], stddev=0.03)),
            'decoder_b1': tf.Variable(tf.random_normal([self._input_size], stddev=0.03)),
            'decoder_b2': tf.Variable(tf.random_normal([self._hiden_size1], stddev=0.03)),
            'decoder_b3': tf.Variable(tf.random_normal([self._hiden_size2], stddev=0.03)),
        }

    def initial_mlp_network(self):

        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([self._hiden_size3, self._numOfOutput], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([self._numOfOutput]), name='b2')
        return tf.nn.softmax(tf.add(tf.matmul(encoder_op, W2), b2))

    def initial_autoencode_network(self, encode_layer, encode_bias, decode_layer, decode_bias, input_decode):

        # Building the encoder
        def encoder(self, x):

            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self._weights[encode_layer]),
                                           self._biases[encode_bias]))

            return layer_1

        # Building the decoder
        def decoder(self, x):
            # Decoder Hidden layer with sigmoid activation #1
            layer_2 = tf.nn.relu(tf.add(tf.matmul(x, self._weights[decode_layer]),
                                           self._biases[decode_bias]))
            # Decoder Hidden layer with sigmoid activation #2
            # layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, self._weights['decoder_h2']),
            #                                self._biases['decoder_b2']))
            return layer_2
        global encoder_op
        # Construct model
        encoder_op = encoder(self, input_decode)
        decoder_op = decoder(self, encoder_op)
        return decoder_op, encoder_op

    def calculate_AutoEncoder(self, decoder_op, training_data, training_label, y_predict):
        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = y_predict

        # Define loss and optimizer, minimize the squared error
        cross_entropy = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        # add an optimiser
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(cross_entropy)
        optimizer = tf.train.RMSPropOptimizer(self._learning_rate).minimize(cross_entropy)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start Training
        # Start a new TF session
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)

            total_batch = int(len(training_label) / self._batch_size)
            for epoch in range(self.epochautoencoder):
                avg_cost = 0
                for i in range(total_batch):
                    # Prepare Data
                    # Get the next batch of MNIST data (only images are needed, not labels)
                    batch_x, _ = next_batch(self._batch_size, training_data, training_label)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cross_entropy], feed_dict={self._X: batch_x})
                    # Display logs per step
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

    def calculate_session(self, y_, training_data, training_label, test_data, test_label):
        # J=−1m∑i=1m∑j=1ny(i)jlog(yj_(i))+(1–y(i)j)log(1–yj_(i))
        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self._Y * tf.log(y_clipped)
                                                      + (1 - self._Y) * tf.log(1 - y_clipped), axis=1))

        # add an optimiser
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(cross_entropy)

        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(self._Y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # start the session
        # with tf.Session() as sess:
        # initialise the variables
        # start the session
        with tf.Session() as sess:
            # initialise the variables
            sess.run(init_op)
            total_batch = int(len(training_label) / self._batch_size)
            for epoch in range(self._epochs):
                avg_cost = 0
                for i in range(total_batch):
                    # batch_x, batch_y = next_batch(self._batch_size,training_data[i*self._batch_size:min(i*self._batch_size+self._batch_size,len(training_data))], training_label[i*self._batch_size:min(i*self._batch_size+self._batch_size,len(training_label))])
                    batch_x, batch_y = next_batch(self._batch_size, training_data, training_label)
                    # batch_x, batch_y = training_data[i*self._batch_size:min(i*self._batch_size+self._batch_size,len(training_data))], training_label[i*self._batch_size:min(i*self._batch_size+self._batch_size,len(training_label))]
                    _, c = sess.run([optimiser, cross_entropy], feed_dict={self._X: batch_x, self._Y: batch_y})
                    avg_cost += c / total_batch
                    my_accuracy = (sess.run(accuracy, feed_dict={self._X: batch_x, self._Y: batch_y}))
                    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost),
                      "accurity: ", my_accuracy)
                # print(batch_y.shape)
            my_accuracy = (sess.run(accuracy, feed_dict={self._X: test_data, self._Y: test_label}))
            my_predict_label = (sess.run(tf.argmax(y_, axis=1), feed_dict={self._X: test_data, self._Y: test_label}))
            return my_accuracy, my_predict_label
