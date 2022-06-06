import tensorflow as tf
import numpy as np
from ReadDataset import next_batch


class CNN(object):
    def __init__(self, learning_rate=0.1, epochs=5, batch_size=200, hiden_size=300, neorun_size=28 * 28,
                 numOfOutput=36,layer_size=4,isSave=False,numInputChannel=1,filter_shape=[5,5], pool_shape=[2,2]):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._hiden_size = hiden_size
        self._input_size = neorun_size
        self._output_size = numOfOutput
        self._layer_size = layer_size
        self._numInputChannel=numInputChannel
        self._fiter_shape = filter_shape
        self._pool_shape = pool_shape
        # declare the training data placeholders
        # input x - for 28 x 28 pixels = 784
        self._X = tf.placeholder(tf.float32, [None, self._input_size], name='X')
        # now declare the output data placeholder - 35 digits
        self._Y = tf.placeholder(tf.float32, [None, self._output_size], name='Y')

        # now declare the weights connecting the input to the hidden layer
        self._W1 = tf.Variable(tf.random_normal([14*14*self._layer_size, self._hiden_size], stddev=0.03), name='W1')
        self._b1 = tf.Variable(tf.random_normal([self._hiden_size]), name='b1')
        # and the weights connecting the hidden layer to the output layer
        self._W2 = tf.Variable(tf.random_normal([self._hiden_size, self._output_size], stddev=0.03), name='W2')
        self._b2 = tf.Variable(tf.random_normal([self._output_size]), name='b2')
        self._isSave = isSave

        # setup the filter input shape for tf.nn.conv_2d
        conv_filt_shape = [filter_shape[0], filter_shape[1], self._numInputChannel,
                           self._layer_size]

        # initialise weights and bias for the filter
        self._convWeights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                   name="conv" + '_W')
        self._convBias = tf.Variable(tf.truncated_normal([self._layer_size]), name="conv" + '_b')



    def initial_mlp_network(self):
        print(self._W1)
        def create_new_conv_layer(self,input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):

            # setup the convolutional layer operation
            out_layer = tf.nn.conv2d(input_data, self._convWeights, [1, 1, 1, 1], padding='SAME')

            # add the bias
            out_layer += self._convBias

            # apply a ReLU non-linear activation
            out_layer = tf.nn.relu(out_layer)

            # now perform max pooling
            ksize = [1, pool_shape[0], pool_shape[1], 1]
            strides = [1, 2, 2, 1]
            out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                                       padding='SAME')

            return out_layer
        #
        x_shaped = tf.reshape(self._X, [-1, 28, 28, 1])
        # create some convolutional layers
        layer1 = create_new_conv_layer(self,x_shaped, self._numInputChannel,  self._layer_size , self._fiter_shape, self._pool_shape, name='layer1')
        # layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

        flattened = tf.reshape(layer1, [-1, 14 * 14 *  self._layer_size ])


        # calculate the output of the hidden layer
        hidden_out = tf.add(tf.matmul(flattened, self._W1), self._b1)
        hidden_out = tf.nn.relu(hidden_out)
        return tf.nn.softmax(tf.add(tf.matmul(hidden_out, self._W2), self._b2))

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
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        with tf.Session(config=config) as sess:
            # initialise the variables
            sess.run(init_op)
            total_batch = int(len(training_label) / self._batch_size)
            for epoch in range(self._epochs):
                avg_cost = 0
                for i in range(total_batch):
                    batch_x, batch_y = next_batch(self._batch_size, training_data, training_label)
                    #  batch_x, batch_y = training_data[i*batch_size:min(i*batch_size+batch_size,len(training_data))], training_label[i*batch_size:min(i*batch_size+batch_size,len(training_label))]

                    _, c = sess.run([optimiser, cross_entropy], feed_dict={self._X: batch_x, self._Y: batch_y})
                    avg_cost += c / total_batch
                    my_accuracy = (sess.run(accuracy, feed_dict={self._X: batch_x, self._Y: batch_y}))
                    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost),
                          "accurity: " ,my_accuracy)

            if(self._isSave):
             np.savetxt("W1.txt", sess.run(self._W1))
             np.savetxt("W2.txt", sess.run(self._W2))
             np.savetxt("B1.txt", sess.run(self._b1))
             np.savetxt("B2.txt", sess.run(self._b2))
             print(self._convWeights)
             valweight=tf.reshape(self._convWeights, (-1,self._layer_size*self._fiter_shape[0]*self._fiter_shape[1]))
             np.savetxt("convW.txt", sess.run(valweight))
             np.savetxt("convBias.txt", sess.run(self._convBias))
             # np.savetxt("predict.txt", my_predict_label)

            # with tf.device('/cpu:0'):
            my_accuracy = (sess.run(accuracy, feed_dict={self._X: test_data, self._Y: test_label}))
            # my_accuracytrain = (sess.run(accuracy, feed_dict={self._X: training_data, self._Y: training_label}))
            my_predict_label = (sess.run(tf.argmax(y_, axis=1), feed_dict={self._X: test_data, self._Y: test_label}))

            return my_accuracy,my_predict_label

    def initial_cnn_with_file(self,fileW1="W1.txt",fileW2="W2.txt",fileB1="B1.txt",fileB2="B2.txt",fileconWeight="convW.txt",fileconBias="convBias.txt",filepath="result/"):
        dataW1=np.loadtxt(filepath+fileW1)
        self._W1 = tf.Variable(np.array(dataW1),name='W1',dtype=tf.float32)
        dataW2 = np.loadtxt(filepath+fileW2)
        self._W2 = tf.Variable(np.array(dataW2),name='W2',dtype=tf.float32)
        datab1 = np.loadtxt(filepath+fileB1)
        self._b1 = tf.Variable(np.array(datab1),name='b1',dtype=tf.float32)
        datab2 = np.loadtxt(filepath+fileB2)
        self._b2 = tf.Variable(np.array(datab2),name='b2',dtype=tf.float32)
        datab2 = np.loadtxt(filepath+fileconWeight)
        self._convWeights = tf.Variable(np.array(datab2), name='convW', dtype=tf.float32)
        self._convWeights = tf.reshape(self._convWeights, (5,5,1,4))
        datab2 = np.loadtxt(filepath+fileconBias)
        self.__convBias = tf.Variable(np.array(datab2), name='convW', dtype=tf.float32)


    def calculate_test_data(self,sess,y_,test_data):
        my_predict_label = (sess.run(tf.argmax(y_, axis=1), feed_dict={self._X: test_data}))
        print(my_predict_label,"}")
        return my_predict_label
