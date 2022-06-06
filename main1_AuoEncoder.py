import tensorflow as tf
import numpy as np
from CNN import CNN
from AutoEncoder import AutoEncoder
from ReadDataset import read

sess = tf.InteractiveSession()

training_data, training_label = read()
test_data, test_label = read(dataset="testing")

with tf.device('/cpu:0'):
  # One hot encoding
  training_label = sess.run(tf.one_hot(indices=training_label, depth=max(training_label + 1), dtype=np.float64))
  test_label = sess.run(tf.one_hot(indices=test_label, depth=max(test_label + 1), dtype=np.float64))

autoencoder = AutoEncoder()
decoder_op, layer_1 = autoencoder.initial_autoencode_network('encoder_h1', 'encoder_b1', 'decoder_h1', 'decoder_b1',autoencoder._X)
autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, autoencoder._X)
decoder_op, layer_2 = autoencoder.initial_autoencode_network('encoder_h2', 'encoder_b2', 'decoder_h2', 'decoder_b2',layer_1)
autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, layer_1)
decoder_op, layer_3 = autoencoder.initial_autoencode_network('encoder_h3', 'encoder_b3', 'decoder_h3', 'decoder_b3',layer_2)
autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, layer_2)

y_ = autoencoder.initial_mlp_network()
cnn_accuricy, cnn_predict_label = autoencoder.calculate_session(y_, training_data, training_label, test_data,
                                                                test_label)
print(cnn_accuricy)
