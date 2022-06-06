import tensorflow as tf
import numpy as np
from CNN import CNN
from AutoEncoder import AutoEncoder
from ReadDataset import read
from ReadDataset import patitioning
from ReadDataset import splitData
from ReadDataset import splitLabel

np.set_printoptions(threshold=np.nan)


def one_hot_labeles(training_label, depth_trainlabel, test_label_onehot, depth_testlabel):
    # One hot encoding
    training_label_onehot = sess.run(
        tf.one_hot(indices=training_label, depth=depth_trainlabel, dtype=np.float64))
    test_label_onehot = sess.run(
        tf.one_hot(indices=test_label_onehot, depth=depth_testlabel, dtype=np.float64))
    return training_label_onehot, test_label_onehot


def createLayer(index, numOfOutput):
    training_data = np.array(Datasplit[index], dtype=np.float32)
    training_label = np.array(DatasplitLabel[index], dtype=np.float32)
    test_data = np.array(Datatest[index], dtype=np.float32)
    test_label = np.array(DatatestLabel[index], dtype=np.float32)

    autoencoder = AutoEncoder(numOfOutput=numOfOutput, epochs=10)
    decoder_op, layer_1 = autoencoder.initial_autoencode_network('encoder_h1', 'encoder_b1', 'decoder_h1', 'decoder_b1',
                                                                 autoencoder._X)
    autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, autoencoder._X)
    decoder_op, layer_2 = autoencoder.initial_autoencode_network('encoder_h2', 'encoder_b2', 'decoder_h2', 'decoder_b2',
                                                                 layer_1)
    autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, layer_1)
    decoder_op, layer_3 = autoencoder.initial_autoencode_network('encoder_h3', 'encoder_b3', 'decoder_h3', 'decoder_b3',
                                                                 layer_2)
    autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, layer_2)
    y_ = autoencoder.initial_mlp_network()
    with tf.device('/cpu:0'):
      training_label_onehot = sess.run(
            tf.one_hot(indices=training_label, depth=numOfOutput, dtype=np.float64))
      test_label_onehot = sess.run(
            tf.one_hot(indices=test_label, depth=numOfOutput, dtype=np.float64))
    cnn_accuricy, cnn_predict_label = autoencoder.calculate_session(y_, training_data, training_label_onehot, test_data,
                                                             test_label_onehot)
    print(cnn_accuricy)
    return cnn_accuricy

sess = tf.InteractiveSession()

training_data, training_label = read()
test_data, test_label = read(dataset="testing")

training_label_split = patitioning(training_label)
test_label_split = patitioning(test_label)

with tf.device('/cpu:0'):
 training_label_onehot, test_label_onehot = one_hot_labeles(training_label_split, max(training_label_split + 1),
                                                           test_label_split, max(test_label_split + 1))

autoencoder = AutoEncoder(numOfOutput=7, epochs=15)
decoder_op, layer_1 = autoencoder.initial_autoencode_network('encoder_h1', 'encoder_b1', 'decoder_h1', 'decoder_b1',autoencoder._X)
autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, autoencoder._X)
decoder_op, layer_2 = autoencoder.initial_autoencode_network('encoder_h2', 'encoder_b2', 'decoder_h2', 'decoder_b2',layer_1)
autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, layer_1)
decoder_op, layer_3 = autoencoder.initial_autoencode_network('encoder_h3', 'encoder_b3', 'decoder_h3', 'decoder_b3',layer_2)
autoencoder.calculate_AutoEncoder(decoder_op, training_data, training_label, layer_2)

y_ = autoencoder.initial_mlp_network()
autoencoder_accuricy, autoencoder_predict_label  = autoencoder.calculate_session(y_, training_data, training_label_onehot, test_data, test_label_onehot)
print(autoencoder_accuricy)
# split data and result for next layouts
Datasplit = splitData(training_data, training_label_split)
Datatest = splitData(test_data, autoencoder_predict_label)
DatasplitLabel = splitLabel(training_label, training_label_split)
DatatestLabel = splitLabel(test_label, autoencoder_predict_label)

print("Calculate layer2")
res1 = createLayer(0,4)
res2 = createLayer(1,6)
res3 = createLayer(2,6)
res4 = createLayer(3,4)
res5 = createLayer(4,2)
res6 = createLayer(5,2)
res7=createLayer(6, 12)

print(res1,res2,res3,res4,res5,res6,res7)
