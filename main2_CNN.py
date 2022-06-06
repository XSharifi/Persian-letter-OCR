import tensorflow as tf
import numpy as np
from CNN import CNN
from AutoEncoder import AutoEncoder
from ReadDataset import read
from ReadDataset import patitioning
from ReadDataset import splitData
from ReadDataset import splitLabel
np.set_printoptions(threshold=np.nan)



def createLayer(index, numOfOutput):

    training_data = np.array(Datasplit[index], dtype=np.float32)
    training_label = np.array(DatasplitLabel[index], dtype=np.float32)
    test_data = np.array(Datatest[index], dtype=np.float32)
    test_label = np.array(DatatestLabel[index], dtype=np.float32)

    cnn = CNN(numOfOutput=numOfOutput,learning_rate=0.01, epochs=40,batch_size=500)
    # output layer
    y_ = cnn.initial_mlp_network()
    with tf.device('/cpu:0'):
      training_label_onehot = sess.run(
        tf.one_hot(indices=training_label, depth=numOfOutput, dtype=np.float64))
      test_label_onehot = sess.run(
        tf.one_hot(indices=test_label, depth=numOfOutput, dtype=np.float64))

    cnn_accuricy, cnn_predict_label = cnn.calculate_session(y_, training_data, training_label_onehot, test_data,
                                                             test_label_onehot)
    print(cnn_accuricy)
    return cnn_accuricy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=config)

training_data, training_label = read()
test_data, test_label = read(dataset="testing")
training_label_split = patitioning(training_label)
test_label_split = patitioning(test_label)
# One hot encoding
training_label_onehot = sess.run(
    tf.one_hot(indices=training_label_split, depth=max(training_label_split + 1), dtype=np.float64))
test_label_onehot = sess.run(
    tf.one_hot(indices=test_label_split, depth=max(test_label_split + 1), dtype=np.float64))
cnn = CNN(numOfOutput=7, epochs=30,batch_size=500)
# output layer
y_ = cnn.initial_mlp_network()
cnn_accuricy, cnn_predict_label = cnn.calculate_session(y_, training_data, training_label_onehot, test_data,
                                                        test_label_onehot)

# split data and result for next layouts
Datasplit = splitData(training_data, training_label_split)
Datatest = splitData(test_data, cnn_predict_label)
DatasplitLabel = splitLabel(training_label, training_label_split)
DatatestLabel = splitLabel(test_label, cnn_predict_label)

print("Calculate layer2")
res1 = createLayer(0,4)
res2 = createLayer(1,6)
res3 = createLayer(2,6)
res4 = createLayer(3,4)
res5 = createLayer(4,2)
res6 = createLayer(5,2)
res7=createLayer(6,12)
print("step1 accuricy:",cnn_accuricy)
print(res1,res2,res3,res4,res5,res6,res7)