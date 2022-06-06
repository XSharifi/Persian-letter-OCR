import os
import struct
import numpy as np
from scipy.misc import imresize

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    # print(idx)
    idx = idx[:num]
    # print(idx)
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def read(dataset="training", path="MNIST_data"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":

        # fname_img = os.path.join(path, 'Persian-Character-DB-Training.cdb')
        fname_lbl = os.path.join(path, 'Persian-Character-DB-Training.cdb')
        # fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        # fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        # fname_img = os.path.join(path, 'Persian-Character-DB-Test.cdb')
        fname_lbl = os.path.join(path, 'Persian-Character-DB-Test.cdb')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    MAX_COMMENT = 512
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:

        # magic, num,row,cols,e,d,f = struct.unpack(">IIIIIII", flbl.read(28))
        header = struct.unpack(">bbbbbbb", flbl.read(7))
        yy = struct.unpack("h", flbl.read(2))
        m = struct.unpack("b", flbl.read(1))
        d = struct.unpack("b", flbl.read(1))
        W = struct.unpack("h", flbl.read(2))[0]
        H = struct.unpack("h", flbl.read(2))[0]
        TotalRec = struct.unpack("i", flbl.read(4))[0]
        nMaxCount = struct.unpack("h", flbl.read(2))
        count = str(nMaxCount[0]) + "i"
        LaterCount = struct.unpack(count, flbl.read(nMaxCount[0] * 4))
        imgType = struct.unpack("b", flbl.read(1))[0]
        count = str(MAX_COMMENT) + "c"
        comments = struct.unpack(count, flbl.read(MAX_COMMENT * 1))
        reserved = struct.unpack("490b", flbl.read(490 * 1))

        if ((W > 0) & (H > 0)):
            normal = True
        else:
            normal = False
        Data = []
        Labels = []

        for i in range(0, TotalRec):
            startWord = struct.unpack("h", flbl.read(2))
            Labels.append(struct.unpack("h", flbl.read(2))[0])
            confidenc = struct.unpack("h", flbl.read(2))
            if (not (normal)):
                W = struct.unpack("h", flbl.read(2))[0]
                H = struct.unpack("h", flbl.read(2))[0]
            ByteCount = struct.unpack("h", flbl.read(2))
            Data.append(np.zeros(shape=(H, W), dtype=np.uint8))
            # Data.append([])
            if (imgType == 0):

                for y in range(0, H):
                    bWhite = True
                    counter = 0
                    while counter < W:
                        WBcount = struct.unpack("b", flbl.read(1))[0]
                        x = 0

                        while (x < WBcount):
                            # print(bWhite,i,y,x + counter)
                            if (bWhite):
                                Data[i][y][x + counter] = 0
                            else:
                                Data[i][y][x + counter] = 1
                            x += 1

                        bWhite = not (bWhite)
                        counter = counter + WBcount

                Data[i] = imresize(Data[i], (28, 28))


                Data[i] = np.reshape(Data[i], 28 * 28)

    Labels = np.array(Labels, dtype=np.float32)
    Data = np.array(Data, dtype=np.float32)

    return Data, Labels
    # get_img = lambda idx: [Labels[idx], Data[idx]]
    # #
    # # # Create an iterator which returns each image in turn
    # for i in range(len(Labels)):
    #     yield get_img

def maplabel(index):
    keyindex = {0: 'alef', 1: 'be', 2: 'pe', 3: 'te', 4: 'the', 5: 'jim', 6: 'che', 7: 'he', 8: 'khe', 9: 'dal', 10: 'zal', 11: 're', 12: 'ze', 13: 'zhe', 14: 'sin', 15: 'shin',
                16: 'sad',
                17: 'zad', 18: 'ta', 19: 'za', 20: 'ain', 21: 'ghain', 22: 'fe', 23: 'ghe', 24: 'kaf', 25: 'gaf', 26: 'lam', 27: 'mim', 28: 'noon', 29: 'vav', 30: 'ha', 31: 'ya',
                32: 'hamze', 33: 'alef-hat', 34: 'ha-bein', 35: 'ha-end'}
    return keyindex[index]
def getGroup(i):
    group = [[1, 2, 3, 4], [5, 6, 7, 8, 20, 21], [9, 10, 11, 12, 13, 29], [14, 15, 16, 17], [18, 19], [24, 25],
             [0, 22, 23, 26, 27, 28, 30, 31, 32, 33, 34, 35]]
    return group[i]


def keyvalueData(index):
    keyindex = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0, 10: 1, 11: 2, 12: 3, 13: 4, 14: 0, 15: 1,
                16: 2,
                17: 3, 18: 0, 19: 1, 20: 4, 21: 5, 22: 1, 23: 2, 24: 0, 25: 1, 26: 3, 27: 4, 28: 5, 29: 5, 30: 6, 31: 7,
                32: 8, 33: 9, 34: 10, 35: 11}

    return keyindex[index]


def splitLabel(labels_old, labels):
    Datalabel = [[] for _ in range(7)]
    for n, i in enumerate(labels):
        if i == 0:
            if labels_old[n] in getGroup(0):
                Datalabel[0].append(keyvalueData(labels_old[n]))
            else:
                Datalabel[0].append(100)
        elif i == 1:
            if labels_old[n] in getGroup(1):
                Datalabel[1].append(keyvalueData(labels_old[n]))
            else:
                Datalabel[1].append(100)
        elif i == 2:
            if labels_old[n] in getGroup(2):
                Datalabel[2].append(keyvalueData(labels_old[n]))
            else:
                Datalabel[2].append(100)
        elif i == 3:
            if labels_old[n] in getGroup(3):
                Datalabel[3].append(keyvalueData(labels_old[n]))
            else:
                Datalabel[3].append(100)
        elif i == 4:
            if labels_old[n] in getGroup(4):
                Datalabel[4].append(keyvalueData(labels_old[n]))
            else:
                Datalabel[4].append(100)
        elif i == 5:
            if labels_old[n] in getGroup(5):
                Datalabel[5].append(keyvalueData(labels_old[n]))
            else:
                Datalabel[5].append(100)
        elif i == 6:
            if labels_old[n] in getGroup(6):
                Datalabel[6].append(keyvalueData(labels_old[n]))
            else:
                Datalabel[6].append(100)

    return Datalabel


def splitData(data, labels):
    Dataitem = [[] for _ in range(7)]
    for n, i in enumerate(labels):
        if i == 0:
            Dataitem[0].append(data[n])
        elif i == 1:
            Dataitem[1].append(data[n])
        elif i == 2:
            Dataitem[2].append(data[n])
        elif i == 3:
            Dataitem[3].append(data[n])
        elif i == 4:
            Dataitem[4].append(data[n])
        elif i == 5:
            Dataitem[5].append(data[n])
        elif i == 6:
            Dataitem[6].append(data[n])

    return Dataitem


def patitioning(tuple):
    # second = open('second.txt', 'w')
    partitioned_data = []

    for n, i in enumerate(tuple):
        if i in getGroup(0):
            partitioned_data.append(0)
        elif i in getGroup(1):
            partitioned_data.append(1)
        elif i in getGroup(2):
            partitioned_data.append(2)
        elif i in getGroup(3):
            partitioned_data.append(3)
        elif i in getGroup(4):
            partitioned_data.append(4)
        elif i in getGroup(5):
            partitioned_data.append(5)
        elif i in getGroup(6):
            partitioned_data.append(6)
        else:
            print("invalid label")
    # for i in partitioned_data:
    #     second.write("%s\n" % str(i))
    return np.array(partitioned_data, dtype=np.float32)
