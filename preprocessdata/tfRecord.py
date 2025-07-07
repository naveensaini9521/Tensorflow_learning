import tensorflow as tf
import numpy as np
import IPython.display as display

def _bytes_features(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Features(bytes_list=tf.train.BytesList(value=[value]))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.IntList(value=[value]))

print(_bytes_features(b'test_string'))
print(_bytes_features(u'test_bytes'.encode('utf-8')))
print(_float_features(np.exp(1)))
print(_int64_features(True))
print(_int64_features(1))

feature = _float_feature(np.exp(1))

feature.SerializeToString()