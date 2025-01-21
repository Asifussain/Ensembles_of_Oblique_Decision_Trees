import tensorflow as tf
from functools import reduce


def tf_kron_prod(a, b):
    res = tf.einsum('ij,ik->ijk', a, b)
    res = tf.reshape(res, [-1, tf.reduce_prod(res.shape[1:])])
    return res


def tf_bin(x, cut_points, temperature=0.1):
    # x is a N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = tf.shape(cut_points)[0]
    weights = tf.reshape(tf.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points = tf.sort(cut_points)  # make sure cut_points is monotonically increasing
    b = tf.cumsum(tf.concat([tf.constant([0.0], dtype=tf.float32), tf.cast(cut_points, tf.float32) * -1], axis=0))
    h = tf.matmul(x, weights) + b
    res = tf.nn.softmax(tf.divide(h, temperature))
    return res


def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    leaf = reduce(lambda a, b: tf_kron_prod(a, b),
                  [tf_bin(x[:, z[0]:z[0] + 1], z[1], temperature) for z in enumerate(cut_points_list)])
    return tf.matmul(leaf, leaf_score)