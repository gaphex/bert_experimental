import tensorflow as tf
import numpy as np


class L2Retriever:
    def __init__(self, dim, top_k=3, use_norm=False, use_gpu=True):
        self.dim = dim
        self.top_k = top_k
        self.use_norm = use_norm
        config = tf.ConfigProto(
            device_count={'GPU': (1 if use_gpu else 0)}
        )
        self.session = tf.Session(config=config)
        
        self.dtype = "float32"
        
        self.query = tf.placeholder(self.dtype, [self.dim])
        self.kbase = tf.placeholder(self.dtype, [None, self.dim])
        if self.use_norm:
            self.norm = tf.placeholder(self.dtype, [None, 1])
        else:
            self.norm = None
        
        self.build_graph()

    def build_graph(self):

        distance = self.euclidean_distances(self.kbase, self.query, self.norm)
        top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=self.top_k)
        top_dists = tf.negative(top_neg_dists)

        self.top_distances = top_dists
        self.top_indices = top_indices

    def predict(self, kbase, query, norm=None):
        query = np.squeeze(query)
        feed_dict = {self.query: query, self.kbase: kbase}
        if self.use_norm:
          feed_dict[self.norm] = norm
        
        I, D = self.session.run([self.top_indices, self.top_distances],
                                feed_dict=feed_dict)
        
        return I, D
      
    @staticmethod
    def euclidean_distances(kbase, query, norm=None):
        query = tf.reshape(query, (1, -1))

        if norm is None:
          XX = tf.keras.backend.batch_dot(kbase, kbase, axes=1)
        else:
          XX = norm
        YY = tf.matmul(query, tf.transpose(query))
        XY = tf.matmul(kbase, tf.transpose(query))

        distance = XX - 2 * XY + YY
        distance = tf.sqrt(tf.reshape(distance, (-1,)))

        return distance
    
    @staticmethod
    def compute_squared_l2_norm(mat):
        square_norms = np.sum(mat**2, axis=1, keepdims=True)
        return square_norms
      
    def __call__(self, kbase, query, norm=None):
        return self.predict(kbase, query, norm)
