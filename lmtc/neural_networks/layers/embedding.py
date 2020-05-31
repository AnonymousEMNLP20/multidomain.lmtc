from tensorflow.keras.layers import Layer, Embedding
from gensim.models import KeyedVectors
import numpy as np


class PretrainedEmbedding(Layer):
    def __init__(self, dims=200, **kwargs):
        self.dims= dims
        super(PretrainedEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        embeddings = KeyedVectors.load_word2vec_format(self.word_embedding_path, binary=True)
        word_encodings_weights = np.concatenate((np.zeros((1, embeddings.syn0.shape[-1]), dtype=np.float32), embeddings.syn0), axis=0)
        self.embeddings = Embedding(len(word_encodings_weights), word_encodings_weights.shape[-1],
                                    weights=[word_encodings_weights], trainable=False)

    def call(self, x, mask=None):
        return self.embeddings(x)

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[1], self.dims
