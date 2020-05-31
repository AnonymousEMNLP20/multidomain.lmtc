from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 kernel_regularizer=None, bias_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.


        Note: The layer has been tested with Keras 1.x

        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...

        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_attention = return_attention
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='attention_W'.format(self.name))

        if self.bias:
            self.b = self.add_weight((1,),
                                     initializer='zeros',
                                     name='attention_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            eij *= K.cast(mask, K.floatx())

        # compute softmax
        a = K.expand_dims(K.softmax(eij, axis=-1))
        weighted_input = x * a
        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


class ContextualAttention(Layer):
    def __init__(self,
                 kernel_regularizer=None, u_regularizer=None, bias_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements a context-aware Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.


        Note: The layer has been tested with Keras 1.x

        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...

        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_attention = return_attention
        super(ContextualAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(ContextualAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        # Dot product with context vector U
        ait = dot_product(uit, self.u)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())

        # compute softmax
        a = K.expand_dims(K.softmax(ait, axis=-1))
        weighted_input = x * a
        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


class LabelDrivenAttention(Layer):

    def __init__(self, kernel_regularizer=None, bias_regularizer=None, return_attention=False, graph_op='concat', cutoff=None, **kwargs):

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)
        self.init = initializers.get('he_normal')
        self.supports_masking = True
        self.return_attention = return_attention
        self.graph_op = graph_op
        self.cutoff = cutoff
        super(LabelDrivenAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        assert input_shape[0][-1] == input_shape[1][-1]

        self.W_d = self.add_weight((input_shape[1][-1], input_shape[0][-1]),
                                   initializer=self.init,
                                   regularizer=self.W_regularizer,
                                   name='{}_Wd'.format(self.name))

        self.b_d = self.add_weight((input_shape[1][-1],),
                                   initializer='zeros',
                                   regularizer=self.b_regularizer,
                                   name='{}_bd'.format(self.name))

        if self.graph_op is not None:
            self.W_p = []
            self.W_c = []
            self.W_s = []
            self.b_g = []
            for i in range(2):
                self.W_p.append(self.add_weight((input_shape[1][-1], input_shape[1][-1]),
                                                initializer=self.init,
                                                regularizer=self.W_regularizer,
                                                name='{}_Wp{}'.format(self.name, i + 1)))

                self.W_c.append(self.add_weight((input_shape[1][-1], input_shape[1][-1]),
                                                initializer=self.init,
                                                regularizer=self.W_regularizer,
                                                name='{}_Wc{}'.format(self.name, i + 1)))

                self.W_s.append(self.add_weight((input_shape[1][-1], input_shape[1][-1]),
                                                initializer=self.init,
                                                regularizer=self.W_regularizer,
                                                name='{}_Ws{}'.format(self.name, i + 1)))

                self.b_g.append(self.add_weight((input_shape[1][-1],),
                                                initializer='zeros',
                                                regularizer=self.b_regularizer,
                                                name='{}_bg{}'.format(self.name, i + 1)))

            if self.graph_op == 'concat':
                self.W_o = self.add_weight((input_shape[1][-1]*2, input_shape[1][-1]),
                                           initializer=self.init,
                                           regularizer=self.W_regularizer,
                                           name='{}_Wo'.format(self.name))
                self.b_o = self.add_weight((input_shape[1][-1]*2,),
                                           initializer='zeros',
                                           regularizer=self.b_regularizer,
                                           name='{}_bo'.format(self.name))

        super(LabelDrivenAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):
        # Unfold inputs (document representations, label representations, labels' graph)
        if len(x) == 3:
            doc_reps, label_reps, labels_graph = x
        else:
            doc_reps, label_reps = x
            labels_graph = None

        doc2_reps = K.tanh(dot_product(doc_reps, self.W_d) + self.b_d)

        # Compute Attention Scores
        doc_a = dot_product(doc2_reps, label_reps[:self.cutoff])

        def label_wise_attention(values):
            doc_repi, ai = values
            ai = K.softmax(K.transpose(ai))
            label_aware_doc_rep = K.dot(ai, doc_repi)
            if self.return_attention:
                return [label_aware_doc_rep, ai]
            else:
                return [label_aware_doc_rep, label_aware_doc_rep]

        label_aware_doc_reprs, attention_scores = K.tf.map_fn(label_wise_attention, [doc_reps, doc_a])

        if labels_graph is not None:
            # 2-level Graph Convolution
            label_reps_p = dot_product(label_reps, self.W_p[0])
            label_reps_c = dot_product(label_reps, self.W_c[0])
            label_reps_s = dot_product(label_reps, self.W_s[0])
            graph_h_p = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_p, [1, 0]), labels_graph), [1, 0])
            graph_h_c = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_c, [1, 0]), K.permute_dimensions(labels_graph, [1, 0])), [1, 0])
            parents_sums = K.expand_dims(K.sum(labels_graph, axis=0), axis=-1)
            parents_sums = K.tf.where(K.tf.equal(0.0, parents_sums), K.tf.ones_like(parents_sums), parents_sums)
            children_sums = K.expand_dims(K.sum(K.permute_dimensions(labels_graph, [1, 0]), axis=0), axis=-1)
            children_sums = K.tf.where(K.tf.equal(0.0, children_sums), K.tf.ones_like(children_sums), children_sums)
            graph_h_p = graph_h_p / parents_sums
            graph_h_c = graph_h_c / children_sums
            label_reps_g = K.tanh(label_reps_s + graph_h_p + graph_h_c + self.b_g[0])

            label_reps_p = dot_product(label_reps_g, self.W_p[1])
            label_reps_c = dot_product(label_reps_g, self.W_c[1])
            label_reps_s = dot_product(label_reps_g, self.W_s[1])
            graph_h_p = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_p, [1, 0]), labels_graph), [1, 0])
            graph_h_c = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_c, [1, 0]), K.permute_dimensions(labels_graph, [1, 0])), [1, 0])
            graph_h_p = graph_h_p / parents_sums
            graph_h_c = graph_h_c / children_sums
            label_reps_g = K.tanh(label_reps_s + graph_h_p + graph_h_c + self.b_g[1])

            # Combine label embeddings + graph-aware label embeddings
            if self.graph_op == 'concat':
                label_reps = K.concatenate([label_reps, label_reps_g], axis=-1)
                label_aware_doc_reprs = K.tanh(dot_product(label_aware_doc_reprs, self.W_o) + self.b_o)
            else:
                label_reps = label_reps + label_reps_g

        # Compute label-scores
        label_aware_doc_reprs = K.sum(label_aware_doc_reprs * label_reps[:self.cutoff], axis=-1)
        label_aware_doc_reprs = K.sigmoid(label_aware_doc_reprs)

        if self.return_attention:
            return [label_aware_doc_reprs, attention_scores]

        return label_aware_doc_reprs

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0][0], self.cutoff),
                    (input_shape[0][0], input_shape[1][0], input_shape[0][1])]
        return input_shape[0][0], self.cutoff


class LabelWiseAttention(Layer):

    def __init__(self, kernel_regularizer=None, bias_regularizer=None,
                 return_attention=False, n_classes=14268, **kwargs):

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)
        self.init = initializers.get('he_normal')
        self.supports_masking = True
        self.return_attention = return_attention
        self.n_classes = n_classes
        super(LabelWiseAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.Wa = self.add_weight((self.n_classes, input_shape[-1]),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  name='{}_Wa'.format(self.name))

        self.Wo = self.add_weight((self.n_classes, input_shape[-1]),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  name='{}_Wo'.format(self.name))

        self.bo = self.add_weight((self.n_classes,),
                                  initializer='zeros',
                                  regularizer=self.b_regularizer,
                                  name='{}_bo'.format(self.name))

        super(LabelWiseAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):

        a = dot_product(x, self.Wa)

        def label_wise_attention(values):
            doc_repi, ai = values
            ai = K.softmax(K.transpose(ai))
            label_aware_doc_rep = K.dot(ai, doc_repi)
            if self.return_attention:
                return [label_aware_doc_rep, ai]
            else:
                return [label_aware_doc_rep, label_aware_doc_rep]

        label_aware_doc_reprs, attention_scores = K.tf.map_fn(label_wise_attention, [x, a])

        # Compute label-scores
        label_aware_doc_reprs = K.sum(label_aware_doc_reprs * self.Wo, axis=-1) + self.bo
        label_aware_doc_reprs = K.sigmoid(label_aware_doc_reprs)

        if self.return_attention:
            return [label_aware_doc_reprs, attention_scores]

        return label_aware_doc_reprs

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], self.n_classes),
                    (input_shape[0], input_shape[1], self.n_classes, input_shape[-1])]
        return input_shape[0], self.n_classes


class MultiHeadSelfAttention(Layer):

    def __init__(self, n_heads: int, units: int, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.units = units
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        x = input_shape
        return x[0], x[1], x[2] // 3

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs,  mask=None):
        def shape_list(x):
            tmp = K.int_shape(x)
            tmp = list(tmp)
            tmp[0] = -1
            tmp[1] = tf.shape(x)[1]
            return tmp

        def split_heads(x, n_heads: int, k: bool = False):
            x_shape = shape_list(x)
            m = x_shape[-1]
            new_x_shape = x_shape[:-1] + [n_heads, m // n_heads]
            new_x = K.reshape(x, new_x_shape)
            return K.permute_dimensions(new_x, [0, 2, 3, 1] if k else [0, 2, 1, 3])

        def merge_heads(x):
            new_x = K.permute_dimensions(x, [0, 2, 1, 3])
            x_shape = shape_list(new_x)
            new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
            return K.reshape(new_x, new_x_shape)

        def scaled_dot_product_attention(q, k, v):
            # Attention(Q, K, V) = (softmax(Q*K_T} / sqrt(d_k)) * V
            w = K.batch_dot(q, k)
            w = w / K.sqrt(K.cast(shape_list(v)[-1], K.floatx()))
            # Apply mask
            if mask is not None:
                w = K.cast(mask, K.floatx()) * w + (1.0 - K.cast(mask, K.floatx())) * -1e9
            w = K.softmax(w)
            return K.batch_dot(w, v)

        # Split Q, V, A representations
        _q, _k, _v = inputs[:, :, :self.units], inputs[:, :, self.units:2 * self.units], inputs[:, :, -self.units:]

        # Split heads for Q, V, A representations
        q = split_heads(_q, self.n_heads)  # Queries
        k = split_heads(_k, self.n_heads, k=True)  # Keys
        v = split_heads(_v, self.n_heads)  # Values

        ia = scaled_dot_product_attention(q, k, v)

        return merge_heads(ia)

    def get_config(self):
        config = {
            'n_heads': self.n_heads,
            'units': self.units
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

