from keras import backend as K
from keras import initializers, regularizers
from keras.layers import Layer, Add


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
        print(input_shape[0][-1], input_shape[1][-1])
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
            print('OOOO GRAPH')
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
                 return_attention=False, n_classes=14268, label_revisions=None, encoder=None, **kwargs):

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)
        self.init = initializers.get('he_normal')
        self.supports_masking = True
        self.return_attention = return_attention
        self.n_classes = n_classes
        self.label_revisions = label_revisions
        self.encoder = encoder
        super(LabelWiseAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) == 2:
            input_shape = input_shape[0]

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

        if self.encoder == 'bert':
            self.add = Add()

        super(LabelWiseAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):

        if self.encoder == 'bert':
            x, cls_doc_rep = x

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

        if self.encoder == 'bert':
            cls_doc_rep = K.repeat_elements(K.expand_dims(cls_doc_rep, axis=1), rep=4271, axis=1)
            print(label_aware_doc_reprs.shape)
            label_aware_doc_reprs = self.add([label_aware_doc_reprs, cls_doc_rep]) / 2
            print(label_aware_doc_reprs.shape)

        if self.label_revisions is not None:
            for revisionist in self.revisionists:
                label_aware_doc_reprs = self.concat([label_aware_doc_reprs, self.Wa])
                label_aware_doc_reprs = revisionist(label_aware_doc_reprs)

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
