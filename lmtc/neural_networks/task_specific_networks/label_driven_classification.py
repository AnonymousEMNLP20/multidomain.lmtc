import os

import numpy as np
from gensim.models import KeyedVectors
from keras import backend as K
from keras.layers import Bidirectional
from keras.layers import Dense, Conv1D, Activation, Embedding
from keras.layers import GRU, CuDNNGRU, add, concatenate
from keras.layers import Input, SpatialDropout1D, Dropout
from keras.models import Model
from keras.optimizers import Adam

from lmtc.data import VECTORS_DIR
from lmtc.experiments.configurations.configuration import Configuration
from lmtc.neural_networks.layers import Camouflage, GlobalMeanPooling1D, TimestepDropout, SymmetricMasking, \
    ElmoEmbeddingLayer, LabelDrivenAttention, LabelWiseAttention, BERT
from lmtc.neural_networks.neural_network import NeuralNetwork
from lmtc.neural_networks.utilities import AccumulatedAdam


class LabelDrivenClassification(NeuralNetwork):
    def __init__(self, label_terms_ids, labels_graph, true_labels_cutoff):
        super().__init__()
        self._conf = 'NONE'
        self._cuDNN = Configuration['task']['cuDNN']
        self._decision_type = Configuration['task']['decision_type']
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.label_encoder = Configuration['model']['label_encoder']
        self.token_encoder = Configuration['model']['document_encoder']
        self.word_embedding_path = os.path.join(VECTORS_DIR, Configuration['model']['embeddings'])
        self.label_terms_ids = label_terms_ids
        self.true_labels_cutoff = true_labels_cutoff
        if 'graph' in Configuration['model']['architecture'].lower():
            self.labels_graph = labels_graph
        else:
            self.labels_graph = None

    def __del__(self):
        K.clear_session()
        del self._model

    def compile(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        K.clear_session()

        if Configuration['model']['hierarchical']:
            self._compile_hierarchical_label_wise_attention(n_hidden_layers=n_hidden_layers,
                                                            hidden_units_size=hidden_units_size,
                                                            dropout_rate=dropout_rate,
                                                            word_dropout_rate=word_dropout_rate, lr=lr)

        elif self.label_encoder:
            self._compile_label_wise_attention_zero(n_hidden_layers=n_hidden_layers,
                                                    hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                                    word_dropout_rate=word_dropout_rate, lr=lr)
        else:
            self._compile_label_wise_attention(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                                               dropout_rate=dropout_rate,
                                               word_dropout_rate=word_dropout_rate, lr=lr)

    def PretrainedEmbedding(self):

        inputs = Input(shape=(None,), dtype='int32')
        embeddings = KeyedVectors.load_word2vec_format(self.word_embedding_path, binary=False)
        word_embeddings_weights = K.cast_to_floatx(
            np.concatenate((np.zeros((1, embeddings.syn0.shape[-1]), dtype=np.float32), embeddings.syn0), axis=0))
        embeds = Embedding(len(word_embeddings_weights), word_embeddings_weights.shape[-1],
                           weights=[word_embeddings_weights], trainable=False)(inputs)

        return Model(inputs=inputs, outputs=embeds, name='embedding')

    def TokenEncoder(self, inputs, encoder, dropout_rate, word_dropout_rate, hidden_layers, hidden_units_size):

        # Apply variational drop-out
        if encoder != 'bert':
            inner_inputs = SpatialDropout1D(dropout_rate)(inputs)
            inner_inputs = TimestepDropout(word_dropout_rate)(inner_inputs)

        if encoder == 'grus':
            # Bi-GRUs over token embeddings
            for i in range(hidden_layers):
                if self._cuDNN:
                    bi_grus = Bidirectional(CuDNNGRU(units=hidden_units_size, return_sequences=True))(inner_inputs)
                else:
                    bi_grus = Bidirectional(GRU(units=hidden_units_size, return_sequences=True, activation="tanh",
                                                recurrent_activation='sigmoid'))(inner_inputs)
                bi_grus = Camouflage(mask_value=0)(inputs=[bi_grus, inputs])

                if i == 0:
                    inner_inputs = SpatialDropout1D(dropout_rate)(bi_grus)
                else:
                    inner_inputs = add([bi_grus, inner_inputs])
                    inner_inputs = SpatialDropout1D(dropout_rate)(inner_inputs)

            outputs = Camouflage()([inner_inputs, inputs])
        elif encoder == 'cnns':
            # CNNs over token embeddings
            for i in range(hidden_layers):
                convs = Conv1D(filters=hidden_units_size, kernel_size=3, strides=1, padding="same")(inner_inputs)
                convs = Activation('tanh')(convs)
                convs = SpatialDropout1D(dropout_rate)(convs)
                convs = Camouflage(mask_value=0)(inputs=[convs, inputs])
                inner_inputs = SpatialDropout1D(dropout_rate)(convs)

            outputs = Camouflage()([inner_inputs, inputs])
        elif encoder == 'bert':
            dropout = Dropout(dropout_rate)
            bert_cls, bert_encodings = BERT(output_representation='all')(inputs[0])
            bert_cls = dropout(bert_cls)
            bert_encodings = dropout(bert_encodings)
            bert_encodings = Camouflage(mask_value=0)(inputs=[bert_encodings, inputs[1]])
            outputs = [bert_encodings, bert_cls]

        return outputs

    def LabelEncoder(self, inputs, encoder, dropout_rate, hidden_units_size):

        # Apply variational drop-out
        if encoder not in ['node2vec', 'word2vec+node2vec']:
            inner_inputs = SpatialDropout1D(dropout_rate)(inputs)
        if encoder == 'word2vec+node2vec':
            inner_inputs = SpatialDropout1D(dropout_rate)(inputs[0])

        if encoder == 'word2vec':
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inputs])
            outputs = GlobalMeanPooling1D()(inner_inputs)
        elif encoder == 'word2vec+':
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inputs])
            inner_inputs = GlobalMeanPooling1D()(inner_inputs)
            outputs = Dense(units=hidden_units_size)(inner_inputs)
        elif encoder == 'node2vec':
            outputs = Dropout(dropout_rate)(inputs)
        elif encoder == 'word2vec+node2vec' and 'graph' in Configuration['model']['architecture'].lower():
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inner_inputs])
            inner_inputs = GlobalMeanPooling1D()(inner_inputs)
            inner_inputs = concatenate([inner_inputs, inputs[1]])
            outputs = Dense(units=hidden_units_size, activation='tanh')(inner_inputs)
        elif encoder == 'word2vec+node2vec':
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inner_inputs])
            inner_inputs = GlobalMeanPooling1D()(inner_inputs)
            inner_inputs = Dense(units=hidden_units_size, activation='tanh')(inner_inputs)
            inner_inputs2 = Dropout(dropout_rate)(inputs[1])
            inner_inputs2 = Dense(units=hidden_units_size, activation='tanh')(inner_inputs2)
            inner_inputs = concatenate([inner_inputs, inner_inputs2])
            outputs = Dense(units=hidden_units_size, activation='tanh')(inner_inputs)

        return outputs

    def _compile_label_wise_attention(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        # Document Encoding
        if self.elmo:
            document_inputs = Input(shape=(1,), dtype='string', name='document_inputs')
            document_elmos = ElmoEmbeddingLayer()(document_inputs)
            self._features = document_elmos.shape[-1].value
            document_inputs2 = Input(shape=(None,), name='document_inputs2')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs2)
            document_embs = concatenate([document_embs, document_elmos])
            inputs = [document_inputs, document_inputs2]
        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs)
            self._features = document_embs.shape[-1].value
            inputs = [document_inputs]

        document_encodings = self.TokenEncoder(inputs=document_embs, encoder=self.token_encoder,
                                               dropout_rate=dropout_rate,
                                               word_dropout_rate=word_dropout_rate, hidden_layers=n_hidden_layers,
                                               hidden_units_size=hidden_units_size)

        if Configuration['model']['return_attention']:
            document_label_encodings, document_attentions = \
                LabelWiseAttention(return_attention=True, n_classes=self.true_labels_cutoff,
                                   label_revisions=Configuration['model']['revisions'], encoder=self.token_encoder)(
                    document_encodings)

            loss = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            losses = {'label_wise_attention_1': loss}
            loss_weights = {'label_wise_attention_1': 1.0}
        else:
            document_label_encodings = \
                LabelWiseAttention(return_attention=False, n_classes=self.true_labels_cutoff,
                                   label_revisions=Configuration['model']['revisions'], encoder=self.token_encoder)(
                    document_encodings)
            losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            loss_weights = None

        if Configuration['model']['revisions'] is None:
            self._model = Model(inputs=inputs,
                                outputs=[document_label_encodings] if not Configuration['model'][
                                    'return_attention'] else [document_label_encodings,
                                                              document_attentions])
        else:
            model = Model(inputs=inputs,
                          outputs=[document_label_encodings] if not Configuration['model']['return_attention'] else [
                              document_label_encodings,
                              document_attentions])
            from keras.utils.multi_gpu_utils import multi_gpu_model
            self._model = multi_gpu_model(model, gpus=2)

        self._model.compile(optimizer=Adam(lr=lr) if Configuration['training']['accumulation_steps'] == 0 else
        AccumulatedAdam(lr=lr, accumulation_steps=Configuration['training']['accumulation_steps'], clipvalue=5.0),
                            loss=losses, loss_weights=loss_weights)

    def _compile_label_wise_attention_zero(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate,
                                           lr):

        # Document Encoding
        if self.elmo:
            document_inputs = Input(shape=(1,), dtype='string', name='document_inputs')
            document_elmos = ElmoEmbeddingLayer()(document_inputs)
            self._features = document_elmos.shape[-1].value
            document_inputs2 = Input(shape=(None,), name='document_inputs2')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs2)
            document_embs = concatenate([document_embs, document_elmos])
            inputs = [document_inputs, document_inputs2]
        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs)
            self._features = document_embs.shape[-1].value
            inputs = [document_inputs]

        document_encodings = self.TokenEncoder(inputs=document_embs, encoder=self.token_encoder,
                                               dropout_rate=dropout_rate,
                                               word_dropout_rate=word_dropout_rate, hidden_layers=n_hidden_layers,
                                               hidden_units_size=hidden_units_size)

        if self.labels_graph is None:
            if self.label_encoder not in ['node2vec', 'word2vec+node2vec+']:
                self.label_terms_ids = self.label_terms_ids[:self.true_labels_cutoff]
            else:
                self.label_terms_ids = [self.label_terms_ids[0][:self.true_labels_cutoff],
                                        self.label_terms_ids[1][:self.true_labels_cutoff]]

        # Labels Encoding
        if self.elmo:
            labels_inputs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids, dtype='string'),
                                  name='label_inputs')
            labels_embs = ElmoEmbeddingLayer()(labels_inputs)
            inputs.append(labels_inputs)
        elif self.label_encoder in ['word2vec', 'word2vec+']:
            labels_embs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids, dtype=K.tf.float32),
                                name='label_inputs')
            inputs.append(labels_embs)
        elif self.label_encoder == 'word2vec+node2vec':
            labels_inputs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids, dtype=K.tf.int32),
                                  name='label_inputs')
            labels_embs = self.pretrained_embeddings(labels_inputs)
            inputs.append(labels_inputs)
        elif self.label_encoder == 'node2vec':
            labels_embs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids[1], dtype=K.tf.float32),
                                name='label_node2vec_inputs')
            inputs.append(labels_embs)
        elif self.label_encoder == 'node2vec+':
            labels_inputs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids[0], dtype=K.tf.int32),
                                  name='label_centroids_inputs')
            labels_embs = self.pretrained_embeddings(labels_inputs)
            labels_inputs2 = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids[1], dtype=K.tf.float32),
                                   name='label_node2vec_inputs')
            labels_embs = [labels_embs, labels_inputs2]
            inputs.append(labels_inputs)
            inputs.append(labels_inputs2)

        label_encodings = self.LabelEncoder(labels_embs, encoder=self.label_encoder, dropout_rate=dropout_rate,
                                            hidden_units_size=document_encodings.shape[-1].value)

        # Set Labels' graph as input
        if self.labels_graph is not None:
            labels_graph = Input(tensor=K.tf.convert_to_tensor(self.labels_graph, dtype=K.tf.float32),
                                 name='label_graph')
            inputs.append(labels_graph)
        else:
            labels_graph = None

        # Label-wise Attention Mechanism matching documents with labels
        if Configuration['model']['return_attention']:
            if labels_graph is None:
                outputs, document_attentions = \
                    LabelDrivenAttention(return_attention=True, graph_op=None, cutoff=self.true_labels_cutoff)(
                        [document_encodings, label_encodings])
            elif 'add' in Configuration['model']['architecture'].lower():
                outputs, document_attentions = \
                    LabelDrivenAttention(return_attention=True, graph_op='add', cutoff=self.true_labels_cutoff)(
                        [document_encodings, label_encodings, labels_graph])
            else:
                outputs, document_attentions = \
                    LabelDrivenAttention(return_attention=True, graph_op='concat', cutoff=self.true_labels_cutoff)(
                        [document_encodings, label_encodings, labels_graph])
            loss = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            losses = {'outputs': loss}
            loss_weights = {'outputs': 1.0}
        else:
            if labels_graph is None:
                outputs = \
                    LabelDrivenAttention(return_attention=False, graph_op=None, cutoff=self.true_labels_cutoff)(
                        [document_encodings, label_encodings])
            elif 'add' in Configuration['model']['architecture'].lower():
                outputs = \
                    LabelDrivenAttention(return_attention=False, graph_op='add', cutoff=self.true_labels_cutoff)(
                        [document_encodings, label_encodings, labels_graph])
            else:
                outputs = \
                    LabelDrivenAttention(return_attention=False, graph_op='concat', cutoff=self.true_labels_cutoff)(
                        [document_encodings, label_encodings, labels_graph])
            losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            loss_weights = None

        # Compile network
        self._model = Model(inputs=inputs,
                            outputs=[outputs] if not Configuration['model']['return_attention'] else [outputs,
                                                                                                      document_attentions])

        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss=losses, loss_weights=loss_weights)
