from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Conv1D, Activation
from tensorflow.keras.layers import Input, SpatialDropout1D
from tensorflow.keras.layers import GRU, add, concatenate
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from lmtc.experiments.configurations.configuration import Configuration
from lmtc.neural_networks.layers import Camouflage
from lmtc.neural_networks.layers import TimestepDropout, SymmetricMasking, \
    ELMO, LabelDrivenAttention, LabelWiseAttention, BERT, PretrainedEmbedding
from lmtc.neural_networks.neural_network import NeuralNetwork
from ..optimizers import AdvancedAdam


class LabelDrivenClassification(NeuralNetwork):
    def __init__(self, label_terms_ids, labels_graph, true_labels_cutoff):
        super().__init__()
        self._conf = 'NONE'
        self._cuDNN = Configuration['task']['cuDNN']
        self._decision_type = Configuration['task']['decision_type']
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.label_encoder = Configuration['model']['label_encoder']
        self.token_encoder = Configuration['model']['document_encoder']
        self.word_embedding_path = Configuration['model']['embeddings']
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
            self._compile_hierarchical_label_wise_attention(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                                                            dropout_rate=dropout_rate, word_dropout_rate=word_dropout_rate, lr=lr)

        elif self.label_encoder:
            self._compile_label_wise_attention_zero(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                                    word_dropout_rate=word_dropout_rate, lr=lr)
        else:
            self._compile_label_wise_attention(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                               word_dropout_rate=word_dropout_rate, lr=lr)

    def TokenEncoder(self, inputs, encoder, dropout_rate, word_dropout_rate, hidden_layers, hidden_units_size):

        # Apply variational drop-out
        if encoder != 'bert':
            inner_inputs = SpatialDropout1D(dropout_rate)(inputs)
            inner_inputs = TimestepDropout(word_dropout_rate)(inner_inputs)

        if encoder == 'grus':
            # Bi-GRUs over token embeddings
            for i in range(hidden_layers):
                if self._cuDNN:
                    bi_grus = Bidirectional(GRU(units=hidden_units_size, return_sequences=True))(inner_inputs)
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

        return outputs

    def LabelEncoder(self, inputs, encoder, dropout_rate, hidden_units_size):

        # Apply variational drop-out + Mask input to exclude paddings
        inner_inputs = SpatialDropout1D(dropout_rate)(inputs)

        if encoder == 'cnns':
            inner_inputs = Conv1D(filters=hidden_units_size, kernel_size=3, strides=1, padding="same")(inner_inputs)
            inner_inputs = Camouflage(mask_value=0)(inputs=[inner_inputs, inputs])
            outputs = GlobalMaxPooling1D()(inner_inputs)
        elif encoder == 'grus':
            if self._cuDNN:
                outputs = GRU(units=hidden_units_size)(inner_inputs)
            else:
                outputs = GRU(units=hidden_units_size, activation="tanh",
                              recurrent_activation='sigmoid')(inner_inputs)
        elif encoder == 'average+':
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inputs])
            inner_inputs = GlobalMeanPooling1D()(inner_inputs)
            outputs = Dense(units=hidden_units_size)(inner_inputs)

        elif encoder == 'average':
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inputs])
            outputs = GlobalMeanPooling1D()(inner_inputs)
        elif encoder == 'bert':
            inner_inputs = BERT(output_representation='sequence_output', trainable=False)(inputs)
            outputs = Camouflage(mask_value=0)(inputs=[inner_inputs, inputs])

        return outputs

    def _compile_label_wise_attention(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        # Document Encoding
        if self.token_encoder == 'bert':
            word_inputs = Input(shape=(None,), name='word_inputs', dtype='int32')
            mask_inputs = Input(shape=(None,), name='mask_inputs', dtype='int32')
            seg_inputs = Input(shape=(None,), name='seg_inputs', dtype='int32')
            # start_inputs = Input(shape=(None,), name='start_inputs', dtype='int32')
            document_embs = [word_inputs, mask_inputs, seg_inputs]
            inputs = [word_inputs, mask_inputs, seg_inputs]
        elif self.elmo:
            document_inputs = Input(shape=(1, ), dtype='string', name='document_inputs')
            document_elmos = ELMO()(document_inputs)
            self._features = document_elmos.shape[-1].value
            document_inputs2 = Input(shape=(None,), name='document_inputs2')
            self.pretrained_embeddings = PretrainedEmbedding(path=self.word_embedding_path, dims=200)
            document_embs = self.pretrained_embeddings(document_inputs2)
            document_embs = concatenate([document_embs, document_elmos])
            inputs = [document_inputs, document_inputs2]
        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = PretrainedEmbedding(path=self.word_embedding_path, dims=200)
            document_embs = self.pretrained_embeddings(document_inputs)
            self._features = document_embs.shape[-1].value
            inputs = [document_inputs]

        document_encodings = self.TokenEncoder(inputs=document_embs, encoder=self.token_encoder, dropout_rate=dropout_rate,
                                               word_dropout_rate=word_dropout_rate, hidden_layers=n_hidden_layers,
                                               hidden_units_size=hidden_units_size)

        # Label-wise Attention Mechanism matching documents with labels
        if Configuration['model']['return_attention']:
            document_label_encodings, document_attentions = \
                LabelWiseAttention(return_attention=True, n_classes=self.true_labels_cutoff)(document_encodings)

            loss = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            losses = {'label_wise_attention_1': loss}
            loss_weights = {'label_wise_attention_1': 1.0}
        else:
            document_label_encodings = \
                LabelWiseAttention(return_attention=False, n_classes=self.true_labels_cutoff)(document_encodings)
            losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            loss_weights = None

        self._model = Model(inputs=inputs,
                            outputs=[document_label_encodings] if not Configuration['model']['return_attention'] else [document_label_encodings, document_attentions])

        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0) if self.token_encoder != 'bert'
                            else AdvancedAdam(lr=lr, multipliers={'label_wise_attention_1': hidden_units_size}, clipvalue=5.0),
                            loss=losses, loss_weights=loss_weights)

    def _compile_label_wise_attention_zero(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        # Document Encoding
        if self.token_encoder == 'bert':
            word_inputs = Input(shape=(None,), name='word_inputs', dtype='int32')
            mask_inputs = Input(shape=(None,), name='mask_inputs', dtype='int32')
            seg_inputs = Input(shape=(None,), name='seg_inputs', dtype='int32')
            document_embs = [word_inputs, mask_inputs, seg_inputs]
        elif self.elmo:
            document_inputs = Input(shape=(1, ), dtype='string', name='document_inputs')
            document_embs = ELMO()(document_inputs)
            self._features = document_embs.shape[-1].value
        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = PretrainedEmbedding(path=self.word_embedding_path, dims=200)
            document_embs = self.pretrained_embeddings(document_inputs)
            self._features = document_embs.shape[-1].value

        document_ngram_encodings = self.TokenEncoder(inputs=document_embs, encoder=self.token_encoder, dropout_rate=dropout_rate,
                                                     word_dropout_rate=word_dropout_rate, hidden_layers=n_hidden_layers,
                                                     hidden_units_size=hidden_units_size)

        if self.labels_graph is None:
            self.label_terms_ids = self.label_terms_ids[:self.true_labels_cutoff]
        # Labels Encoding
        if self.elmo:
            labels_inputs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids, dtype='string'), name='label_inputs')
            labels_embs = ELMO()(labels_inputs)
        else:
            labels_inputs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids, dtype=K.tf.int32), name='label_inputs')
            labels_embs = self.pretrained_embeddings(labels_inputs)

        label_encodings = self.LabelEncoder(labels_embs, encoder=self.label_encoder, dropout_rate=dropout_rate,
                                            hidden_units_size=document_ngram_encodings.shape[-1].value)

        # Set Labels' graph as input
        if self.labels_graph is not None:
            labels_graph = Input(tensor=K.tf.convert_to_tensor(self.labels_graph, dtype=K.tf.float32), name='label_graph')
        else:
            labels_graph = None

        # Label-wise Attention Mechanism matching documents with labels
        if Configuration['model']['return_attention']:
            if labels_graph is None:
                outputs, document_attentions = \
                    LabelDrivenAttention(return_attention=True, graph_op=None, cutoff=self.true_labels_cutoff)([document_ngram_encodings, label_encodings])
            elif 'add' in Configuration['model']['architecture'].lower():
                outputs, document_attentions = \
                    LabelDrivenAttention(return_attention=True, graph_op='add', cutoff=self.true_labels_cutoff)([document_ngram_encodings, label_encodings, labels_graph])
            else:
                outputs, document_attentions = \
                    LabelDrivenAttention(return_attention=True, graph_op='concat', cutoff=self.true_labels_cutoff)([document_ngram_encodings, label_encodings, labels_graph])
            loss = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            losses = {'outputs': loss}
            loss_weights = {'outputs': 1.0}
        else:
            if labels_graph is None:
                outputs = \
                    LabelDrivenAttention(return_attention=False, graph_op=None, cutoff=self.true_labels_cutoff)([document_ngram_encodings, label_encodings])
            elif 'add' in Configuration['model']['architecture'].lower():
                outputs = \
                    LabelDrivenAttention(return_attention=False, graph_op='add', cutoff=self.true_labels_cutoff)([document_ngram_encodings, label_encodings, labels_graph])
            else:
                outputs = \
                    LabelDrivenAttention(return_attention=False, graph_op='concat', cutoff=self.true_labels_cutoff)([document_ngram_encodings, label_encodings, labels_graph])
            losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            loss_weights = None

        # Compile network
        self._model = Model(inputs=[document_inputs, labels_inputs] if labels_graph is None else [document_inputs, labels_inputs, labels_graph],
                            outputs=[outputs] if not Configuration['model']['return_attention'] else [outputs, document_attentions])
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss=losses, loss_weights=loss_weights)

    def _compile_hierarchical_label_wise_attention(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        # Document Encoding
        if self.elmo:
            section_inputs = Input(shape=(1, ), dtype='string', name='document_inputs')
            section_embs = ELMO()(section_inputs)
            self._features = section_embs.shape[-1].value
        else:
            section_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = PretrainedEmbedding(path=self.word_embedding_path, dims=200)
            section_embs = self.pretrained_embeddings(section_inputs)
            self._features = section_embs.shape[-1].value

        section_ngram_encodings = self.TokenEncoder(inputs=section_embs, encoder=self.token_encoder, dropout_rate=dropout_rate,
                                                    word_dropout_rate=word_dropout_rate, hidden_layers=n_hidden_layers,
                                                    hidden_units_size=hidden_units_size)

        section_label_aware_encodings = LabelWiseAttention(n_classes=self.true_labels_cutoff,
                                                           return_attention=False)(section_ngram_encodings)
        section_encoder = Model(inputs=section_inputs, outputs=section_label_aware_encodings)

        # Document Input Layer
        if self.elmo:
            doc_inputs = Input(shape=(None, 1,), dtype='string', name='document_inputs')
        else:
            doc_inputs = Input(shape=(None, None,), name='document_inputs')

        # Distribute sentences
        section_encodings = TimeDistributed(section_encoder, name='sentence_encodings')(doc_inputs)
        # Classification
        outputs = GlobalMaxPooling1D()(section_encodings)

        # Compile network
        self._model = Model(inputs=[doc_inputs],
                            outputs=[outputs])
        self._model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                            loss='binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy')
