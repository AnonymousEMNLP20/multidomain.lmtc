from keras import backend as K
from keras.layers import GlobalMaxPooling1D, Dropout, Dense, Input, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from lmtc.experiments.configurations.configuration import Configuration
from lmtc.neural_networks.layers import SymmetricMasking, Camouflage, ALBERT, BERT
from lmtc.neural_networks.neural_network import NeuralNetwork
from ..utilities import AccumulatedAdam


class DocumentClassification(NeuralNetwork):
    def __init__(self, label_terms):
        super().__init__()
        self._cuDNN = Configuration['task']['cuDNN']
        self._decision_type = Configuration['task']['decision_type']
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.n_classes = label_terms
        self._attention_mechanism = Configuration['model']['attention_mechanism']

    def __del__(self):
        K.clear_session()
        del self._model

    def compile(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        K.clear_session()
        if Configuration['model']['hierarchical'] and Configuration['model']['architecture'] == 'BERT':
            self._compile_hier_bert(dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'BERT':
            self._compile_bert(dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'ALBERT':
            self._compile_albert(dropout_rate=dropout_rate, lr=lr)
        else:
            raise Exception('DNN architecture is not supported!!!')

    def _compile_bert(self, dropout_rate, lr):

        word_inputs = Input(shape=(None, 3), dtype='int32')
        doc_encoding = BERT()(word_inputs)

        doc_encoding = Dropout(dropout_rate)(doc_encoding)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        self._model = Model(inputs=[word_inputs], outputs=[outputs])
        self._model.compile(optimizer=Adam(lr=lr) if Configuration['training']['accumulation_steps'] == 0 else
                            AccumulatedAdam(lr=lr, accumulation_steps=Configuration['training']['accumulation_steps']),
                            loss=losses, loss_weights=loss_weights)

    def _compile_albert(self, dropout_rate, lr):

        word_inputs = Input(shape=(None, 3), dtype='int32')
        doc_encoding = ALBERT()(word_inputs)
        doc_encoding = Dropout(dropout_rate)(doc_encoding)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        self._model = Model(inputs=[word_inputs], outputs=[outputs])
        self._model.compile(optimizer=Adam(lr=lr) if Configuration['training']['accumulation_steps'] == 0 else
                            AccumulatedAdam(lr=lr, accumulation_steps=Configuration['training']['accumulation_steps']),
                            loss=losses, loss_weights=loss_weights)

    def _compile_hier_bert(self, dropout_rate, lr):

            word_inputs = Input(shape=(None, 3), dtype='int32')

            bert_cls = BERT()(word_inputs)
            section_encoder = Dropout(dropout_rate)(bert_cls)

            section_encoder = Model(inputs=word_inputs, outputs=section_encoder)

            section_inputs = Input(shape=(None, None, 3), name='input_layer_documents', dtype='int32')

            # Distribute sentences
            section_encodings = TimeDistributed(section_encoder, name='sentence_encodings')(section_inputs)

            # Compute mask input to exclude padded sentences
            mask = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]), name='find_masking')(section_inputs)
            section_encodings = Camouflage(mask_value=0, name='camouflage')([section_encodings, mask])

            # Attention over BI-LSTM (context-aware) sentence embeddings
            if self._attention_mechanism == 'maxpooling':
                doc_encoding = GlobalMaxPooling1D(name='max_pooling')(section_encodings)
            elif self._attention_mechanism == 'attention':
                sentence_encodings = SymmetricMasking(mask_value=0, name='masking')([section_encodings, section_encodings])
                doc_encoding = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(),
                                                   return_attention=False, name='section_attentions')(sentence_encodings)

            doc_encoding = Dropout(dropout_rate)(doc_encoding)

            # Final output (projection) layer
            outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                            name='outputs')(doc_encoding)

            losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
            loss_weights = None

            # Wrap up model + Compile with optimizer and loss function
            self._model = Model(inputs=[section_inputs], outputs=[outputs])
            self._model.compile(optimizer=Adam(lr=lr) if Configuration['training']['accumulation_steps'] == 0 else
                                AccumulatedAdam(lr=lr, accumulation_steps=Configuration['training']['accumulation_steps']),
                                loss=losses, loss_weights=loss_weights)
