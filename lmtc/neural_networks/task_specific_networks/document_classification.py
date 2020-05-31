import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MinMaxNorm
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, Dropout, Dense, Embedding
from tensorflow.keras.layers import TimeDistributed, add, GRU, Input, SpatialDropout1D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from ..optimizers import AccumulatedAdam
from lmtc.experiments.configurations.configuration import Configuration
from lmtc.neural_networks.neural_network import NeuralNetwork
from lmtc.neural_networks.layers import PretrainedEmbedding, TimestepDropout, SymmetricMasking, Camouflage, Attention, \
                                                ContextualAttention, ELMO, BERT, ROBERTA, SCIBERT


class DocumentClassification(NeuralNetwork):
    def __init__(self, label_terms):
        super().__init__()
        self._cuDNN = Configuration['task']['cuDNN']
        self._decision_type = Configuration['task']['decision_type']
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.n_classes = label_terms
        print('classes:', self.n_classes)
        self._attention_mechanism = Configuration['model']['attention_mechanism']
        self.word_embedding_path = Configuration['model']['embeddings']

    def __del__(self):
        K.clear_session()
        del self._model

    def compile(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        if Configuration['model']['hierarchical']:
            if Configuration['model']['architecture'] == 'BERT':
                self.compile_hier_bert(dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'TRANSFORMERS':
            self._compile_transformers(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                       word_dropout_rate=word_dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'BERT':
            self._compile_bert(dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'ROBERTA':
            self._compile_roberta(dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'DISTILBERT':
            self._compile_distilbert(dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'XLNET':
            self._compile_xlnet(dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['attention_mechanism']:
            self._compile_bigrus_attention(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size, dropout_rate=dropout_rate,
                                           word_dropout_rate=word_dropout_rate, lr=lr)
        else:
            raise Exception("Described method is supported")

    def compile_hier_bert(self, dropout_rate, lr):
        # Init BERT section encoder
        bert = SCIBERT(dropout_rate=dropout_rate)
        bert(np.zeros((1, 64), dtype=np.int32))
        doc_inputs = Input(shape=(None, None,), name='document_inputs', dtype='int32')

        # Distribute sentences
        section_encodings = TimeDistributed(bert, name='sentence_encodings')(doc_inputs)

        # Compute mask input to exclude padded sentences
        mask = Lambda(lambda x: K.sum(x, axis=2), output_shape=lambda s: (s[0], s[1]), name='find_masking')(doc_inputs)
        section_encodings = Camouflage(mask_value=0, name='camouflage')([section_encodings, mask])

        if self._attention_mechanism == 'maxpooling':
            doc_encoding = GlobalMaxPooling1D(name='max_pooling')(section_encodings)
        elif self._attention_mechanism == 'attention':
            if Configuration['model']['return_attention']:
                doc_encoding, section_attentions = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(),
                                                                       return_attention=True, name='self_attention')(
                    section_encodings)
            else:
                doc_encoding = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(),
                                                   return_attention=False, name='self_attention')(
                    section_encodings)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        self._model = Model(inputs=[doc_inputs], outputs=[outputs])
        self._model.compile(Adam(lr=lr), loss=losses, loss_weights=loss_weights)

    def _compile_bert(self, dropout_rate, lr):

        word_inputs = Input(shape=(None,), name='word_inputs', dtype='int32')
        mask_inputs = Input(shape=(None,), name='mask_inputs', dtype='int32')
        seg_inputs = Input(shape=(None,), name='seg_inputs', dtype='int32')
        if Configuration['model']['bert'] == 'scibert':
            from transformers import BertConfig
            model_path = '/home/alakazam/scibert_scivocab_uncased/'
            bert = BERT.from_pretrained(model_path+'pytorch_model.bin',
                                               from_pt=True,
                                               config=BertConfig().from_pretrained(model_path+'config.json'))
        elif Configuration['model']['bert'] == 'legalbert_small':
            from transformers import BertConfig
            model_path = '/home/alakazam/legal_bert_small/'
            bert = BERT.from_pretrained(model_path+'pytorch_model.bin',
                                               from_pt=True,
                                               config=BertConfig().from_pretrained(model_path+'config.json'))
        elif Configuration['model']['bert'] == 'bert_uncased_L-6_H-512_A-8':
            from transformers import TFAutoModel
            bert = TFAutoModel.from_pretrained("google/bert_uncased_L-6_H-512_A-8", from_pt=True)
        else:
            bert = BERT.from_pretrained('bert-base-uncased')
        bert_encodings = bert([word_inputs, mask_inputs, seg_inputs])
        doc_encoding = tf.squeeze(bert_encodings[0][:, 0:1, :], axis=1)
        doc_encoding = Dropout(dropout_rate)(doc_encoding)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        self._model = Model(inputs=[word_inputs, mask_inputs, seg_inputs], outputs=[outputs])
        self._model.compile(Adam(lr=lr), loss=losses, loss_weights=loss_weights)

    def _compile_roberta(self, dropout_rate, lr):

        word_inputs = Input(shape=(None,), name='word_inputs', dtype='int32')
        mask_inputs = Input(shape=(None,), name='mask_inputs', dtype='int32')
        seg_inputs = Input(shape=(None,), name='seg_inputs', dtype='int32')
        roberta = ROBERTA.from_pretrained('roberta-base')
        roberta_encodings = roberta([word_inputs, mask_inputs, seg_inputs])[0]
        doc_encoding = tf.squeeze(roberta_encodings[:, 0:1, :], axis=1)
        doc_encoding = Dropout(dropout_rate)(doc_encoding)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        self._model = Model(inputs=[word_inputs, mask_inputs, seg_inputs], outputs=[outputs])
        self._model.compile(optimizer=AccumulatedAdam(lr=lr, accumulation_steps=2), loss=losses, loss_weights=loss_weights)
