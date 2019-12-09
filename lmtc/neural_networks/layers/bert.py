import os
import logging
import tensorflow_hub as hub
from keras.layers import Layer, Lambda
import keras.backend as K
from lmtc.data import MODELS_DIR
from lmtc.experiments.configurations.configuration import Configuration

LOGGER = logging.getLogger(__name__)


class BERT(Layer):
    def __init__(self, output_representation='pooled_output', trainable=True, **kwargs):
        self.bert = None
        self.trainable = trainable
        super(BERT, self).__init__(**kwargs)

        self.output_representation = output_representation

    def build(self, input_shape):
        if Configuration['model']['bert'] == 'biobert':
            self.bert = hub.Module((os.path.join(MODELS_DIR, 'bert', 'biobert')),
                                   trainable=self.trainable, name="{}_module".format(self.name))
        elif Configuration['model']['bert'] == 'clinicalbert':
            self.bert = hub.Module((os.path.join(MODELS_DIR, 'bert', 'clinicalbert')),
                                   trainable=self.trainable, name="{}_module".format(self.name))
        elif Configuration['model']['bert'] == 'scibert':
            self.bert = hub.Module((os.path.join(MODELS_DIR, 'bert', 'scibert')),
                                   trainable=self.trainable, name="{}_module".format(self.name))
        elif Configuration['model']['bert'] == 'legalbert':
            self.bert = hub.Module((os.path.join(MODELS_DIR, 'bert', 'legalbert')),
                                   trainable=self.trainable, name="{}_module".format(self.name))
        elif Configuration['model']['bert'] == 'bertbase':
            self.bert = hub.Module('https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                                   trainable=self.trainable, name="{}_module".format(self.name))
        else:
            raise Exception("Not supported BERT version!!!")

        # Remove unused layers and set trainable parameters
        if self.trainable:
            self.trainable_weights += [var for var in self.bert.variables
                                       if not "/cls/" in var.name and not "/pooler/" in var.name]
        super(BERT, self).build(input_shape)

    def call(self, x, mask=None):

        splits = Lambda(lambda k: K.tf.split(k, num_or_size_splits=3, axis=2))(x)

        inputs = []
        for i in range(len(splits)):
            inputs.append(Lambda(lambda s: K.tf.squeeze(s, axis=-1), name='squeeze_{}'.format(i))(splits[i]))

        outputs = self.bert(dict(input_ids=inputs[0], input_mask=inputs[1], segment_ids=inputs[2]), as_dict=True, signature='tokens')[
            'sequence_output']

        if self.output_representation == 'pooled_output':
                return K.tf.squeeze(outputs[:, 0:1, :], axis=1)
        elif self.output_representation == 'all':
            return [K.tf.squeeze(outputs[:, 0:1, :], axis=1), outputs]
        else:
            return outputs

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.output_representation == 'pooled_output':
            return (None, 1024 if 'L-24' in Configuration['model']['bert'] else 768)
        elif self.output_representation == 'all':
            return [(None, 1024 if 'L-24' in Configuration['model']['bert'] else 768), (None, None, 1024 if 'L-24' in Configuration['model']['bert'] else 768)]
        else:
            return (None, None, 1024 if 'L-24' in Configuration['model']['bert'] else 768)


class ALBERT(Layer):
    def __init__(self, output_representation='pooled_output', trainable=True, **kwargs):
        self.albert = None
        self.trainable = trainable
        super(ALBERT, self).__init__(**kwargs)

        self.output_representation = output_representation

    def build(self, input_shape):
        if Configuration['model']['bert'] == 'albert-base':
            self.albert = hub.Module("https://tfhub.dev/google/albert_base/1",
                                     trainable=self.trainable, tags=set(["train"]), name="{}_module".format(self.name))
        else:
            self.albert = hub.Module("https://tfhub.dev/google/albert_large/1",
                                     trainable=self.trainable, tags=set(["train"]), name="{}_module".format(self.name))

        # Remove unused layers and set trainable parameters
        if self.trainable:
            self.trainable_weights += [var for var in self.albert.variables
                                       if not "/cls/" in var.name and not "/pooler/" in var.name]
        super(ALBERT, self).build(input_shape)

    def call(self, x, mask=None):

        splits = Lambda(lambda k: K.tf.split(k, num_or_size_splits=3, axis=2))(x)

        inputs = []
        for i in range(len(splits)):
            inputs.append(Lambda(lambda s: K.tf.squeeze(s, axis=-1), name='squeeze_{}'.format(i))(splits[i]))

        outputs = self.albert(dict(input_ids=inputs[0], input_mask=inputs[1], segment_ids=inputs[2]),
                              as_dict=True, signature='tokens')['sequence_output']

        if self.output_representation == 'pooled_output':
            return K.tf.squeeze(outputs[:, 0:1, :], axis=1)
        elif self.output_representation == 'all':
            return [K.tf.squeeze(outputs[:, 0:1, :], axis=1), outputs]
        else:
            return outputs

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        hidden_units = 768 if Configuration['model']['bert'] == 'albert-base' else 1024
        if self.output_representation == 'pooled_output':
            return (None, hidden_units)
        elif self.output_representation == 'all':
            return [(None, hidden_units), (None, None, hidden_units)]
        else:
            return (None, None, hidden_units)
