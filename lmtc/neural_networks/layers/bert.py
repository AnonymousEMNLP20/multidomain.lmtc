import tensorflow as tf
import transformers


class SCIBERT(tf.keras.Model):

    def __init__(self, dropout_rate):
        super(SCIBERT, self).__init__()
        self.bert = transformers.TFAutoModel.from_pretrained('allenai/scibert_scivocab_uncased', from_pt=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, **kwargs):
        bert_encodings = self.bert(inputs)
        sent_encoding = tf.squeeze(bert_encodings[:, 0:1, :], axis=1)
        return self.dropout(sent_encoding)

    def compute_output_shape(self, input_shape):
        return None, 768


class BERT(transformers.TFRobertaModel):

    def __init__(self, config, *inputs, **kwargs):
        super(BERT, self).__init__(config, *inputs, **kwargs)


class ROBERTA(transformers.TFRobertaModel):

    def __init__(self, config, *inputs, **kwargs):
        super(ROBERTA, self).__init__(config, *inputs, **kwargs)
        self.roberta.call = tf.function(self.roberta.call)


class DISTILBERT(transformers.TFDistilBertModel):

    def __init__(self, config, *inputs, **kwargs):
        super(DISTILBERT, self).__init__(config, *inputs, **kwargs)
        self.distilbert.call = tf.function(self.distilbert.call)
