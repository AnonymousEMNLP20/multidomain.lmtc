import os
from keras import backend as K
from keras.models import load_model
from lmtc.data import MODELS_DIR
from .layers import TimestepDropout, Camouflage, SymmetricMasking, LayerNormalization, \
    LabelWiseAttention, ElmoEmbeddingLayer, GlobalMeanPooling1D, LabelDrivenAttention

def fake_loss(y_true, y_pred):
    return 0


class NeuralNetwork:
    def __init__(self):
        self._model = None

    def compile(self, *args, **kwargs):
        raise NotImplemented

    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self._model.fit_generator(*args, **kwargs)

    def evaluate_generator(self, *args, **kwargs):
        return self._model.evaluate_generator(*args, **kwargs)

    def predict_generator(self, *args, **kwargs):
        return self._model.predict_generator(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    def summary(self, *args, **kwargs):
        return self._model.summary(*args, **kwargs)

    def dump(self, model_name: str, folder='train'):
        self._model.save(os.path.join(MODELS_DIR, folder, '{}.h5'.format(model_name)))

    def load(self, filename):
        self._model = load_model(filename, custom_objects={'TimestepDropout': TimestepDropout,
                                                           'Camouflage': Camouflage,
                                                           'SymmetricMasking': SymmetricMasking,
                                                           'LayerNormalization': LayerNormalization,
                                                           'LabelWiseAttention': LabelWiseAttention,
                                                           'ElmoEmbeddingLayer': ElmoEmbeddingLayer,
                                                           'GlobalMeanPooling1D': GlobalMeanPooling1D,
                                                           'LabelDrivenAttention': LabelDrivenAttention,
                                                           'loss': fake_loss})
        print('Model loaded...')

    def load_weights(self, filename):
        self._model.load_weights(filename)

    def save_weights(self, filename):
        self._model.save_weights(filename)

    def get_model_shape(self, element):
        if element == 'input':
            return self._model.input.shape[-1].value
        else:
            return self._model.output.shape[-1].value

    def __del__(self):
        K.clear_session()
        del self._model
