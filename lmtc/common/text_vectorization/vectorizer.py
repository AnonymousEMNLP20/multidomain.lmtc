from typing import List
import os
import numpy as np
import pickle
from lmtc.data import VECTORS_DIR, MODELS_DIR
from lmtc.experiments.configurations.configuration import Configuration
from lmtc.neural_networks.layers.custom_bert.vocab import BERTTextEncoder
from lmtc.neural_networks.layers.custom_albert.vocab import ALBERTTextEncoder


class Vectorizer(object):

    def __init__(self):
        pass

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):
        raise NotImplementedError


class BERTVectorizer(Vectorizer):

    def __init__(self):
        super().__init__()

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):

        if Configuration['model']['bert'] == 'biobert':
            bert_tokenizer = BERTTextEncoder(vocab_file=os.path.join(MODELS_DIR, 'bert', 'biobert', 'assets', 'vocab.txt'),
                                             do_lower_case=False, max_len=max_sequence_size)
        elif Configuration['model']['bert'] == 'clinicalbert':
            bert_tokenizer = BERTTextEncoder(vocab_file=os.path.join(MODELS_DIR, 'bert', 'clinicalbert', 'assets', 'vocab.txt'),
                                             do_lower_case=False, max_len=max_sequence_size)
        elif Configuration['model']['bert'] == 'scibert':
            bert_tokenizer = BERTTextEncoder(vocab_file=os.path.join(MODELS_DIR, 'bert', 'scibert', 'assets', 'vocab.txt'),
                                             do_lower_case=True, max_len=max_sequence_size)
        elif Configuration['model']['bert'] == 'legalbert':
            bert_tokenizer = BERTTextEncoder(vocab_file=os.path.join(MODELS_DIR, 'bert', 'legalbert', 'assets', 'vocab.txt'),
                                             do_lower_case=True, max_len=max_sequence_size)
        elif Configuration['model']['bert'] == 'albert-base':
            vocab_model = 'vocab_base.model' if Configuration['model']['bert'] == 'albert-base' else 'vocab_large.model'
            bert_tokenizer = ALBERTTextEncoder(spm_model_file=os.path.join(VECTORS_DIR, 'bert', vocab_model),
                                               do_lower_case=True, max_len=max_sequence_size)
        elif Configuration['model']['bert'] == 'bert-base':
            bert_tokenizer = BERTTextEncoder(vocab_file=os.path.join(VECTORS_DIR, 'bert-base', 'vocab.txt'),
                                             do_lower_case=True, max_len=max_sequence_size)
        else:
            raise Exception('BERT version {} is not supported'.format(Configuration['model']['bert']))

        token_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)
        seg_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)
        mask_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)
        start_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)

        for i, tokens in enumerate(sequences):
            text = ' '.join([token for token in tokens[:max_sequence_size]])
            if Configuration['model']['bert'] in ['legalbert_500K', 'legalbert_1M']:
                text = text.replace('\n', 'newline')
            tokens, starts = bert_tokenizer.encode(text)
            token_indices[i, :len(tokens)] = tokens
            mask_indices[i, :len(tokens)] = np.ones((len(tokens)), dtype=np.int32)
            start_indices[i, :len(tokens)] = starts

        return np.concatenate((np.reshape(token_indices, [len(sequences), max_sequence_size, 1]),
                               np.reshape(mask_indices, [len(sequences), max_sequence_size, 1]),
                               np.reshape(seg_indices, [len(sequences), max_sequence_size, 1])), axis=-1), start_indices


class ELMoVectorizer(Vectorizer):

    def __init__(self):
        super().__init__()

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):

        word_inputs = []
        # Encode ELMo embeddings
        for i, tokens in enumerate(sequences):
            sequence = ' '.join([token for token in tokens[:max_sequence_size]])
            if len(tokens) < max_sequence_size:
                sequence = sequence + ' ' + ' '.join(['#' for i in range(max_sequence_size - len(tokens))])
            word_inputs.append([sequence])
        return np.asarray(word_inputs)


class W2VVectorizer(Vectorizer):

    def __init__(self, w2v_model='glove.6B.200d.bin'):
        super().__init__()
        w2v_index = w2v_model.split('/')[-1].replace('.bin', '.index')
        self.indices = {'word': pickle.load(open(os.path.join(VECTORS_DIR, 'indices', w2v_index), 'rb'))}

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, features=['word', 'shape', 'pos']):
        """
        Produce W2V indices for each token in the list of tokens
        :param sequences: list of lists of tokens
        :param max_sequence_size: maximum padding
        :param features: features to be considered
        """

        word_inputs = np.zeros((len(sequences), max_sequence_size, ), dtype=np.int32)
        for i, sentence in enumerate(sequences):
            for j, token in enumerate(sentence[:max_sequence_size]):
                if token.lower() in self.indices['word']:
                    word_inputs[i][j] = self.indices['word'][token.norm]
                else:
                    word_inputs[i][j] = self.indices['word']['UNKNOWN']

        return word_inputs
