from typing import List
import os
import numpy as np
import pickle
from lmtc.data import VECTORS_DIR, MODELS_DIR
from lmtc.document_model.model import Token, Tag
from lmtc.experiments.configurations.configuration import Configuration
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer


class Vectorizer(object):

    def __init__(self):
        pass

    def vectorize_inputs(self, sequences: List[List[Token]], max_sequence_size=100, **kwargs):
        raise NotImplementedError

    def vectorize_targets(self, sequences: List[List[Token]], tags: List[List[List[Tag]]]):
        raise NotImplementedError


class BERTVectorizer(Vectorizer):

    def __init__(self):
        if Configuration['model']['bert'] == 'biobert':
            self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file=os.path.join(MODELS_DIR, 'biobert', 'assets', 'vocab.txt'))
        elif Configuration['model']['bert'] == 'roberta':
            self.bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif Configuration['model']['bert'] == 'scibert':
            self.bert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        super().__init__()

    def vectorize_inputs(self, sequences: List[List[Token]], max_sequence_size=100, **kwargs):

        inputs = np.zeros((2, len(sequences), max_sequence_size, ), dtype=np.int32)
        # Encode BERT embeddings
        for i, tokens in enumerate(sequences):
            tokens = self.bert_tokenizer.encode(' '.join([token.token_text for token in tokens[:max_sequence_size]]))
            if max_sequence_size <= len(tokens):
                inputs[0, i, :max_sequence_size] = tokens[:max_sequence_size-1] + [tokens[-1]]
                inputs[1, i, :max_sequence_size] = np.ones((max_sequence_size,), dtype=np.int32)
            else:
                inputs[0, i, :len(tokens)] = tokens
                inputs[1, i, :len(tokens)] = np.ones((len(tokens),), dtype=np.int32)

        return inputs

    def vectorize_targets(self, sequences: List[List[Token]], tags: List[List[List[Tag]]]):
        pass


class ELMoVectorizer(Vectorizer):

    def __init__(self):
        super().__init__()

    def vectorize_inputs(self, sequences: List[List[Token]], max_sequence_size=100, **kwargs):

        word_inputs = []
        # Encode ELMo embeddings
        for i, tokens in enumerate(sequences):
            sequence = ' '.join([token.token_text for token in tokens[:max_sequence_size]])
            if len(tokens) < max_sequence_size:
                sequence = sequence + ' ' + ' '.join(['#' for i in range(max_sequence_size - len(tokens))])
            word_inputs.append([sequence])
        return np.asarray(word_inputs)

    def vectorize_targets(self, sequences: List[List[Token]], tags: List[List[List[Tag]]]):
        pass


class W2VVectorizer(Vectorizer):

    def __init__(self, w2v_model='glove.6B.200d.bin'):
        super().__init__()
        w2v_index = w2v_model.split('/')[-1].replace('.bin', '.index')
        self.indices = {'word': pickle.load(open(os.path.join(VECTORS_DIR, 'indices', w2v_index), 'rb'))}

    def vectorize_inputs(self, sequences: List[List[Token]], max_sequence_size=100, features=['word', 'shape', 'pos']):
        """
        Produce W2V indices for each token in the list of tokens
        :param sequences: list of lists of tokens
        :param max_sequence_size: maximum padding
        :param features: features to be considered
        """

        word_inputs = np.zeros((len(sequences), max_sequence_size, ), dtype=np.int32)
        if 'shape' in features:
            shape_inputs = np.zeros((len(sequences), max_sequence_size,), dtype=np.int32)
        if 'pos' in features:
            pos_inputs = np.zeros((len(sequences), max_sequence_size,), dtype=np.int32)

        for i, sentence in enumerate(sequences):
            for j, token in enumerate(sentence[:max_sequence_size]):
                if token.norm in self.indices['word']:
                    word_inputs[i][j] = self.indices['word'][token.norm]
                else:
                    word_inputs[i][j] = self.indices['word']['UNKNOWN']
                if 'shape' in features:
                    if token.features['shape'] in self.indices['shape']:
                        shape_inputs[i][j] = self.indices['shape'][token.features['shape']]
                    else:
                        shape_inputs[i][j] = self.indices['shape']['UNKNOWN']
                if 'pos' in features:
                    if token.features['pos'] in self.indices['pos']:
                        pos_inputs[i][j] = self.indices['pos'][token.features['pos']]
                    else:
                        pos_inputs[i][j] = self.indices['pos']['UNKNOWN']
        if features == ['word', 'shape', 'pos']:
            return word_inputs, shape_inputs, pos_inputs
        else:
            return word_inputs

    def vectorize_targets(self, sequences: List[List[Token]], tags: List[List[List[Tag]]], max_sequence_size=100):
        """
        Produce (pos, shape, label) for each token in the list of tokens
        :param sequences: list of lists of tokens
        :param tags: list of lists of tags
        :param max_sequence_size: maximum padding
        """

        labels = np.zeros((len(tags), max_sequence_size, len(Configuration['classes'])), dtype=np.int8)

        for i, sentence_tags in enumerate(tags):
            for j, token_tags in enumerate(sentence_tags[:max_sequence_size]):
                for tag in token_tags:
                    if tag.name in Configuration['task']['classes']:
                        labels[i][j][Configuration['task']['classes'][tag.name]] = 1
                if not sum(labels[i][j]):
                    labels[i][j][0] = 1

        return labels
