import json
import logging
import os
import pickle
import re
import tempfile
import time
import glob
import tqdm
from copy import deepcopy
from collections import Counter

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score

from lmtc.common.text_vectorization.vectorizer import W2VVectorizer, ELMoVectorizer, BERTVectorizer
from lmtc.data import DATA_SET_DIR, PREDICTIONS_DIR
from lmtc.loaders.base import LoaderFactory
from lmtc.document_model.model import Token
from lmtc.experiments.configurations.configuration import Configuration
from lmtc.metrics.retrieval import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k, mean_rprecision
from lmtc.neural_networks.task_specific_networks.document_classification import DocumentClassification
from lmtc.neural_networks.task_specific_networks.label_driven_classification import LabelDrivenClassification
from lmtc.neural_networks.utilities import probas_to_classes
from ..experiment import Experiment

LOGGER = logging.getLogger(__name__)


class LMTC(Experiment):

    def __init__(self):
        super().__init__()
        if 'elmo' in Configuration['model']['token_encoding']:
            self.vectorizer = ELMoVectorizer()
            self.vectorizer2 = W2VVectorizer(w2v_model=Configuration['model']['embeddings'])
        elif Configuration['model']['architecture'].lower() in ['bert', 'roberta', 'xlnet']:
            self.vectorizer = BERTVectorizer()
        else:
            self.vectorizer = W2VVectorizer(w2v_model=Configuration['model']['embeddings'])
        self.load_label_descriptions()

    def load_label_descriptions(self):
        LOGGER.info('Load labels\' data')
        LOGGER.info('-------------------')

        # Load train dataset and count labels
        train_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    train_counts[concept] += 1

        train_concepts = set(list(train_counts))

        frequent, few = [], []
        for i, (label, count) in enumerate(train_counts.items()):
            if count > Configuration['sampling']['few_threshold']:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.json'))
        rest_concepts = set()
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    rest_concepts.add(concept)

        # Load label descriptors
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], '{}.json'.format(Configuration['task']['dataset']))) as file:
            data = json.load(file)
            none = set(data.keys())

        none = none.difference(train_concepts.union((rest_concepts)))
        parents = []
        for key, value in data.items():
            parents.extend(value['parents'])
        none = none.intersection(set(parents))

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))
        true_zero = deepcopy(zero)
        zero = zero + list(none)

        self.label_ids = dict()
        self.margins = [(0, len(frequent)+len(few)+len(true_zero))]
        k = 0
        for group in [frequent, few, zero]:
            self.margins.append((k, k+len(group)))
            for concept in group:
                self.label_ids[concept] = k
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent)+len(few)+len(true_zero))

        label_terms = []
        self.label_terms_text = []
        for i, (label, index) in enumerate(self.label_ids.items()):
            label_terms.append([Token(None, 0, 0, 'UNK', 'UNK', token)
                                for token in word_tokenize(data[label]['label']) if re.search('[A-Za-z]', token)])
            self.label_terms_text.append(data[label]['label'])

        self.label_terms_ids = self.vectorizer.vectorize_inputs(label_terms, max_sequence_size=Configuration['sampling']['max_label_size'], features=['word'])
        LOGGER.info('Labels shape:    {}'.format(len(label_terms)))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(true_zero)))

        # Compute label hierarchy depth and build labels' graph
        self.labels_graph = np.zeros((len(self.label_ids), len(self.label_ids)), dtype=np.float32)

        self.true_labels_cutoff = len(frequent)+len(few)+len(true_zero)

    def process_dataset(self, documents):

        samples = []
        targets = []
        for document in documents:
            if Configuration['model']['hierarchical']:
                samples.append(document.sentences)
            else:
                samples.append(document.tokens)
            targets.append(document.tags)

        del documents
        return samples, targets

    def encode_dataset(self, sequences, tags, max_sequences_size=None, max_sequence_size=None):
        max_sequences_size = min(max_sequences_size, Configuration['sampling']['max_sequences_size'])
        max_sequence_size = min(max_sequence_size, Configuration['sampling']['max_sequence_size'])
        if Configuration['model']['hierarchical']:
            samples = np.zeros((len(sequences), max_sequences_size,  max_sequence_size), dtype=np.int32)
            targets = np.zeros((len(sequences), self.true_labels_cutoff), dtype=np.int8)
            for i, (sub_sequences, document_tags) in enumerate(zip(sequences, tags)):
                sample = self.vectorizer.vectorize_inputs(sub_sequences[:max_sequences_size],
                                                          max_sequence_size=max_sequence_size,
                                                          features=['word'])
                samples[i, :len(sample[0])] = sample[0]
                for tag in document_tags:
                    if tag.name in list(self.label_ids)[:self.true_labels_cutoff]:
                        targets[i][self.label_ids[tag.name]] = 1
            samples = np.asarray(samples)
        else:
            samples = self.vectorizer.vectorize_inputs(sequences,
                                                       max_sequence_size=max_sequence_size,
                                                       features=['word'])

            if 'elmo' in Configuration['model']['token_encoding']:
                samples2 = self.vectorizer2.vectorize_inputs(sequences,
                                                             max_sequence_size=max_sequence_size,
                                                             features=['word'])
            targets = np.zeros((len(sequences), self.true_labels_cutoff), dtype=np.int8)
            for i, (document_tags) in enumerate(tags):
                for tag in document_tags:
                    if tag.name in list(self.label_ids)[:self.true_labels_cutoff]:
                        targets[i][self.label_ids[tag.name]] = 1

        del sequences, tags

        if 'bert' in Configuration['model']['architecture'].lower() and Configuration['model']['hierarchical']:
            return samples, targets
        elif 'distilbert' in Configuration['model']['architecture'].lower():
            return [samples[0], samples[1]], targets
        elif 'bert' in Configuration['model']['architecture'].lower():
            return [samples[0], samples[1], np.zeros(samples[0].shape, dtype=np.int32)], targets
        elif 'xlnet' in Configuration['model']['architecture'].lower():
            return samples[0], targets
        elif 'elmo' in Configuration['model']['token_encoding']:
            return [samples, samples2], targets

        return samples, targets

    def run_operation(self):
        LOGGER.info('\n---------------- {} Starting ----------------'.format(Configuration['task']['operation_mode']))

        if Configuration['task']['operation_mode'] == 'train':
            LOGGER.info('\n---------------- Simple training ----------------')
            LOGGER.info('\n---------------- Model Parameters ----------------')
            for param_name, value in Configuration['model'].items():
                LOGGER.info('\t{}: {}'.format(param_name, value))
            LOGGER.info('\n---------------- Training Parameters ----------------')
            for param_name, value in Configuration['training'].items():
                LOGGER.info('\t{}: {}'.format(param_name, value))
            LOGGER.info('\n---------------- Sampling Parameters ----------------')
            for param_name, value in Configuration['sampling'].items():
                LOGGER.info('\t{}: {}'.format(param_name, value))
            LOGGER.info('\n---------------- Evaluation Parameters ----------------')
            for param_name, value in Configuration['evaluation'].items():
                LOGGER.info('\t{}: {}'.format(param_name, value))

            self.train()
        else:
            raise Exception('Operation mode "{}" is not supported'.format(Configuration['task']['operation_mode']))

    def train(self):
        # Load training/validation data
        LOGGER.info('Load training/validation data')
        LOGGER.info('------------------------------')

        if Configuration['sampling']['load_from_disk']:
            train_filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.{}'.format(Configuration['task']['dataset_type'])))
            train_generator = SampleGeneratorFromDisk(train_filenames, experiment=self, batch_size=Configuration['training']['batch_size'])

            val_filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.{}'.format(Configuration['task']['dataset_type'])))
            val_generator = SampleGeneratorFromDisk(val_filenames, experiment=self, batch_size=Configuration['training']['batch_size'])
        else:
            documents = self.load_dataset('train')
            train_samples, train_tags = self.process_dataset(documents)
            train_generator = SampleGenerator(train_samples, train_tags, experiment=self, batch_size=Configuration['training']['batch_size'])

            documents = self.load_dataset('dev')
            val_samples, val_tags = self.process_dataset(documents)
            val_generator = SampleGenerator(val_samples, val_tags, experiment=self, batch_size=Configuration['training']['batch_size'])

        # Compile neural network
        LOGGER.info('Compile neural network')
        LOGGER.info('------------------------------')
        if 'label' in Configuration['model']['architecture'].lower():
            network = LabelDrivenClassification(self.label_terms_ids, self.labels_graph, self.true_labels_cutoff)
        else:
            print(self.true_labels_cutoff)
            network = DocumentClassification(self.true_labels_cutoff)

        network.compile(n_hidden_layers=Configuration['training']['n_hidden_layers'],
                        hidden_units_size=Configuration['training']['hidden_units_size'],
                        dropout_rate=Configuration['training']['dropout_rate'],
                        word_dropout_rate=Configuration['training']['word_dropout_rate'],
                        lr=Configuration['training']['lr'])

        network.summary(line_length=200, print_fn=LOGGER.info)

        with tempfile.NamedTemporaryFile(delete=True) as w_fd:
            weights_file = w_fd.name

        early_stopping = EarlyStopping(monitor='val_loss', patience=Configuration['training']['patience'], restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto',
                                           verbose=1, save_best_only=True, save_weights_only=True)

        # Fit model
        LOGGER.info('Fit model')
        LOGGER.info('-----------')
        start_time = time.time()
        fit_history = network.fit_generator(train_generator,
                                            validation_data=val_generator,
                                            workers=os.cpu_count(),
                                            epochs=Configuration['training']['epochs'],
                                            callbacks=[early_stopping, model_checkpoint])


        best_epoch = np.argmin(fit_history.history['val_loss']) + 1
        n_epochs = len(fit_history.history['val_loss'])
        val_loss_per_epoch = '- ' + ' '.join('-' if fit_history.history['val_loss'][i] < np.min(fit_history.history['val_loss'][:i])
                                             else '+' for i in range(1, len(fit_history.history['val_loss'])))
        LOGGER.info('\nBest epoch: {}/{}'.format(best_epoch, n_epochs))
        LOGGER.info('Val loss per epoch: {}\n'.format(val_loss_per_epoch))

        del train_generator, val_generator
        if Configuration['sampling']['load_from_disk']:
            val_generator = SampleGeneratorFromDisk(val_filenames, experiment=self, batch_size=Configuration['training']['batch_size'], shuffle=False)
        else:
            val_generator = SampleGenerator(val_samples, val_tags, experiment=self, batch_size=Configuration['training']['batch_size'], shuffle=False)

        LOGGER.info('Load valid data')
        LOGGER.info('------------------------------')
        self.calculate_performance(network=network, generator=val_generator)

        LOGGER.info('Load test data')
        LOGGER.info('------------------------------')

        if Configuration['sampling']['load_from_disk']:
            test_filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.{}'.format(Configuration['task']['dataset_type'])))
            test_generator = SampleGeneratorFromDisk(test_filenames, experiment=self, batch_size=Configuration['training']['batch_size'], shuffle=False)
        else:
            test_documents = self.load_dataset('test')
            test_samples, test_tags = self.process_dataset(test_documents)
            test_generator = SampleGenerator(test_samples, test_tags, experiment=self, batch_size=Configuration['training']['batch_size'], shuffle=False)

        self.calculate_performance(network=network, generator=test_generator, name=Configuration['task']['log_name'])

        total_time = time.time() - start_time
        LOGGER.info('\nTotal Training Time: {} secs'.format(total_time))

    def calculate_performance(self, network, generator, name=None):

        generator_size = len(generator.filenames) if Configuration['sampling']['load_from_disk'] else len(generator.data_samples)
        true_targets = np.zeros(shape=(generator_size, self.true_labels_cutoff), dtype=np.int8)
        predictions = np.zeros(shape=(generator_size, self.true_labels_cutoff), dtype=np.float32)

        if Configuration['model']['return_attention']:
            begin = 0
            for i, (x_batch, y_batch) in enumerate(generator):
                true_targets[begin:begin+len(y_batch)] = y_batch
                predictions[begin:begin+len(y_batch)] = network.predict(x_batch)[0]
                begin += len(y_batch)
        else:
            begin = 0
            for i, (x_batch, y_batch) in enumerate(generator):
                true_targets[begin:begin+len(y_batch)] = y_batch
                predictions[begin:begin+len(y_batch)] = network.predict(x_batch)
                begin += len(y_batch)

        if Configuration['evaluation']['save_predictions'] and name:
            with open(os.path.join(PREDICTIONS_DIR, '{}.predictions'.format(name)), 'wb') as file:
                pickle.dump(predictions, file, protocol=4)
            with open(os.path.join(PREDICTIONS_DIR, '{}.gold'.format(name)), 'wb') as file:
                pickle.dump(true_targets, file, protocol=4)

        print('MADE PREDICTIONS')

        pred_targets = probas_to_classes(predictions)

        report_statistics = {}
        for freq in ['Overall', 'Frequent', 'Few', 'Zero']:
            report_statistics[freq] = {}
            report_statistics[freq]['R-Precision'] = 0
            for average_type in ['micro', 'macro', 'weighted']:
                report_statistics[freq][average_type] = {}
                for metric in ['P', 'R', 'F1']:
                    report_statistics[freq][average_type][metric] = 0

            for metric in ['R@', 'P@', 'RP@', 'NDCG@']:
                report_statistics[freq][metric] = {}
                for i in range(1, Configuration['evaluation']['evaluation@k'] + 1):
                    report_statistics[freq][metric][i] = 0

        template = 'R@{:<2d} : {:1.3f}   P@{:<2d} : {:1.3f}   RP@{:<2d} : {:1.3f}   NDCG@{:<2d} : {:1.3f}'

        # Overall
        for labels_range, frequency, message in zip(self.margins, ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall',
                                                     'Frequent Labels (>{} Occurrences in train set)'.format(Configuration['sampling']['few_threshold']),
                                                     'Few-shot (<={} Occurrences in train set)'.format(Configuration['sampling']['few_threshold']),
                                                     'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            LOGGER.info('\n' + message)
            LOGGER.info('--------------------------------------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted'] if Configuration['evaluation']['advanced_mode'] else ['micro']:
                p = report_statistics[frequency][average_type]['P'] = precision_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                r = report_statistics[frequency][average_type]['R'] = recall_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                f1 = report_statistics[frequency][average_type]['F1'] = f1_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                support = np.sum(true_targets[:, start:end])
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}   Support: {}'.format(average_type, p, r, f1, support))

            r_precision = report_statistics[frequency]['R-Precision'] = mean_rprecision(true_targets[:, start:end], predictions[:, start:end])
            LOGGER.info('R-Precision: {:1.4f}'.format(r_precision))
            if Configuration['evaluation']['advanced_mode']:
                for i in range(1, Configuration['evaluation']['evaluation@k'] + 1):
                    r_k = report_statistics[frequency]['R@'][i] = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    p_k = report_statistics[frequency]['P@'][i] = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    rp_k = report_statistics[frequency]['RP@'][i] = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    ndcg_k = report_statistics[frequency]['NDCG@'][i] = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
                    LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
            LOGGER.info('--------------------------------------------------------------------------------')

        return report_statistics

    def average_report(self, average_statistics):
        LOGGER.info('\n---------- Average Classification Results ----------\n')

        template = 'R@{} : {}   P@{} : {}   RP@{} : {}   NDCG@{} : {}'

        for dataset in ['validation', 'test']:
            LOGGER.info('\n{} Classification Report\n'.format(dataset.capitalize()))
            for frequency in ['Overall', 'Frequent', 'Few', 'Zero']:
                LOGGER.info(frequency)
                LOGGER.info('----------------------------------------------------')
                for average_type in ['micro', 'macro', 'weighted']:
                    LOGGER.info('{:8} - Precision: {}   Recall: {}   F1: {}'.format(
                        average_type,
                        self.metric_to_str(average_statistics[dataset][frequency][average_type]['P']),
                        self.metric_to_str(average_statistics[dataset][frequency][average_type]['R']),
                        self.metric_to_str(average_statistics[dataset][frequency][average_type]['F1'])))

                for i in range(1, Configuration['evaluation']['evaluation@k'] + 1):
                    LOGGER.info(template.format(i, self.metric_to_str(average_statistics[dataset][frequency]['R@'][i]),
                                                i, self.metric_to_str(average_statistics[dataset][frequency]['P@'][i]),
                                                i, self.metric_to_str(average_statistics[dataset][frequency]['RP@'][i]),
                                                i, self.metric_to_str(average_statistics[dataset][frequency]['NDCG@'][i])))
                LOGGER.info('----------------------------------------------------')

    def metric_to_str(self, metric):
        return '{:.3f} (std={:.4f} se={:.4f} rel_se={:.4f})'.format(metric['mean'], metric['std'], metric['st_error'], metric['rel_st_error'])

    def calculate_std_standard_error(self, values):
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        standard_error = std / np.sqrt(len(values))
        relative_standard_error = standard_error / (mean + np.finfo(float).eps)

        return {'mean': mean, 'std': std, 'st_error': standard_error, 'rel_st_error': relative_standard_error}


class SampleGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, samples, targets, experiment, batch_size=32, shuffle=True):
        """Initialization"""
        self.data_samples = samples
        self.targets = targets
        self.batch_size = batch_size
        self.indices = np.arange(len(samples))
        self.experiment = experiment
        self.shuffle = shuffle

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.data_samples) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of batch's sequences + targets
        samples = [self.data_samples[k] for k in indices]
        targets = [self.targets[k] for k in indices]
        if Configuration['sampling']['dynamic_batching']:
            if Configuration['model']['hierarchical']:
                max_sequences_size = max([len(sample) for sample in samples])
                max_sequence_size = max([len(sentence) for sample in samples for sentence in sample])
            else:
                max_sequences_size = 0
                max_sequence_size = max([len(sample) for sample in samples])
        else:
            max_sequences_size = Configuration['sampling']['max_sequences_size']
            max_sequence_size = Configuration['sampling']['max_sequence_size']

        # Vectorize inputs (x,y)
        x_batch, y_batch = self.experiment.encode_dataset(sequences=samples, tags=targets,
                                                          max_sequences_size=max_sequences_size,
                                                          max_sequence_size=max_sequence_size)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


class SampleGeneratorFromDisk(Sequence):
    'Generates data for Keras'

    def __init__(self, filenames, experiment, batch_size=32, shuffle=True):
        """Initialization"""
        self.filenames = filenames
        self.batch_size = batch_size
        self.indices = np.arange(len(filenames))
        self.experiment = experiment
        self.shuffle = shuffle
        self.loader = LoaderFactory.get_loader(Configuration['task']['dataset_type'])

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of batch's sequences + targets
        filenames = [self.filenames[k] for k in indices]

        documents = [self.loader.read_file(filename) for filename in filenames]
        samples, targets = self.experiment.process_dataset(documents)

        if Configuration['sampling']['dynamic_batching']:
            if Configuration['model']['hierarchical']:
                max_sequences_size = max([len(sample) for sample in samples])
                max_sequence_size = max([len(sentence) for sample in samples for sentence in sample])
            else:
                max_sequences_size = 0
                max_sequence_size = max([len(sample) for sample in samples])
        else:
            max_sequences_size = Configuration['sampling']['max_sequences_size']
            max_sequence_size = Configuration['sampling']['max_sequence_size']

        # Vectorize inputs (x,y)
        x_batch, y_batch = self.experiment.encode_dataset(sequences=samples, tags=targets,
                                                          max_sequences_size=max_sequences_size,
                                                          max_sequence_size=max_sequence_size)

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
