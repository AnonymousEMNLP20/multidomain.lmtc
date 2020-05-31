import glob
import os
import tqdm

from lmtc.data import DATA_SET_DIR
from lmtc.experiments.configurations.configuration import Configuration
from lmtc.loaders.base import LoaderFactory


class Experiment(object):

    def __init__(self):
        pass

    def load_dataset(self, dataset_name):
        """
        Load dataset and return list of documents
        :param dataset_name: the name of the dataset
        :return: list of Document objects
        """
        filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], dataset_name, '*.{}'.format(Configuration['task']['dataset_type'])))
        loader = LoaderFactory.get_loader(Configuration['task']['dataset_type'])

        documents = []
        for filename in tqdm.tqdm(sorted(filenames)):
            documents.append(loader.read_file(filename))

        return documents

    def process_dataset(self, documents):
        """
        Process dataset documents (samples) and create targets
        :param documents: list of Document objects
        :return: samples, targets
        """
        raise NotImplementedError

    def encode_dataset(self, sequences, targets):
        """
        Encode dataset samples feature matrices and targets in one-hot vectors
        :param sequences:
        :param targets:
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Train pipeline
        :return: None
        """
        raise NotImplementedError

    def hyper_optimization(self):
        """
        Hyper Optimization pipeline
        :return: None
        """
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def plot_histograms(self):
        raise NotImplementedError
