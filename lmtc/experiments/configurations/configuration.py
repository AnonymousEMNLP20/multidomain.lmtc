import json
import os
import logging
from time import gmtime, strftime
from keras import backend as K
from . import CONFIG_DIR
from nlp_research.data import LOGGING_DIR
from nlp_research.common.text_preprocessor import Tagger

LOGGER = logging.getLogger(__name__)

parameters = {}


class ParameterStore(type):
    def __getitem__(cls, key: str):
        global parameters
        return parameters[key]

    def __setitem__(cls, key, value):
        global parameters
        parameters[key] = value


class Configuration(object, metaclass=ParameterStore):

    @staticmethod
    def configure(task_type, task_name):
        global parameters

        if task_type not in ['sequence_tagging', 'xmtc', 'embeddings']:
            raise Exception('Task type "{}" is not supported'.format(task_type))

        if task_name:
            config_file = os.path.join(CONFIG_DIR, task_type, '{}.json'.format(task_name))
        else:
            config_file = os.path.join(CONFIG_DIR, task_type, '{}.json'.format(task_type))

        if not os.path.exists(config_file):
            raise Exception('Configuration file "{}" does not exists'.format(config_file))

        with open(config_file, 'r') as f:
            parameters = json.load(f)

        # Setup Logging
        timestamp = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        log_name = '{}_{}_{}_{}_{}'.format(Configuration['task']['dataset'].upper(),
                                           'HIERARCHICAL' if Configuration['model']['hierarchical'] else 'FLAT',
                                           Configuration['model']['architecture'].upper(), parameters['task']['operation_mode'], timestamp)

        # Check if resume optimization is enabled
        if parameters['task']['operation_mode'] == 'hyperopt':
            trials_log = parameters['hyper_optimization']['log_name']
            log_folder = os.path.join(LOGGING_DIR, 'hyperopt')
            trials_folder = os.path.join(LOGGING_DIR, 'hyperopt', 'trials')
            if trials_log:
                trials_log = '{}.trials'.format(trials_log)
                if os.path.isfile(os.path.join(trials_folder, trials_log)):
                    log_name = parameters['hyper_optimization']['log_name']
                    if not os.path.isfile(os.path.join(log_folder, log_name + '.txt')):
                        raise Exception('Hyperopt optimization resume failed. Could not find "{}{}{}" file'.format(
                            log_folder, os.sep, log_name + '.txt'))
                else:
                    raise Exception('Hyperopt optimization resume failed. Could not find "{}{}{}" file'.format(
                        trials_folder, os.sep, trials_log))
            else:
                trials_log = '{}_{}_{}_{}_{}.trials'.format(
                    Configuration['task']['dataset'].upper(), 'HIERARCHICAL' if Configuration['model']['hierarchical'] else 'FLAT',
                    Configuration['model']['architecture'].upper(), parameters['task']['operation_mode'], timestamp)

            parameters['hyper_optimization']['log_name'] = os.path.join(trials_folder, trials_log)

        parameters['task']['log_name'] = log_name
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(LOGGING_DIR, parameters['task']['operation_mode'].lower(), log_name + '.txt'),
                            filemode='a')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        # Check if GPU acceleration is available
        if not len(K.tensorflow_backend._get_available_gpus()):
            parameters['task']['cuDNN'] = False

        # Initialize spacy tagger
        parameters['sampling']['tagger'] = Tagger(lang=parameters['task']['task_language'])

    def search_space(self) -> dict:
        """
            Property that builds the hyper parameter search space. It practically returns the 'hyper_optimization'
            property with some adjustments.
        """
        search_space = {
            'n_hidden_layers': parameters['hyper_optimization']['n_hidden_layers'],
            'hidden_units_size': parameters['hyper_optimization']['hidden_units_size'],
            'batch_size': parameters['hyper_optimization']['batch_size'],
            'dropout_rate': parameters['hyper_optimization']['dropout_rate'],
            'word_dropout_rate': parameters['hyper_optimization']['word_dropout_rate'],
            'learning_rate': parameters['hyper_optimization']['learning_rate']
        }

        return search_space

    @classmethod
    def __getitem__(cls, item: str):
        global parameters
        return parameters[item]
