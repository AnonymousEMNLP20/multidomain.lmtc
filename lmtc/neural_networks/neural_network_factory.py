import importlib


class NeuralNetworkFactory:
    @staticmethod
    def create_model(architecture):
        return getattr(importlib.import_module('nlp_research.neural_networks.task_specific_networks.{0}'.format(architecture.lower())),
                       architecture)()
