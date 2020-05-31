import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
import click
from lmtc.experiments.configurations.configuration import Configuration
from lmtc.experiments import Experiment
cli = click.Group()


@cli.command()
@click.option('--task_name', default='eurovoc_classification')
def run(task_type, task_name):
    Configuration.configure(task_type, task_name)
    experiment = Experiment()
    experiment.train()


if __name__ == '__main__':
    run()
