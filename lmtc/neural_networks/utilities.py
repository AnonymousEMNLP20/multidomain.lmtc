from lmtc.experiments.configurations.configuration import Configuration


def probas_to_classes(probabilities):
    if probabilities.shape[-1] > 1:
        if Configuration['task']['decision_type'] == 'multi_label':
            return (probabilities > 0.5).astype('int8')
        else:
            return probabilities.argmax(axis=-1)
