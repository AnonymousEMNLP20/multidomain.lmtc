import glob
import os
import json
from collections import Counter
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import numpy as np
import parabel  # pip install git+https://github.com/tomtung/parabel-rs.git#subdirectory=python -v
from sklearn.metrics import precision_score, recall_score, f1_score
from gensim.models import KeyedVectors

# Global Parameters
FEW_THRESHOLD = 50
dataset_folder = '/Users/kiddo/PycharmProjects/nlp.research/nlp_research/data/datasets/eurovoc_en'
dataset_name = 'eurovoc'

embeddings = KeyedVectors.load_word2vec_format('/Users/kiddo/PycharmProjects/nlp.research/nlp_research/data/vectors/word2vec/en/glove.6B.200d.bin', binary=True)


def probas_to_classes(probabilities):
    return (probabilities > 0.5).astype('int8')


def mean_precision_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_precision_score(y_t, y_s, k=k))

    return np.mean(p_ks)


def mean_recall_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)


def mean_ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    Mean NDCG @k : float
    """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)


def mean_rprecision_k(y_true, y_score, k=10):
    """Mean r-precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_rprecision_score(y_t, y_s, k=k))

    return np.mean(p_ks)


def mean_rprecision(y_true, y_score):
    """Mean r-precision
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    Returns
    -------
    mean r-precision : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(default_rprecision_score(y_t, y_s))

    return np.mean(p_ks)


def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevant levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevant levels.")

    pos_label = unique_y[1]

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / k


def default_rprecision_score(y_true, y_score):
    """R-Precision
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        raise ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevant levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:n_pos])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by n_pos such that the best achievable score is always 1.0.
    return float(n_relevant) / n_pos


def ranking_rprecision_score(y_true, y_score, k=10):
    """R-Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevant levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(k, n_pos)


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevant levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


class Parabel():

    def __init__(self):
        print('Load labels\' data')
        print('-------------------')
        # Find labels and frequency groups
        train_files = glob.glob(os.path.join(dataset_folder, 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    train_counts[concept] += 1

        train_concepts = set(list(train_counts))

        frequent, few = [], []
        for i, (label, count) in enumerate(train_counts.items()):
            if count > FEW_THRESHOLD:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(dataset_folder, 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(dataset_folder, 'test', '*.json'))
        rest_concepts = set()
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    rest_concepts.add(concept)

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))

        self.label_ids = dict()
        self.margins = [(0, len(frequent) + len(few) + len(zero))]
        k = 0
        for group in [frequent, few, zero]:
            self.margins.append((k, k + len(group)))
            for concept in group:
                self.label_ids[concept] = k
                k += 1

        print('Labels\' list ready with {} labels'.format(len(self.label_ids)))

    def build_datasets(self):
        # Configure TFIDF Vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), tokenizer=word_tokenize, use_idf=True, stop_words='english', lowercase=True,
                                     max_features=400000)
        documents = []
        for folder in ['train', 'dev']:

            filenames = glob.glob(os.path.join(dataset_folder, folder, '*.json'))
            for i, filename in enumerate(filenames):
                with open(filename) as file:
                    data = json.load(file)

                    documents.append(data['header'] + '\n' + data['recitals'] + '\n'.join(data['main_body']) + '\n' + data['attachments'])

        vectorizer = vectorizer.fit(documents)
        vocab = vectorizer.get_feature_names()

        print('TFIDF Vectorizer configured...')

        # Create datasets in txt format
        for folder in ['train', 'dev', 'test']:
            x = []
            filenames = glob.glob(os.path.join(dataset_folder, folder, '*.json'))
            tags = []
            for i, filename in tqdm.tqdm(enumerate(filenames)):
                with open(filename) as file:
                    data = json.load(file)

                x.append(data['header'] + '\n' + data['recitals'] + '\n'.join(data['main_body']) + '\n' + data['attachments'])
                sample_tags = []
                for j, tag in enumerate(data['concepts']):
                    if tag in self.label_ids:
                        sample_tags.append(self.label_ids[tag])

                tags.append(sample_tags)

            x = vectorizer.transform(x)

            filename = os.path.join(dataset_folder, dataset_folder.split('/')[-1] + '_' + folder + '.txt')

            file = open(filename, 'w')

            file.write('{} {} {}\n'.format(len(filenames), 400000, len(self.label_ids)))

            for sample, targets in zip(x, tags):
                labels = ','.join([str(target) for target in targets])
                features = ' '
                vector = np.zeros((200,), dtype=np.flot32)
                tfidf_sum = 0
                for index, value in zip(sample.indices, sample.data):
                    if vocab[index] in embeddings:
                        tfidf_sum += value
                        vector += value * embeddings[vocab[index]]
                vector /= tfidf_sum
                for k, value in enumerate(vector):
                    features += '{}:{:.6f} '.format(k, value)
                row = labels + features[:-1] + '\n'
                file.write(row)
            file.close()
            print('{} dataset ready in {}...'.format(folder.title(), filename))

    def train(self):
        # Train
        trainer = parabel.Trainer()
        model = trainer.train_on_data(os.path.join(dataset_folder, dataset_folder.split('/')[-1] + '_train.txt'))
        # Serialize & de-serialize
        model.save("model.bin")

    def evaluate(self):
        model = parabel.Model.load("model.bin")
        print('Model loaded...')
        for folder in ['dev', 'test']:
            def load_dataset(data_path):
                with open(data_path) as file:
                    lines = file.readlines()
                    predictions = np.zeros((len(lines) - 1, len(self.label_ids)), dtype=np.float32)
                    targets = np.zeros((len(lines) - 1, len(self.label_ids)), dtype=np.int8)
                    for i, line in tqdm.tqdm(enumerate(lines[1:])):
                        x = []
                        features = line.split(' ')[1:]
                        labels = line.split(' ')[0]
                        for label in labels.split(','):
                            targets[i][int(label)] = 1
                        for feature in features:
                            key, value = feature.split(':')
                            x.append((int(key), float(value)))
                        label_score_pairs = model.predict(x, top_k=len(self.label_ids))
                        for (key, value) in label_score_pairs:
                            predictions[i][key] = value
                return predictions, targets

            # Predict
            dataset_filename = os.path.join(dataset_folder, dataset_folder.split('/')[-1] + '_' + folder + '.txt')
            predictions, true_targets = load_dataset(dataset_filename)
            print(predictions.shape)
            print(true_targets.shape)
            # Calculate performance
            print('----- {} Classification Results -----'.format(folder.title()))
            calculate_perfomance(predictions, true_targets, self.margins)


def calculate_perfomance(predictions, true_targets, margins):
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
            for i in range(1, 20):
                report_statistics[freq][metric][i] = 0

    template = 'R@{:<2d} : {:1.3f}   P@{:<2d} : {:1.3f}   RP@{:<2d} : {:1.3f}   NDCG@{:<2d} : {:1.3f}'

    # Overall
    for labels_range, frequency, message in zip(margins, ['Overall', 'Frequent', 'Few', 'Zero'],
                                                ['Overall',
                                                 'Frequent Labels (>{} Occurrences in train set)'.format(FEW_THRESHOLD),
                                                 'Few-shot (<={} Occurrences in train set)'.format(FEW_THRESHOLD),
                                                 'Zero-shot (No Occurrences in train set)']):
        start, end = labels_range
        print('\n' + message)
        print('--------------------------------------------------------------------------------')
        for average_type in ['micro', 'macro', 'weighted']:
            p = report_statistics[frequency][average_type]['P'] = precision_score(true_targets[:, start:end], pred_targets[:, start:end],
                                                                                  average=average_type)
            r = report_statistics[frequency][average_type]['R'] = recall_score(true_targets[:, start:end], pred_targets[:, start:end],
                                                                               average=average_type)
            f1 = report_statistics[frequency][average_type]['F1'] = f1_score(true_targets[:, start:end], pred_targets[:, start:end],
                                                                             average=average_type)
            support = np.sum(true_targets[:, start:end])
            print('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}   Support: {}'.format(average_type, p, r, f1, support))

        r_precision = report_statistics[frequency]['R-Precision'] = mean_rprecision(true_targets[:, start:end], predictions[:, start:end])
        print('R-Precision: {:1.4f}'.format(r_precision))
        for i in range(1, 21):
            r_k = report_statistics[frequency]['R@'][i] = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
            p_k = report_statistics[frequency]['P@'][i] = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
            rp_k = report_statistics[frequency]['RP@'][i] = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
            ndcg_k = report_statistics[frequency]['NDCG@'][i] = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
            print(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
        print('--------------------------------------------------------------------------------')

    return report_statistics


experiment = Parabel()
experiment.build_datasets()
experiment.train()
experiment.evaluate()
