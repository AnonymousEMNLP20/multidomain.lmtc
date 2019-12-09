import glob
import json
import numpy as np
import matplotlib
import tqdm

from collections import Counter
matplotlib.use('agg')
from matplotlib import pyplot as plt


def count_lengths_and_labels(filenames):
    documents_lengths = []
    labels_per_doc = []
    labels_counter = Counter()
    with tqdm.tqdm(total=len(filenames), ncols=120) as progress_bar:
        for filename in filenames:
            with open(file=filename, mode='r') as fin:
                datum = json.load(fp=fin)
                documents_lengths.append(len(datum['tokenized_text'].split()))
                for concept in datum['concepts']:
                    labels_counter[concept] += 1
                labels_per_doc.append(len(datum['concepts']))
            progress_bar.update(n=1)
    return sorted(documents_lengths), sorted(labels_per_doc), labels_counter


def count_lengths_by_section(filenames):
    documents_lengths = []
    with tqdm.tqdm(total=len(filenames), ncols=120) as progress_bar:
        for filename in filenames:
            with open(file=filename, mode='r') as fin:
                datum = json.load(fp=fin)
                documents_lengths.append(len(datum['sections']))
            progress_bar.update(n=1)
    return sorted(documents_lengths)


def count_lengths_by_section_words(filenames):
    documents_lengths = []
    with tqdm.tqdm(total=len(filenames), ncols=120) as progress_bar:
        for filename in filenames:
            with open(file=filename, mode='r') as fin:
                datum = json.load(fp=fin)
                for section in datum['tokenized_sections']:
                    documents_lengths.append(len(section.split()))
            progress_bar.update(n=1)
    return sorted(documents_lengths)


def print_statistics(documents_lengths, labels_per_doc, intervals_nb=20):
    for i in range(intervals_nb):
        interval = round((((i + 1) / intervals_nb) * len(documents_lengths)))
        print(
            'DATASET ({0:3d}%): DOCS: {1:7d} DOC LENGTH: MAX: {2:5d} AVG: {3:7.2f} LABELS/DOC: MAX: {4:3d} MEAN: {5:.2f}'.format(
                (i + 1) * 5,
                interval,
                np.max(documents_lengths[:interval]),
                np.mean(documents_lengths[:interval]),
                np.max(labels_per_doc[:interval]),
                np.mean(labels_per_doc[:interval])
            )
        )


def print_statistics_by_section(documents_lengths, intervals_nb=20):
    for i in range(intervals_nb):
        interval = round((((i + 1) / intervals_nb) * len(documents_lengths)))
        print(
            'DATASET ({0:3d}%): DOCS: {1:7d} DOC LENGTH: MAX: {2:5d} AVG: {3:7.2f}'.format(
                (i + 1) * 5,
                interval,
                np.max(documents_lengths[:interval]),
                np.mean(documents_lengths[:interval])
            )
        )


def plot_length_distribution(documents_lengths, fig_path):
    plt.clf()
    fig = plt.figure(figsize=(10, 6))
    plt.hist(documents_lengths, bins=100)
    plt.title('Documents')
    plt.xlabel('# tokens')
    plt.ylabel('Frequency')
    plt.savefig(fig_path)
    plt.close(fig=fig)


def plot_labels_distribution(labels_counter, fig_path):
    plt.clf()
    fig = plt.figure(figsize=(10, 6))
    plt.hist([count if count <= 500 else 501 for label, count in labels_counter.most_common()], bins=100, range=(0, 501))
    plt.title('Labels')
    plt.xlabel('# labels')
    plt.ylabel('# documents')
    plt.savefig(fig_path)
    plt.close(fig=fig)


def print_few(labels_counter):
    with open(file='/home/cave-of-time/jaga/MIMIC/dataset/icd9_codes.json', mode='r') as fin:
        icd9_codes = json.load(fp=fin)
    icd9_codes_nb = len(icd9_codes)
    present = len([count for concept, count in labels_counter.most_common()])
    print('All MIMIC ICD9 Codes                          : {:5} (100.00%)'.format(icd9_codes_nb))
    print('MIMIC ICD9 Codes that appear at least 1 time  : {:5} ({:2.2f}%)'.format(present, present * 100 / icd9_codes_nb))

    appearances = [(threshold, len([count for concept, count in labels_counter.most_common() if count >= threshold]))
                   for threshold in [10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]]
    for threshold, min_c in appearances:
        print(
            'MIMIC ICD9 Codes that appear at least {:3} times : {:5} ({:2.2f}%)'.format(
                threshold,
                min_c,
                min_c * 100 / icd9_codes_nb
            )
        )


def print_zero(train_filenames, development_filenames, test_filenames):
    train_concepts = set()
    development_concepts = set()
    test_concepts = set()
    with tqdm.tqdm(total=len(train_filenames), ncols=120) as progress_bar:
        for filename in train_filenames:
            with open(file=filename, mode='r') as fin:
                datum = json.load(fp=fin)
                train_concepts.update(datum['concepts'])
            progress_bar.update(n=1)

    with tqdm.tqdm(total=len(development_filenames), ncols=120) as progress_bar:
        for filename in development_filenames:
            with open(file=filename, mode='r') as fin:
                datum = json.load(fp=fin)
                development_concepts.update(datum['concepts'])
            progress_bar.update(n=1)

    with tqdm.tqdm(total=len(test_filenames), ncols=120) as progress_bar:
        for filename in test_filenames:
            with open(file=filename, mode='r') as fin:
                datum = json.load(fp=fin)
                test_concepts.update(datum['concepts'])
            progress_bar.update(n=1)

    print('Development zero-shot concepts: {}'.format(len(development_concepts.difference(train_concepts))))
    print('Test zero-shot concepts: {}'.format(len(test_concepts.difference(train_concepts))))


def print_icd9_statistics():
    with open(file='/home/cave-of-time/jaga/MIMIC/dataset/mimic.json', mode='r') as fin:
        icd9_codes = json.load(fp=fin)
    lengths = np.array([len(icd9_codes[icd9_code]['label'].split()) for icd9_code in icd9_codes])
    print('Max icd9 description length: {0:.2f}'.format(np.max(lengths)))
    print('Min icd9 description length: {0:.2f}'.format(np.min(lengths)))
    print('Avg icd9 description length: {0:.2f}'.format(np.mean(lengths)))


def main():
    train_filenames = glob.glob(pathname='/home/cave-of-time/jaga/MIMIC/dataset/train/*')
    development_filenames = glob.glob(pathname='/home/cave-of-time/jaga/MIMIC/dataset/development/*')
    test_filenames = glob.glob(pathname='/home/cave-of-time/jaga/MIMIC/dataset/test/*')

    all_filenames = train_filenames + development_filenames + test_filenames

    # documents_lengths, labels_per_doc, labels_counter = count_lengths_and_labels(filenames=train_filenames)
    # print_statistics(documents_lengths=documents_lengths, labels_per_doc=labels_per_doc)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='train_doc_length.png')
    # plot_labels_distribution(labels_counter=labels_counter, fig_path='train_labels.png')
    #
    # documents_lengths, labels_per_doc, labels_counter = count_lengths_and_labels(filenames=development_filenames)
    # print_statistics(documents_lengths=documents_lengths, labels_per_doc=labels_per_doc)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='development_doc_length.png')
    # plot_labels_distribution(labels_counter=labels_counter, fig_path='development_labels.png')
    #
    # documents_lengths, labels_per_doc, labels_counter = count_lengths_and_labels(filenames=test_filenames)
    # print_statistics(documents_lengths=documents_lengths, labels_per_doc=labels_per_doc)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='test_doc_length.png')
    # plot_labels_distribution(labels_counter=labels_counter, fig_path='test_labels.png')

    # print('Tokens per summary...')
    # print('---------------------')
    #
    # documents_lengths, labels_per_doc, labels_counter = count_lengths_and_labels(filenames=all_filenames)
    # print_statistics(documents_lengths=documents_lengths, labels_per_doc=labels_per_doc)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='all_doc_length.png')
    # plot_labels_distribution(labels_counter=labels_counter, fig_path='all_labels.png')
    #
    # print_few(labels_counter=labels_counter)
    # print_zero(
    #     train_filenames=train_filenames,
    #     development_filenames=development_filenames,
    #     test_filenames=test_filenames
    # )

    print_icd9_statistics()

    #####################################

    # documents_lengths = count_lengths_by_section(filenames=train_filenames)
    # print_statistics_by_sentence(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='train_doc_length_by_sentence.png')
    #
    # documents_lengths = count_lengths_by_section(filenames=development_filenames)
    # print_statistics_by_sentence(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='development_doc_length_by_sentence.png')
    #
    # documents_lengths = count_lengths_by_section(filenames=test_filenames)
    # print_statistics_by_sentence(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='test_doc_length_by_sentence.png')

    # print('Sections per summary...')
    # print('---------------------')
    #
    # documents_lengths = count_lengths_by_section(filenames=all_filenames)
    # print_statistics_by_section(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='all_doc_length_by_section.png')

    #####################################

    # documents_lengths = count_lengths_by_section_words(filenames=train_filenames)
    # print_statistics_by_section(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='train_doc_length_by_sentence_words.png')
    #
    # documents_lengths = count_lengths_by_section_words(filenames=development_filenames)
    # print_statistics_by_section(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='development_doc_length_by_sentence_words.png')
    #
    # documents_lengths = count_lengths_by_section_words(filenames=test_filenames)
    # print_statistics_by_section(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='test_doc_length_by_sentence_words.png')

    # print('Tokens per section...')
    # print('---------------------')
    #
    # documents_lengths = count_lengths_by_section_words(filenames=all_filenames)
    # print_statistics_by_section(documents_lengths=documents_lengths)
    # plot_length_distribution(documents_lengths=documents_lengths, fig_path='all_doc_length_by_section_words.png')


if __name__ == '__main__':
    main()
