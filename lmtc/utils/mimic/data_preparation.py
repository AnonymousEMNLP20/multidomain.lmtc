import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import tqdm

from gensim.models import KeyedVectors
from typing import Dict, List, Union

section_pattern = re.compile('\\n\\n+')


def bioclean(text):
    return ' '.join(
        re.sub(
            '[.,?;*!%^&_+():-\[\]{}]',
            '',
            text.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()
        ).split()
    )


def load_df(csv_file_path: str, usecols: List[str], dtype: Dict, icd9_code_prefix: Union[str, None] = None):
    with open(file=csv_file_path, mode='r') as fin:
        df = pd.read_csv(
            filepath_or_buffer=fin,
            usecols=usecols,
            dtype=dtype
        )
    if 'ICD9_CODE' in usecols:
        df.ICD9_CODE = icd9_code_prefix + '-' + df.ICD9_CODE
    return df


def icd9_codes_df_to_dict(icd9_codes_df: pd.DataFrame):
    return {
        icd9_code: {
            'concept_id': icd9_code,
            'label': bioclean(long_title),
            'parents': []
        }
        for icd9_code, long_title in zip(icd9_codes_df.ICD9_CODE, icd9_codes_df.LONG_TITLE)
    }


def discharge_summaries_df_to_list(discharge_summaries_df: pd.DataFrame):
    discharge_summaries_groups = discharge_summaries_df.sort_values(
        by=['CHARTDATE', 'HADM_ID']
    ).groupby(
        by='HADM_ID'
    ).aggregate(lambda x: list(set(x)))
    return [
        {
            'summary_id': index,
            'concepts': discharge_summary.ICD9_CODE,
            'text': '\n\n'.join(discharge_summary.TEXT),
            'tokenized_text': bioclean('\n\n'.join(discharge_summary.TEXT)),
            'sections': section_pattern.split('\n\n'.join(discharge_summary.TEXT)),
            'tokenized_sections': [bioclean(section) for section in section_pattern.split('\n\n'.join(discharge_summary.TEXT))]

        } for index, discharge_summary in discharge_summaries_groups.iterrows()
    ]


def split_data(data: List[Dict], seed, train_percentage, development_percentage):
    random.seed(seed)
    random.shuffle(data)
    train_index = round(train_percentage * len(data))
    development_index = round((train_percentage + development_percentage) * len(data))
    return data[:train_index], data[train_index:development_index], data[development_index:]


def refactor_embeddings(embeddings_binary_file, dimension=200):
    embeddings_text_file = embeddings_binary_file.replace('.bin', '.txt')
    print('Loaded embeddings from .bin file...')
    vectors = KeyedVectors.load_word2vec_format(embeddings_binary_file, binary=True)
    vectors.save_word2vec_format(embeddings_text_file, binary=False)
    print('Saved embeddings to .txt file...')
    with open(file=embeddings_text_file, mode='r') as fin:
        embeddings = fin.readlines()[1:]

    print('Loaded embeddings from .txt file...')
    with open(file=embeddings_text_file, mode='w') as fout:
        print('{0:d} 200'.format(len(embeddings) + 1), file=fout)
        print('UNKNOWN {0:s}'.format('{0:.1f} '.format(np.ceil(np.max(vectors.vectors) + 1)) * dimension).strip(), file=fout)
        with tqdm.tqdm(total=len(embeddings), ncols=120) as progress_bar:
            for embedding in embeddings:
                print(embedding.strip(), file=fout)
                progress_bar.update(n=1)
        print('', file=fout)
    print('Saved refactored embeddings to .txt file...')
    vectors = KeyedVectors.load_word2vec_format(embeddings_text_file, binary=False)
    print('Loaded refactored embeddings from .txt file...')
    vectors.save_word2vec_format(embeddings_binary_file, binary=True)
    print('saved refactored embeddings to .bin file...')


def get_embeddings_indices(embeddings_binary_file, embeddings_indices_file):
    vectors = KeyedVectors.load_word2vec_format(embeddings_binary_file, binary=True)
    word2index = {word: index for index, word in enumerate(vectors.index2word)}
    with open(file=embeddings_indices_file, mode='wb') as fout:
        pickle.dump(obj=word2index, file=fout)


def update_icd9_codes(icd9_json_file, rios_icd9_descriptions_file, rios_icd9_parent_child_relations_file):
    with open(file=icd9_json_file, mode='r') as fin:
        icd9_codes = json.load(fin)

    with open(file=rios_icd9_parent_child_relations_file, mode='r') as fin:
        parent2child = {line.strip().split('\t')[1]: line.strip().split('\t')[0] for line in fin}
    with open(file=rios_icd9_descriptions_file, mode='r') as fin:
        description2code = {bioclean(line.strip().split('\t')[1]): line.strip().split('\t')[0] for line in fin}

    for code in icd9_codes:
        icd9_codes[code]['parents'].add(parent2child[description2code])


def convert_idc9(icd9_code, is_disease):
    for i in []:
        pass


def main():
    diagnoses_icd9_codes_df = load_df(
        csv_file_path='/home/cave-of-time/jaga/MIMIC/D_ICD_DIAGNOSES.csv',
        usecols=['ICD9_CODE', 'LONG_TITLE'],
        dtype={'ICD9_CODE': str, 'LONG_TITLE': str},
        icd9_code_prefix='D'
    )
    procedures_icd9_codes_df = load_df(
        csv_file_path='/home/cave-of-time/jaga/MIMIC/D_ICD_PROCEDURES.csv',
        usecols=['ICD9_CODE', 'LONG_TITLE'],
        dtype={'ICD9_CODE': str, 'LONG_TITLE': str},
        icd9_code_prefix='P'
    )

    all_icd9_codes_df = pd.concat(objs=[diagnoses_icd9_codes_df, procedures_icd9_codes_df])
    all_icd9_codes_dict = icd9_codes_df_to_dict(icd9_codes_df=all_icd9_codes_df)

    # diagnoses_df = load_df(
    #     csv_file_path='/home/cave-of-time/jaga/MIMIC/DIAGNOSES_ICD.csv',
    #     usecols=['HADM_ID', 'ICD9_CODE'],
    #     dtype={'HADM_ID': str, 'ICD9_CODE': str},
    #     icd9_code_prefix='D'
    # )
    #
    # procedures_df = load_df(
    #     csv_file_path='/home/cave-of-time/jaga/MIMIC/PROCEDURES_ICD.csv',
    #     usecols=['HADM_ID', 'ICD9_CODE'],
    #     dtype={'HADM_ID': str, 'ICD9_CODE': str},
    #     icd9_code_prefix='P'
    # )
    #
    # discharge_summaries_df = load_df(
    #     csv_file_path='/home/cave-of-time/jaga/MIMIC/NOTEEVENTS.csv',
    #     usecols=['HADM_ID', 'CHARTDATE', 'CATEGORY', 'ISERROR', 'TEXT'],
    #     dtype={'HADM_ID': str, 'CHARTDATE': str, 'CATEGORY': str, 'ISERROR': str, 'TEXT': str}
    # )
    #
    # discharge_summaries_df = discharge_summaries_df[
    #     (discharge_summaries_df.CATEGORY == 'Discharge summary') & (discharge_summaries_df.ISERROR != '1')]
    #
    # discharge_summaries_diagnoses_df = discharge_summaries_df.merge(right=diagnoses_df, on='HADM_ID')
    # discharge_summaries_procedures_df = discharge_summaries_df.merge(right=procedures_df, on='HADM_ID')
    # discharge_summaries_df = pd.concat(objs=[discharge_summaries_diagnoses_df, discharge_summaries_procedures_df])
    # discharge_summaries_df = discharge_summaries_df.merge(right=all_icd9_codes_df, on='ICD9_CODE')
    # discharge_summaries_list = discharge_summaries_df_to_list(discharge_summaries_df=discharge_summaries_df)
    #
    # train_data, development_data, test_data = split_data(
    #     data=discharge_summaries_list,
    #     seed=1891,
    #     train_percentage=0.8,
    #     development_percentage=0.1
    # )
    #
    dataset_dir = '/home/cave-of-time/jaga/MIMIC/dataset'

    if not os.path.exists(path=dataset_dir):
        os.makedirs(dataset_dir)

    with open(file=os.path.join(dataset_dir, 'mimic.json'), mode='w') as fout:
        json.dump(obj=all_icd9_codes_dict, fp=fout, indent=4)

    # train_data_dir = os.path.join(dataset_dir, 'train')
    # if not os.path.exists(path=train_data_dir):
    #     os.makedirs(train_data_dir)
    # for train_datum in train_data:
    #     with open(file=os.path.join(train_data_dir, '{}.json'.format(train_datum['summary_id'])), mode='w') as fout:
    #         json.dump(obj=train_datum, fp=fout, indent=4)
    #
    # development_data_dir = os.path.join(dataset_dir, 'dev')
    # if not os.path.exists(path=development_data_dir):
    #     os.makedirs(development_data_dir)
    # for development_datum in development_data:
    #     with open(file=os.path.join(development_data_dir, '{}.json'.format(development_datum['summary_id'])), mode='w') as fout:
    #         json.dump(obj=development_datum, fp=fout, indent=4)
    #
    # test_data_dir = os.path.join(dataset_dir, 'test')
    # if not os.path.exists(path=test_data_dir):
    #     os.makedirs(test_data_dir)
    # for test_datum in test_data:
    #     with open(file=os.path.join(test_data_dir, '{}.json'.format(test_datum['summary_id'])), mode='w') as fout:
    #         json.dump(obj=test_datum, fp=fout, indent=4)


if __name__ == '__main__':
    # main()
    refactor_embeddings('/home/jaga/XMTC/nlp.research/nlp_research/data/vectors/word2vec/en/pubmed2018_w2v_200D.bin')
    # get_embeddings_indices(
    #     '/home/jaga/XMTC/nlp.research/nlp_research/data/vectors/word2vec/en/pubmed2018_w2v_200D.bin',
    #     '/home/jaga/XMTC/nlp.research/nlp_research/data/vectors/indices/pubmed2018_w2v_200D.index'
    # )
