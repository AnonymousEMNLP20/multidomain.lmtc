import json
import os
from nltk import sent_tokenize
from lmtc.document_model.model import Tag, Document
from lmtc.experiments.configurations.configuration import Configuration


class JSONLoader():

    def read_file(self, filename: str):
        tags = []
        with open(filename) as file:
            data = json.load(file)
        sections = []
        text = ''

        if Configuration['task']['dataset'] == 'eurovoc_en':
            sections = [data['header'], data['recitals']]
            sections.extend(data['main_body'])
            sections.append(data['attachments'])
            text = '\n'.join(sections)
            sections = []
        elif Configuration['task']['dataset'] == 'amazon13k':
            if Configuration['sampling']['preprocessed']:
                if Configuration['model']['hierarchical']:
                    sections.append(data['title_tokenized'])
                    sections.extend(data['sentences_tokenized'])
                else:
                    text = '{} \n {}'.format(data['title_tokenized'], data['description_tokenized'])
            else:
                if Configuration['model']['hierarchical']:
                    sections.append(data['title'])
                    sections.extend(sent_tokenize(data['description']))
                else:
                    text = '{} \n {}'.format(data['title'], data['description'])
                    sections = []
        elif Configuration['task']['dataset'] == 'mimic':
            if Configuration['sampling']['preprocessed']:
                if Configuration['model']['hierarchical']:
                    sections.extend(data['tokenized_sections'])
                else:
                    text = data['tokenized_text']
            else:
                if Configuration['model']['hierarchical']:
                    sections.extend(data['text'].split('\n\n'))
                else:
                    text = data['text']

        for concept in data['concepts']:
            tags.append(concept)
        return Document(text, tags, sentences=sections, filename=os.path.basename(filename))
