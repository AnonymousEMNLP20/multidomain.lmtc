import re
import logging
from lmtc.experiments.configurations.configuration import Configuration
LOGGER = logging.getLogger(__name__)


class Document:
    """
    A document is a combination of text and the positions of the tags in that text.
    """

    def __init__(self, text, tags, sentences=None, filename=None):
        """
        :param text: document text as a string
        :param tags: list of Tag objects
        """

        if text:
            if Configuration['sampling']['preprocessed']:
                self.tokens = [token for token in text.split(' ')]
            else:
                self.tokens = [token for token in Configuration['sampling']['tagger'].tokenize_text(text)]

        if sentences:
            self.sentences = []
            for sentence in sentences:
                if Configuration['sampling']['preprocessed']:
                    self.sentences.append([token for token in sentence.split()])
                else:
                    self.sentences.append([token for token in Configuration['sampling']['tagger'].tokenize_text(sentence)])
        self.tags = tags
        self.text = text
        self.filename = filename

    def __repr__(self):
        return 'Document(tokens={}, tags={}, text={})'.format(self.tokens, self.tags, self.text)
