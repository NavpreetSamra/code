from abc import ABCMeta, abstractmethod, abstractproperty
import warnings
import nltk
import requests
from collections import Counter


class WordCounts():
    __meta__ = ABCMeta

    @abstractproperty
    def text(self):
        warnings.warn('access abstract property')

class WordCountsSerial(object):
    def __init__(self, fPath=None, parseType='html', countType='serial',
                 countKwargs=None, wordList=None, auto=True):
        self.fPath = fPath
        self.countType = countType

        if countKwargs:
            self.countKwargs = countKwargs
        else:
            self.countKwargs = {}

        if wordList:
            self.text = wordList

        self.parseType = parseType

        if auto:
            self._auto()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    @property
    def wordCounts(self):
        return self._wordCounts

    @wordCounts.setter
    def wordCounts(self, wordCounts):
        self._wordCounts = wordCounts

    def html_parser(self, fPath):
        self._raw = requests.get(fPath).text
        self._text = [word.lower() for word in
                      nltk.wordpunct_tokenize(self._raw) if word.isalpha()]

    def serial_counter(self, **kwargs):
        self.wordCounts = Counter(self.text)

    def _auto(self):
        if self.fPath:
            self.__getattribute__(self.parseType + '_parser')(self.fPath)
        self.__getattribute__(self.countType + '_counter')(**self.countKwargs)
