from abc import ABCMeta, abstractmethod
import warnings
import nltk
import requests
from collections import Counter


class WordCounts(object):
    __meta__ = ABCMeta

    @property
    def text(self):
        """
        Tokenized list of words
        """
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def wordCounts(self):
        """
        :py:class:`collections.Counter` of words
        """
        return self._wordCounts

    @wordCounts.setter
    def wordCounts(self, wordCounts):
        self._wordCounts = wordCounts

    @abstractmethod
    def parser(self):
        """
        Abstraction for parsing implementations
        """
        warnings.warn('attempted abstract parser')

    @abstractmethod
    def counter(self):
        """
        Abstraction to count word fequencies store
        in :py:attr:`WordCounts.wordCounts`
        """
        warnings.warn('attempted abstract counter')

    @abstractmethod
    def most_common(self, n=None):
        """
        Abstraction  for returning the n most frequent
        words and associated values

        :param int n: number to return, default None returns all
        :return list tuple: key value pairs in sorted order
        """
        warnings.warn('attempted abstract counter')

    @abstractmethod
    def _auto(self):
        warnings.warn('attempted abstract parse')

    @staticmethod
    def word_only_tokenize(txt):
        """
        Tokenize with nltk removing punctuations

        :param str txt: text to toknize
        :return: tokens
        :rtype: list
        """
        return [word.lower() for word in
                nltk.wordpunct_tokenize(txt) if word.isalpha()]

    @staticmethod
    def contraction_tokenize(txt):
        """
        Remove epostraphes (it's -> its) and tokenize with nltk
        removing punctuations

        :param str txt: text to toknize
        :return: tokens
        :rtype: list
        """
        txt.replace("'", "")
        return WordCounts.word_only_tokenize(txt)


class WordCountsSerial(WordCounts):
    """
    Word frequency with :py:class:`collections.Counter` with file (text)
    and html reading interfaces

    :param str fPath: path to file
    :param str parserType: type of file (html | file)
    :param list wordList: option to specify words to count without parsing
                          from file
    :param bool auto: auto run
    """
    def __init__(self, fPath=None, parserType='html',
                 wordList=None, auto=True):
        self.fPath = fPath
        self.parserType = parserType

        self.text = []
        if wordList:
            self.text = wordList

        if auto:
            self._auto()

    def parser(self, fPath, parserType):
        """
        Front end for parsing

        :param str fPath: path to file
        :param str parserType: type of file (html | file)
        """
        self.__getattribute__('_' + parserType + '_parser')(fPath)

    def _html_parser(self, fPath):
        raw = requests.get(fPath).text
        self.text = self.word_only_tokenize(raw)

    def _file_parser(self, fPath):
        with open(fPath) as f:
            for line in f:
                self.text.extend(self.word_only_tokenize(line))

    def counter(self):
        self.wordCounts = Counter(self.text)

    def most_common(self, n=None):
        return self.wordCounts.most_common(n)

    def _auto(self):
        if self.fPath:
            self.parser(self.fPath, self.parserType)
        self.counter()
