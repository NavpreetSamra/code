from abc import ABCMeta, abstractmethod
import warnings
import nltk


class WordCounts(object):
    """
    Abstract Meta Class with shared helper implementations and properties
    """
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
        :return most_common: key value pairs in sorted order
        :rtype: list.tuple.(str, int)
        """
        warnings.warn('attempted abstract counter')

    @abstractmethod
    def _auto(self):
        warnings.warn('attempted abstract _auto')

    @staticmethod
    def words_tokenizer(txt):
        """
        Tokenize with nltk removing punctuations

        :param str txt: text to toknize
        :return: tokens
        :rtype: list
        """
        return [word.lower() for word in
                nltk.wordpunct_tokenize(txt) if word.isalpha()]

    @staticmethod
    def contraction_tokenizer(txt):
        """
        Remove apostraphes (it's -> its) and tokenize with nltk
        removing punctuations

        :param str txt: text to toknize
        :return: tokens
        :rtype: list
        """
        txt = txt.replace("'", "")
        return WordCounts.words_tokenizer(txt)
