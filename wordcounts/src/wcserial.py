import requests
from collections import Counter
from wordcounts import WordCounts


class WordCountsSerial(WordCounts):
    """
    Word frequency with :py:class:`collections.Counter` with file (text)
    and html reading interfaces

    :param str fPath: path to file
    :param str parserType: type of file (html | file)
    :param str tokenizerType: (contraction | word)  from\
            :py:class:`wordcounts.WordCounts`
    :param list wordList: option to specify words to count without parsing
                          from file
    :param bool auto: auto run
    """
    def __init__(self, fPath=None, parserType='html',
                 tokenizerType='contraction', wordList=None, auto=True):
        self.fPath = fPath
        self.parserType = parserType
        self.tokenizerType = tokenizerType

        self.text = []
        if wordList:
            self.text = wordList

        if auto:
            self._auto()

    def parser(self, fPath, tokenizerType, parserType):
        """
        Front end for parsing

        :param str fPath: path to file
        :param str parserType: type of file (html | file)
        """
        self.__getattribute__('_' + parserType + '_parser')(fPath,
                                                            tokenizerType)

    def _html_parser(self, fPath, tokenizerType):
        raw = requests.get(fPath).text
        self.text = self.__getattribute__(tokenizerType + '_tokenizer')(raw)

    def _file_parser(self, fPath, tokenizerType):
        with open(fPath) as f:
            for line in f:
                self.text.extend(self.__getattribute__(tokenizerType +
                                                       '_tokenizer')(line))

    def counter(self):
        """
        Count instances of words in :py:attr:`WordCounts.text`
        """
        self.wordCounts = Counter(self.text)

    def most_common(self, n=None):
        """
        Return (all | the n most) frequent words and associated values

        :param int n: number to return, default None returns all
        :return most_common: key value pairs in sorted order
        :rtype: list.tuple.(str, int)
        """
        return self.wordCounts.most_common(n)

    def _auto(self):
        if self.fPath:
            self.parser(self.fPath, self.tokenizerType, self.parserType)
        self.counter()
