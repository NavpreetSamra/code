import os
import pyspark
from wordcounts import WordCounts


class WordCountsSpark(WordCounts):
    """
    Sorted word frequency with :py:class:`pyspark.SparkContext`

    :param str fPath: path to file
    :param list wordList: option to specify words to count without parsing
                          from file
    :param bool auto: auto run
    """
    def __init__(self, sc, fPath=None, tokenizerType='contraction',
                 auto=True):
        self.fPath = fPath
        self.sc = sc
        self.tokenizerType = tokenizerType

        if auto:
            self._auto()

    @staticmethod
    def parser(sc, fPath, tokenizerType):
        """
        Abstraction for parsing implementations
        """
        text = sc.textFile(fPath).flatMap(lambda line:
                                          WordCounts.__dict__
                                          [tokenizerType +
                                           '_tokenizer'].__func__
                                          (line))

        return text

    @staticmethod
    def counter(text):
        """
        Abstraction to count word fequencies store
        in :py:attr:`WordCounts.wordCounts`
        """
        wordCounts = text.map(lambda word: (word, 1))\
                         .reduceByKey(lambda a, b: a + b)\
                         .sortBy(lambda kv: -kv[1]).collect()
        return wordCounts

    def most_common(self, n=None):
        """
        Abstraction  for returning the n most frequent
        words and associated values

        :param int n: number to return, default None returns all
        :return most_common: key value pairs in sorted order
        :rtype: list.tuple.(str, int)
        """
        if n is None:
            n = len(self.wordCounts)
        return self.wordCounts[:n]

    def _auto(self):
        self.text = self.parser(self.sc, self.fPath, self.tokenizerType)
        self.wordCounts = self.counter(self.text)


def spark_setup(fPath='./wcspark.py'):
    sc = pyspark.SparkContext('local[4]', pyFiles=[os.path.abspath(fPath)])
    return sc

