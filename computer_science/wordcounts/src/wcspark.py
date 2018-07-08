import os
import pyspark
from wordcounts import WordCounts


class WordCountsSpark(WordCounts):
    """
    Sorted word frequency with :py:class:`pyspark.SparkContext`

    :param pysark.SparkContext: spark context that was been passed access\
            :py:class:`wordcounts.src.wordcounts.WordCounts`
    :param str fPath: path to file
    :param str tokenizerType: (contraction | word)  from\
            :py:class:`wordcounts.src.wordcounts.WordCounts`
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
        Parse text of fPath via spark context and selected tokenizer

        :param pysark.SparkContext: spark context that was been passed access\
                :py:class:`wordcounts.src.wordcounts.WordCounts`
        :param str fPath: path to file
        :param str tokenizerType: (contraction | word)  from\
                :py:class:`wordcounts.src.wordcounts.WordCounts`

        :return: text
        :rtype: :py:class:`pyspark.RDD`
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
        Count instances of words in text.

        :param pyspark.RDD text: text

        :return: tuple(word, count)
        :rtype: list
        """
        wordCounts = text.map(lambda word: (word, 1))\
                         .reduceByKey(lambda a, b: a + b)\
                         .sortBy(lambda kv: -kv[1]).collect()

        return wordCounts

    def most_common(self, n=None):
        """
        Return (n | all) most common words and counts in document

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
    """
    Generate spark context with pyFile for counting words

    :param str fPath: path to spark word county py file
    :return: spark context with access to fPath
    :rtype: pyspark.SparkContext
    """
    sc = pyspark.SparkContext('local[4]', pyFiles=[os.path.abspath(fPath)])
    return sc
