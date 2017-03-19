import unittest
from src import wordcounts, wcserial, wcspark


class TestAbstraction(unittest.TestCase):
    def test_tokenizers(self):
        l1 = "Its its"
        l2 = "Its its it's"
        l3 = "Its its its\nIts its it's"

        words = wordcounts.WordCounts.words_tokenizer
        contractions = wordcounts.WordCounts.contraction_tokenizer

        self.assertEqual(words(l1), ['its', 'its'])
        self.assertEqual(words(l2), ['its', 'its', 'it', 's'])
        self.assertEqual(words(l3), ['its', 'its', 'its', 'its',
                                     'its', 'it', 's'])

        self.assertEqual(contractions(l1), ['its', 'its'])
        self.assertEqual(contractions(l2), ['its', 'its', 'its'])
        self.assertEqual(contractions(l3), ['its', 'its', 'its',
                                            'its', 'its', 'its'])


class TestWordCountsSerial(unittest.TestCase):
    def test_line(self):
        wcs = wcserial.WordCountsSerial(fPath='test/line.txt',
                                        parserType='file')
        self.assertEqual(wcs.most_common(),  [('its', 3)])

    def test_file(self):
        wcs = wcserial.WordCountsSerial(fPath='test/test.txt',
                                        parserType='file')
        self.assertEqual(wcs.most_common(1),  [('its', 3)])
        self.assertEqual(wcs.most_common()[1],  ('world', 2))


class TestWordCountsSpark(unittest.TestCase):

    sc = wcspark.spark_setup('./src/wcspark.py')

    def test_line(self):
        wcs = wcspark.WordCountsSpark(self.sc, fPath='test/line.txt')
        self.assertEqual(wcs.most_common(),  [('its', 3)])

    def test_file(self):
        wcs = wcspark.WordCountsSpark(self.sc, fPath='test/test.txt')
        self.assertEqual(wcs.most_common(1),  [('its', 3)])
        self.assertEqual(wcs.most_common()[1],  ('world', 2))

if __name__ == "__main__":
    unittest.main()
