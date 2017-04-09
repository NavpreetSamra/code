import argparse 
import os
from distutils.dir_util import mkpath
import pandas as pd
from sklearn.externals import joblib


class HousingPrediction(object):
    """
    """

    def __init__(self, inputFile, srcDir, outputFolder):
        """
        """
        df = pd.read_csv(inputFile)
        cleaner = joblib.load(srcDir + '/cleaner.pkl')
        model = joblib.load(srcDir + '/model.pkl')
        predictions = model.predict(cleaner.transform(df))
        pd.concat([df, predictions], axis=1).to_csv(outputFolder + 'predictions_' + inputFile, index=False)


def main():
    srcDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("f", "input_file", type=str, help='name of input file')
    parser.add_argument("-d", "--output_folder", type=str, default='./', help='path to output directory defaults to ./')

    args = parser.parse_args()
    mkpath(args.output_folder)

    HousingPrediction(args.input_file, srcDir, args.output_folder)


if __name__ == '__main__':
    main()





