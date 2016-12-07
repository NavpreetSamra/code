import unittest
import os
import numpy as np
import pandas as pd


class Sensors(object):
    def __init__(self, fPath, kwargs):
        self.df = pd.read_csv(fPath, **kwargs['df'])

        fuse = kwargs['sensors']['fuse']
        if fuse:
            self.__getattribute__(fuse)

        filt = kwargs['sensors']['filt']
        if filt['name']:
            self.__getattribute__(filt['name'])(**filt['kwargs'])

        integrates = kwargs['sensors']['integrate']
        if integrates:
            self.__getattribute__(kwargs['integrate'])
        self.integrate_accel(['xf', 'yf', 'zf'])
        self.integrate_vel(['xfvel', 'yfvel', 'zfvel'])

    def filt_lowpass(self, alpha=None, n=None):
        """
        """
        filtd = [np.array(self.df.ix[0][:n])]

        for ind, i in self.df.ix[1:].iterrows():
            if ind-1 < self.df.shape[0] - 1:
                filtd.append(filtd[ind-1] + alpha * (i[:n] - filtd[ind-1]))

        filtd = np.array(filtd)
        for ind, col in enumerate(self.df.columns[:n]):
            self.df[col+'f'] = filtd[:, ind]

    def integrate_accel(self, cols=None, timeCol=None, v0=None):
        """
        """
        if not v0:
            v0 = [np.array([0., 0., 0.])]
        velocity = v0

        if not timeCol:
            self.df['dt'] = 1.
        else:
            self.df['dt'] = 0
            dts = self.df[timeCol][1:] - self.df[timeCol][:-1]
            self.df['dt'][:-1] = dts

        for ind, i in self.df.ix[1:].iterrows():
            if ind-1 < self.df.shape[0] - 1:
                velocity.append(velocity[ind-1] + i['dt'] * i[cols].values)

        velocity = np.array(velocity)
        for ind, col in enumerate(cols):
            self.df[col+'vel'] = velocity[:, ind]

    def integrate_vel(self, cols=None, timeCol=None, x0=None):
        """
        """
        if not x0:
            x0 = [np.array([0., 0., 0.])]
        position = x0

        if 'dt' not in self.df:
            if not timeCol:
                self.df['dt'] = 1.
            else:
                self.df['dt'] = 0
                dts = self.df[timeCol][1:] - self.df[timeCol][:-1]
                self.df['dt'][:-1] = dts

        for ind, i in self.df.ix[1:].iterrows():
            if ind-1 < self.df.shape[0] - 1:
                position.append(position[ind-1] + i['dt'] * i[cols].values)

        position = np.array(position)
        for ind, col in enumerate(cols):
            self.df[col+'pos'] = position[:, ind]

class Position():
    pass


class PhoneDraw():
    def __init__(self, files, clf, labels=None,
                 keywords={'df': {}, 
                     'sensors': {'fuse': None, 'filt': None}, 'ml': {}}):
        """
        Front end for classifying numbers from IMU data.

        :param array-like files: list of files to classify
        :param classifier clf: classifier with predict method for mxn raveled data
        :param Position sensors: Class to parse IMU data
        """
        if not labels:
            labels = [None] * len(files)
        index = [os.path.basename(f) for f in files]
        columns = ['label', 'classificaton']
        self.files = pd.DataFrame(index=index, columns=columns)
        for f, l in zip(files, labels):
            data = Sensors(f, keywords)
            im = Position(data.df.values[-3:,:])
            if clf:
                self.files[f]['classification'] = clf.predict(im.arr)



class TestPhoneDraw(unittest.TestCase):

    def test_x(self):
        pass

if __name__ == "__main__":
    unittest.main()
