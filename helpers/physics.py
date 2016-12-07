import unittest
import os
import numpy as np
import pandas as pd
import math_h as mh
from matplotlib import pyplot as plt


class Sensors(object):
    def __init__(self, fPath, kwargs, auto=True):
        self.df = pd.read_csv(fPath, **kwargs['df'])

        if not auto:
            return

        fuse = kwargs['sensors']['fuse']
        if fuse:
            self.__getattribute__(fuse)

        filt = kwargs['sensors']['filt']
        if filt['name']:
            self.__getattribute__(filt['name'])(**filt['kwargs'])

        integrates = kwargs['sensors']['integrate']
        if integrates:
            self.__getattribute__(kwargs['integrate'])
        velocities = self.integrate_accel(['xf', 'yf', 'zf'])
        positions = self.integrate_vel(velocities)
        self.im = Position(positions)

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
        return velocity

    def integrate_vel(self, velocities, timeCol=None, x0=None):
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

        vel_dt = np.c_[velocities, self.df['dt'].values]
        for ind, i in enumerate(vel_dt[1:, :]):
            if ind-1 < self.df.shape[0] - 1:
                position.append(position[ind-1] + i[:3] * i[3])

        return np.array(position)

class Position(object):
    def __init__(self, xyz, n2=np.array([0, 0, 1]), fit=True):
        centroid, n1 = mh.fit_plane(xyz)
        n2 = mh.enforce_2d(n2)
        xyz_n1 = mh.enforce_2d(xyz.dot(n1)).T
        xyzPlane = xyz - xyz_n1 * n2 - centroid
        if fit:
            self.fit(n1, n2, xyzPlane)

    def fit(self, n1, n2, xyz, fname=""):
        xy = mh.rotate_v1_v2(n1, n2, xyz)[:, :2]
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.plot(xy[:, 0], xy[:, 1], c='k', lw=5)
        plt.axis('off')
        plt.savefig('temp'+fname, bbox_inches='tight')
        plt.close()



class PhoneDraw(object):
    def __init__(self, files, clf, labels=None,
                 keywords={'df': {},
                           'sensors': {'fuse': None, 'filt': None},
                           'ml': {}}):
        """
        Front end for classifying numbers from IMU data.

        :param array-like files: list of files to classify
        :param classifier clf: classifier with predict method\
                for mxn raveled data
        :param Position sensors: Class to parse IMU data
        """
        if not labels:
            labels = [None] * len(files)
        index = [os.path.basename(f) for f in files]
        columns = ['label', 'classificaton']
        self.files = pd.DataFrame(index=index, columns=columns)
        for f, l in zip(files, labels):
            data = Sensors(f, keywords)
            self.data = data
            im = Position(data.df.values[-3:, :])
            if clf:
                self.files[f]['classification'] = clf.predict(im.arr)


class TestPhoneDraw(unittest.TestCase):

    def test_x(self):
        pass

if __name__ == "__main__":
    unittest.main()
