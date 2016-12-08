import unittest
import os
import numpy as np
import pandas as pd
import math_h as mh
from scipy.signal import resample
from matplotlib import pyplot as plt


class Sensors(object):
    def __init__(self, fPath, kwargs, auto=True):
        self.df = pd.read_csv(fPath, **kwargs['df'])
        self.df = self.df[['x', 'y', 'z', 'time']]
        t = self.df.time.values
        tResampled = np.linspace(t.min(),
                                 t.max(),
                                 t.shape[0]
                                 )
        x = np.interp(tResampled, t, self.df.x)
        y = np.interp(tResampled, t, self.df.y)
        z = np.interp(tResampled, t, self.df.z)

        self.df['x'] = x
        self.df['y'] = y
        self.df['z'] = z
        self.df['time'] = tResampled
        n = self.df.shape[0]
        self.df = self.df.ix[n/5:4*n/5].reset_index()
        self.fs = 1./(tResampled[1] - tResampled[0])

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
        self.velocities = self.integrate_accel(['xf', 'yf', 'zf'])
        self.positions = self.integrate_vel(self.velocities)
        self.im = Position(self.positions)

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

    def filt_bandpass(self, low, high, order):

        xyz = self.df[['x', 'y', 'z']].values.T
        g = xyz.mean(axis=1)
        xyz = (xyz.T - g).T
        g = g/np.linalg.norm(g)
        aprime = xyz - np.outer(g,g).dot(xyz)
        eig_vector = np.linalg.svd(aprime.T)[-1][:,0]
        print eig_vector

        basis = np.array([g, eig_vector, np.cross(g, eig_vector)])
        xyz = basis.T.dot(xyz).T


        self.df['xf'] = mh.butter_bandpass_filter(xyz[:, 0], low, high, self.fs, order)
        self.df['yf'] = mh.butter_bandpass_filter(xyz[:, 1], low, high, self.fs, order)
        self.df['zf'] = mh.butter_bandpass_filter(xyz[:, 2], low, high, self.fs, order)


    def integrate_accel(self, cols=None, timeCol='time', v0=None):
        """
        """
        if not v0:
            v0 = [np.array([0., 0., 0.])]
        velocity = v0

        if not timeCol:
            self.df['dt'] = 1.
        else:
            self.df['dt'] = 0
            dts = self.df[timeCol][1:].values - self.df[timeCol][:-1]
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

    def _filt_6dof(self, w=None, n=None, timeCol=None):
        filtd = [np.array(self.df.ix[0][:n])]
        filtd[n/2:] = 0

        if not timeCol:
            self.df['dt'] = 1.
        else:
            self.df['dt'] = 0
            dts = self.df[timeCol][1:] - self.df[timeCol][:-1]
            self.df['dt'][:-1] = dts

        for ind, i in self.df.ix[1:].iterrows():
            if ind-1 < self.df.shape[0] - 1:
                linear = i[:n/2]
                angular = i[n/2:]
                w = filtd[ind-1][n/2:] + (i[n/2:] + self.df.ix[ind-1][n/2:])/2*i['dt']

                filtd.append(filtd[ind-1] + alpha * (i[:n] - filtd[ind-1]))

        filtd = np.array(filtd)
        for ind, col in enumerate(self.df.columns[:n]):
            self.df[col+'f'] = filtd[:, ind]


class Position(object):
    def __init__(self, xyz, n2=np.array([0, 0, 1]), fit=True, n=0):
        centroid, n1 = mh.fit_plane(xyz)
        xyz = xyz - centroid
        n2 = mh.enforce_2d(n2)
        xyz_n1 = mh.enforce_2d(xyz.dot(n1)).T
        xyzPlane = xyz - xyz_n1 * n2 
        self.n = 0 #-> @classmethod
        if fit:
            self.fit(n1, n2, xyzPlane)

    def fit(self, n1, n2, xyz, fname=""):
        xy = mh.rotate_v1_v2(n1, n2, xyz)[:, :2]
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.plot(xy[:, 0], xy[:, 1], c='k', lw=5)
        plt.axis('off')
        plt.savefig(fname + str(self.n), bbox_inches='tight')
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
