import unittest
import os
import numpy as np
import math
import pandas as pd
import math_h as mh
from matplotlib import pyplot as plt
import skinematics as sk


class Sensors(object):
    def __init__(self, fPath, kwargs, acols=['accelX', 'accelY', 'accelZ'], 
                 tcol=['Timestamp'], 
                 gcols=['gyroX(rad/s)', 'gyroY(rad/s)', 'gyroZ(rad/s)'], 
                 mcols=[19, 20, 21], auto=True):

        self.df = pd.read_csv(fPath, parse_dates=True, **kwargs['df']).drop_duplicates('Timestamp')
        n = self.df.shape[0]
        # self.df = self.df.ix[n/20:19*n/20]
        self.build_dfs(acols, tcol, gcols, mcols)

        if not auto:
            return

        # g0 = np.array([0, 0, -1])
        # g = self.dfs['a'].mean()
        # r0 = mh.matrix_from_v1_v2(g0, g)
        # # r0 = np.array([[0,0,1],[0,0,-1],[0,1,0]])
        # omega = self.dfs['g'].values
        # x0 = np.array([0,0,0])
        # _, self.pos = sk.imus.calc_QPos(r0, omega, x0, self.dfs['a'].values, self.fs)
        filt = kwargs['sensors']['filt']
        if filt['name']:
            self.__getattribute__(filt['name'])(**filt['kwargs'])

        fuse = kwargs['sensors']['fuse']
        if fuse:
            self.__getattribute__(fuse)

        integrates = kwargs['sensors']['integrate']
        if integrates:
            self.__getattribute__(kwargs['integrate'])
        self.velocities = self.integrate_accel()
        self.positions = self.integrate_vel()
        self.im = Position(self.positions)

    def build_dfs(self, acols, tcol, gcols, mcols):

        self.dfs = {}
        self.dfs['a'] = pd.DataFrame()
        self.dfs['g'] = pd.DataFrame()
        self.dfs['t'] = pd.DataFrame()
        self.dfs['m'] = pd.DataFrame()
        self.dfs['af'] = pd.DataFrame()
        self.dfs['vel'] = pd.DataFrame()
        self.dfs['pos'] = pd.DataFrame()
        t = self.df[tcol].Timestamp.apply(lambda x: float(x.rsplit(':', 1)[-1])).values
        t = t - t.min()
         
        tResampled = np.linspace(0, t.max() - t.min(), t.shape[0])
        if acols:
            self.dfs['a'] = pd.DataFrame()
            for i in acols:
                self.dfs['a'][i] = np.interp(tResampled, t, self.df[i].values)
        if gcols:
            self.dfs['g'] = self.df[gcols]
        if mcols:
            self.dfs['m'] = self.df[mcols]

        self.dfs['t']['time'] = tResampled
        self.dt = (tResampled[1] - tResampled[0])
        self.fs = 1./self.dt

    def _filt_lowpass(self, alpha=None, n=None):
        """
        """
        filtd = [np.array(self.df.ix[0][:n])]

        for ind, i in self.df.ix[1:].iterrows():
            if ind-1 < self.df.shape[0] - 1:
                filtd.append(filtd[ind-1] + alpha * (i[:n] - filtd[ind-1]))

        filtd = np.array(filtd)
        for ind, col in enumerate(self.df.columns[:n]):
            self.df[col+'f'] = filtd[:, ind]

    def filt_bandpass(self, low, high, order, sixdof=False):

        xyz = self.dfs['a'].values
        # xyz = mh.project(self.dfs['a'].values)
        # xyz = mh.center_rotate(self.dfs['a'].values)
        # xyz = self.dfs['a'].values.T
        # g = xyz.mean(axis=1)
        # xyz = (xyz.T - g).T
        # g = g/np.linalg.norm(g)
        # aprime = xyz - np.outer(g,g).dot(xyz)
        # eig_vector = np.linalg.svd(aprime.T)[-1][:,0]
        # print eig_vector

        # basis = np.array([g, eig_vector, np.cross(g, eig_vector)])
        # xyz = basis.T.dot(xyz).T

        df = self.dfs['af']
        df['xf'] = mh.butter_bandpass_filter(xyz[:, 0], low, high, self.fs, order)
        df['yf'] = mh.butter_bandpass_filter(xyz[:, 1], low, high, self.fs, order)
        df['zf'] = mh.butter_bandpass_filter(xyz[:, 2], low, high, self.fs, order)


        # xyz = self.dfs['a'].values.T
        # g = xyz.mean(axis=1)
        # xyz = (xyz.T - g).T
        # g = g/np.linalg.norm(g)
        # aprime = xyz - np.outer(g,g).dot(xyz)
        # eig_vector = np.linalg.svd(aprime.T)[-1][:,0]
        # print eig_vector

        # basis = np.array([g, eig_vector, np.cross(g, eig_vector)])
        # xyz = basis.T.dot(xyz).T

        if sixdof:
            self._filt_6dof()


    def _filt_6dof(self):
        a = self.dfs['af'] # Bandpass filtered accelerometer
        w = self.dfs['g'] # Angular velocity

        r_acc = [a.ix[0]]
        r_w = [w.ix[0]]
        r_est = [a.ix[0]]

        Axz = [math.atan2(r_est[0][0], r_rest[0][2])]
        Ayz = [math.atan2(r_est[1][0], r_rest[0][2])]

        Wxz = [w.ix[0][0]]
        Wyz = [w.ix[0][1]]

        x = sin(Axz[0]) / np.sqrt(1 + cos(Axz[0])**2 * tan(Ayz[0]**2))
        y = sin(Ayz[0]) / np.sqrt(1 + cos(Ayz[0])**2 * tan(Axz[0]**2))
        zarg = 1 - x**2 - y**2
        z = np.sign(zarg) * zarg

        for aid, wid in zip(a.ix[1:].iterrows(), w.ix[1:].iterrows()):
            ai, ind = aid
            wi, _ = wid
            if ind-1 < self.df.shape[0] - 1:
                wxavg = np.mean([wi[0], Wxz[ind-1]])
                
                Axz = math.atan2(r_est[ind][0], r_rest[ind][2])
                Ayz = math.atan2(r_est[ind][0], r_rest[ind][2])
                x = sin(Axz[0]) / np.sqrt(1 + cos(Axz[0])**2 * tan(Ayz[0]**2))
                y = sin(Ayz[0]) / np.sqrt(1 + cos(Ayz[0])**2 * tan(Axz[0]**2))
                zarg = 1 - x**2 - y**2
                z = np.sign(zarg) * zarg
                # Azx = 
                linear = i[:n/2]
                angular = i[n/2:]
                w = filtd[ind-1][n/2:] + (i[n/2:] + self.df.ix[ind-1][n/2:])/2*i['dt']

                filtd.append(filtd[ind-1] + alpha * (i[:n] - filtd[ind-1]))

        filtd = np.array(filtd)
        for ind, col in enumerate(self.df.columns[:n]):
            self.df[col+'f'] = filtd[:, ind]
    def integrate_accel(self, v0=None):
        """
        """
        df = self.dfs['af']
        g = df.mean().values
        r = mh.matrix_from_v1_v2(g, [0, 0, 1])
        xyz = df.values
        xyz = mh.project(self.dfs['af'].values)
        xyz = mh.center_rotate(self.dfs['a'].values)
        if not v0:
            v0 = [np.array([0., 0., 0.])]
        velocity = v0

        for ind, i in self.dfs['af'].iterrows():
            if ind < self.df.shape[0] - 1:
                velocity.append(velocity[ind] + self.dt * i.values)

        velocity = np.array(velocity)
        for ind, col in enumerate(self.dfs['af'].columns):
            self.dfs['vel'][col.split('f')[0]+'vel'] = velocity[:, ind]
        return velocity

    def integrate_vel(self, x0=None):
        """
        """
        if not x0:
            x0 = [np.array([0., 0., 0.])]
        position = x0

        for ind, i in self.dfs['vel'].iterrows():
            if ind < self.df.shape[0] - 1:
                position.append(position[ind] + self.dt * i.values)

        position = np.array(position)
        for ind, col in enumerate(self.dfs['vel'].columns):
            self.dfs['pos'][col.split('vel')[0]+'pos'] = position[:, ind]
        return np.array(position)



class Position(object):
    def __init__(self, xyz, n2=np.array([0, 0, 1]), fit=True, n=0):
        xyz = xyz[xyz.shape[0]/20:xyz.shape[0]*18/20, :]
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
