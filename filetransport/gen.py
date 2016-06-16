import paramiko as pm
import numpy as np
import re
import os


class Gen(pm.Transport):
    """
    Class for creating and operating on SFTP objects from Paramiko.Transport
    """

    def __init__(self, host, username, password, port=None, account=None, remote_dir='./',
                 local_dir=None, action=None, pattern=None, log=None, verbose=False):
        """
        :param str host: SFTPClient host
        :param str username: SFTPClient username
        :param str password: SFTPClient password
        :param str port: SFTPClient port
        :param str remote_dir: remote directory on SFTP
        :param str local_dir: directory on local machine
        :param str action: push or pull (optional)
        :param Slog log: Slog object
        :param bool verbose: True for terminal  output on file information
        """

        if port:
            socket = tuple(host, port)
        else:
            socket = host

        pm.Transport.__init__(self, socket)
        self.connect(username=username, password=password)
        self.sftp = pm.SFTPClient.from_transport(self)
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.action = action
        self.pattern = pattern
        self.account = account
        self.log = log
        self.verbose = verbose
        if self.verbose:
            print(remote_dir)
            print(self.sftp.listdir(remote_dir))

    def later_files(self, ascending=False, verbose=False):
        """
        List files with stats in remote directory by time \
            default oldest to youngest by modified time

        :parm bool ascending: default to oldest-> youngest.\
            Set to True for reverse
        :param bool verbose: True for terminal  output on file information
        """
        files = self.sftp.listdir(self.remote_dir)

        hold = []
        for file in files:
            if self.pattern:
                if bool(re.search(self.pattern, file)):
                    lstat = self.sftp.lstat(self.remote_dir + '/' + file)
                    hold.append(tuple([file, lstat.st_mtime, lstat.st_size]))
            else:
                lstat = self.sftp.lstat(self.remote_dir + '/' + file)
                hold.append(tuple([file, lstat.st_mtime, lstat.st_size]))
        if hold:
            stats = np.array(hold, dtype=[('files', 'O'),
                             ('time', int), ('size', int)])
            self.sftp_stats = stats[stats['time'].argsort()]

            if verbose:
                print self.sftp_stats

    def pull(self, forced=False):
        """
        Pull file from remote location to local
        """

        files = self.sftp_stats['files']
        if forced:
            l = np.array([True] * len(files))
        else:
            l = np.in1d(files, self.log['files'], invert=True)
        if np.any(l):
            self.sftp_stats = self.sftp_stats[l]

            files = self.sftp_stats['files']
            for f in files:
                self.local_path = self.local_dir + '/' + f
                self.remote_path = self.remote_dir + '/' + f
                if self.verbose:
                    print('pulling ' + self.remote_path)
                    print('to ' + self.local_path)
                self.sftp.get(self.remote_path, self.local_path)
                self.check_pull(f)

        else:
            self.sftp_stats = None

    def check_pull(self, f):
        """
        Check if pulled local file has same size as remote file
        """
        size = self.sftp_stats['size'][np.in1d(self.sftp_stats['files'], f)]
        self.new_stat = os.lstat(self.local_path)
        if not int(self.new_stat.st_size) == int(size):
            try:
                self.warning.extend(f)
            except AttributeError:
                self.warning = list([f])
        elif self.verbose:
            print('pulled ' + f + ' successfully')
