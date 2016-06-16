import json


class Slog(object):
    """
    Class for managing Sftp Logs
    """
    def __init__(self, log_path):
        """
        Constructor for Slog class, open and read existing log
        """

        self.log = open(log_path, 'r+')
        self.read_log()

    def read_log(self):
        """
        Read JSON log, create empty dict if not exists
        """
        try:
            self.json = json.load(self.log)
        except ValueError:
            self.json = {}

    def extend_json(self, name, lstat):
        """
        Extend/Create logged file pulls. account keys file values

        :param str name: name of account in log/config file
        :parm nd.recarray lstat: nx3 ('files' 'time' 'size')
        """

        if self.json[name]:
            self.json[name]['files'].extend(lstat['files'].tolist())
            self.json[name]['size'].extend(lstat['size'].tolist())
            self.json[name]['time'].extend(lstat['time'].tolist())
        else:
            self.json[name] = {}
            print(lstat)
            self.json[name]['files'] = lstat['files'].tolist()
            self.json[name]['size'] = lstat['size'].tolist()
            self.json[name]['time'] = lstat['time'].tolist()
