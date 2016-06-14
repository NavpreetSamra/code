import json
import slog
import gen
import re


class Owner(object):
    """
    Owner for creating sftp objects and performing actions
    """
    def __init__(self, json_path='/home/mweiss/Scripts/msw/sftp.config',
                 log_path='/home/rapleaf/SftpAutomator/logs.json',
                 name=None):
        """
        Constructor for Sftp Owner, requires config and log JSONs
        :param str json_file: json input file path containing metedata for pull
        :param str log_path: path to log file JSON
        """

        self.json_handle = open(json_path, 'r')
        self.transport = {}
        self.warning = {}
        self.log_path = log_path
        self.log = slog.Slog(self.log_path)
        self.gen_sftp_objects(name)

    def gen_sftp_objects(self, name):
        """
        Create Sftp objects based on config files. store in Owner.
        """
        self.json = json.load(self.json_handle)
        for account in self.json:
            if account['name'] not in  self.log.json:
                self.log.json[account['name']] = None
            self.transport[account['name']] = gen.Gen(

                    account['host'], account['username'],
                    account['password'], account['port'],
                    account['name'],
                    account['remote_directory'],
                    account['local_directory'], account['action'],
                    account['regex'], self.log.json[account['name']],
                    verbose=True
                                                            )
            self.transport[account['name']].later_files()
           

    def run_sftp_action(self):
        """
        Execute actions specified in object by config
        """
        log_accounts = [i for i in self.log.json]

        for account in self.json:
            if self.log.json[account['name']]:
                sftp_obj = self.transport[account['name']]

                if sftp_obj.action:
                    if bool(re.search(sftp_obj.action, 'pull')):
                        print(account['name'])
                        if 'sftp_stats' in sftp_obj.__dict__:
                            sftp_obj.pull()

                    elif bool(re.search(sftp_obj.action, 'push')):
                        if 'sftp_stats' in sftp_obj.__dict__:
                            sftp_obj.push()

                    else:
                        raise valueerror('action not push or pull, invalid action \
                                      for ' + account)
            if 'sftp_stats' in self.transport[account['name']].__dict__ and self.transport[account['name']].sftp_stats is not None:
                self.log.extend_json(account['name'], self.transport[account['name']].sftp_stats)
        self.log_action()

    def log_action(self):
        """
        Log files pulled by run_sftp_action
        """
        with open(self.log_path, 'w') as write_out:
            json.dump(self.log.json, write_out)

        for i in self.transport:
            try:
                self.warning[i] = self.transport[i].warning
            except AttributeError:
                pass
        if self.warning:
            raise FileValidationError(self.warning)


class FileValidationError(Exception):
    """
    This Exception should be thrown when a file doesn't transfer properly.
    This is the case when the size on the SFTP doesn't match the size of the local file
    """
    def __init__(self, warnings):
        """
        Constructor for Exception subclass. Logs invalid file transfers

        :param dict warnings: keys = account names values = nd.array of string file names
        """
        Exception.__init__(self, "")

        self.warnings = warnings

    def __str__(self):
        error_message = "The following files have not properly transferred:\n"
        for account_name in self.warnings.keys():
            error_message += "Account name: %s\nFiles:" % (account_name)
            for file in self.warnings[account_name]:
                error_message += "\n\t%s" % (file)
        return error_message



if __name__ == 'main':
    sftp_owner = Owner()
    sftp_owner.run_sftp_action()
