import json
import slog
import gen
import regex


class Owner(object):
    """
    Owner for creating SFTP Objects and performing actions

    Constructor for SFTP Owner, requires config and log JSONs

    :param str json_file: json input file path containing metedata for pull
    :param str log_path: path to log file JSON
    """
    def __init__(self, json_path='/home/mweiss/code/sftp.config',
                 log_path='/home/mweiss/code/logs.json',
                 name=None):

        self.json_handle = open(json_path, 'r')
        self.transport = {}
        self.warning = {}
        self.log_path = log_path
        self.log = slog.Slog(self.log_path)
        self.gen_sftp_objects(name)

    def gen_sftp_objects(self, name):
        """
        Create SFTP Objects based on config files. store in Owner.
        """
        self.json = json.load(self.json_handle)
        for account in self.json:
            if account['name'] not in self.log.json:
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

        for account in self.json:
            if self.log.json[account['name']]:
                sftp_obj = self.transport[account['name']]

                if sftp_obj.action:
                    if bool(regex.search(sftp_obj.action, 'pull')):
                        print(account['name'])
                        if 'sftp_stats' in sftp_obj.__dict__:
                            sftp_obj.pull()

                    elif bool(regex.search(sftp_obj.action, 'push')):
                        if 'sftp_stats' in sftp_obj.__dict__:
                            sftp_obj.push()

            if 'sftp_stats' in self.transport[account['name']].__dict__ and self.transport[account['name']].sftp_stats is not None:
                self.log.extend_json(account['name'], self.transport[account['name']].sftp_stats)
        self.log_action()

    def log_action(self):
        """
        Log files and metadata
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
    FileValidationError throws when a file does not transfer correctly \
            (confirmed by checking size of file.

    Constructor for Exception subclass. Tracks invalid file transfers

    :param dict warnings: keys - acount names \
                          valuees - nd.array of string file names
    """
    def __init__(self, warnings):
        Exception.__init__(self, "")

        self.warnings = warnings

    def __str__(self):
        error_message = "The following files have not properly transferred:\n"
        for account_name in self.warnings.keys():
            error_message += "Account name: %s\nFiles:" % (account_name)
            for file in self.warnings[account_name]:
                error_message += "\n\t%s" % (file)
        return error_message
