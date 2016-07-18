import sqlite3 

class sirlog(object):
    """
    Class for tracking simulation and recording information in databs

    :param str db_path: path to db to use for tracking
    """
    def __init__(self, log_path, name, data):
        self.db = sqlite3.connect(log_path)
        self.cursor = self.db.cursor()
        self.name = name
        self.data = data

    def create_table(self):
        """
        Create table in db

        :param str name: table name
        :param list data: Fields and data types for columns in table
        """
        self.cursor.execut('''CREATE TABLE''' + self.name + '''(''' +
                           " ".join(self.data) + ''')'''
                           )
        self.db.commit()

    def insert(self, row):
        """
        insert row(s) into table
        :param 
        """

