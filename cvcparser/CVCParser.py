#!/usr/bin/env python2.7

import numpy as np
import regex
import os
import string
import argparse
import ntpath


class LogFile(object):
    """
    Class for processing id sync and referrer log files 

    """
    
    def __init__(self, input_file, output_folder='./', init_split='ElmoLogEntry(', 
                 end_split=")", fields=None):
        """
        Constructor for LogFile
        
        :param str input_file: file name to log file
        :param str output_folder: path to output folder (you must have write access here) 
        :param str init_split: Initial split for marking beginning of line in log
        :param str end_split: End split for marking end of line in log
        :param array-like fields: fields to output. None will produce all fields in file
        """
        
        self.file_path =  os.path.abspath(input_file)  
        filename = path_leaf(self.file_path)
        self.output_folder =  os.path.abspath(output_folder + '/processed_logs')  
        self.handle = open(self.file_path, 'r')
                            

        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        
        if fields:
            self.field_select = True
            if not isinstance(fields, list):
                fields = list([fields])
            self.output_file = self.output_folder + '/' + filename \
                               + '_' + "_".join(fields) + '.processed'
        else:
            self.field_select = False
            fields = []
            self.output_file =  self.output_folder + '/' + filename + '.processed'

        self.fields = fields 
        self.data = {}

        self.sep_dict = { "(": ["\(","\)"]}

        self.init_split = init_split
        self.end_split = end_split
        self.build_txt()

    def build_txt(self):
        """
        """
        self.txt_out = open(self.output_file, 'w')
        if self.field_select:
            self.txt_out.write(" | ".join(self.fields) + '\n')        

        for line in self.handle:
            #Trim begginning of line
            line = line.split(self.init_split)[1]
            #Trim end of line
            line = line.rsplit(self.end_split ,1)[0]
            #Parse line
            self.read_line(line)
            if not self.field_select:
                self.fields.extend([i for i in self.data if i not in self.fields])    
            
            write_out = []
            for field in self.fields:
                try:
                    write_out.append(self.data[field])
                except KeyError:
                    write_out.append("")
            self.data = {}        
            self.txt_out.write(" | ".join(write_out) + '\n')

        if not self.field_select: 
            header_file = open(self.output_folder+'/headers','w')
            header_file.write(" | ".join(self.fields) + '\n')        
            header_file.close()
        self.txt_out.close()


    def read_line(self, line):
       
        end_line = False
                   
        table = string.maketrans('[]', '()')
        line = line.translate(table) 
        table = string.maketrans('{}', '()')
        line = line.translate(table) 
        #table = string.maketrans('<>', '()')
        #line = line.translate(table) 

        while line: 
            
            [key, l] = line.split(':', 1)
            if '=' in key:
                print(line)
                raw_input()
            sep = find_sep(l)
            if sep == ',':
                try:
                    [self.data[key], line] = l.split(', ', 1)
                    #line = l.split(self.data[key]+ ', ', 1)[1]
                except ValueError:
                    self.data[key] = l
                    end_line = True
            elif sep:
                self.data[key], line = find_end_iter(l, self.sep_dict)
                #self.data[key] = l.split(value, 1)[0] + value  
            else:
                line = None

    def read_line_fields(self, line):

        table = string.maketrans('[]', '()')
        line = line.translate(table) 
        table = string.maketrans('{}', '()')
        line = line.translate(table) 
        #table = string.maketrans('<>', '()')
        #line = line.translate(table) 

        
        for field in self.fields:
            if " " + field + ":" in line:
                l = line.split(' ' + field + ':', 1)[1]
                sep = find_sep(l)
                if sep == ',':
                    try:
                        [self.data[field], rest] = l.split(',', 1)
                    except ValueError:
                        self.data[field] = l
                        end_line = True
                else:
                    self.data[field], null = find_end_iter(l, self.sep_dict)
                    #self.data[key] = l.split(value, 1)[0] + value  
            else:
                self.data[field] = ""





    def gen_empty_write_record(self):
        """
        """
        self.write_record = np.array(
                                     tuple([' '] * len(self.fields)),
                                         dtype=self.formats
                                    )

def find_end_iter(l, sep_dict):
    """
    """
    exposed_comma = False
    hold=[]
    
    if ',' not in l:
        return l, ""

    while not exposed_comma: 
        if '(' in l:
            if l.index('(') > l.index(','): 
                l1, l = l.split(', ',1)
                hold.extend(l1)
                exposed_comma = True
            else:
                hold_value = regex_gen(l, sep_dict["("])    
                value = l.split(hold_value,1)[0] + hold_value 
                
                hold.extend(value)
                l = l.split(value,1)[1]
        else:
            exposed_comma = True
    return "".join(hold).rstrip(','), l

def regex_gen(txt, seps): 
    [sep1, sep2] = seps
    result = regex.search(r'''
                            (?<rec> #capturing group rec
                             ''' + sep1 + ''' #open parenthesis
                             (?: #non-capturing group
                              [^()]++ #anyting but parenthesis one or more times without backtracking
                              | #or
                               (?&rec) #recursive substitute of group rec
                             )*
                             ''' + sep2 + '''  #close parenthesis
                            )
                           ''',txt,flags=regex.VERBOSE)
    return result.captures('rec')[-1]



def find_sep(l):
    """
    """

    d = {}
    if '(' in l:
        d['('] = l.index('(')


    if ',' in l:
        d[','] = l.index(',')

    
    try:
        key = min(d, key=d.get)
    except ValueError:
        key = None
    return key

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def path_leaf(abspath):
    """
    """
    folder, file = ntpath.split(abspath)
    return file or ntpath.basename(file)

def main():
    """
    """
    parser = argparse.ArgumentParser(description=' IMPORTANT! The above use spec is incorrect, in PYTHON positional arguments come before keyword arguments so you must specify the log file before the options because the log file is required it is a positional argument. Sorry for any confusion. \n \n Processer to turn referrer and similar type log files into pipe seperated files.\n \n Note if you do not prescribe fields the processor adds new fields line by line so if all the fields are not in the first line the last line will have more fields then the first. Prescribing fields guarantees a rectangular file. \n Note: All braces "{}" brackets "[]" and arrows "<>" are converted to parenthesis "()". Next update will try to work around this. \n \n Note: The output will be placed inside a folder named "processed" inside the folder specified by the output folder argument(default ./). If fields are not prescribed the field names will be written to an additional file named "header". If field names are prescribed the column names will appear at the top of the file.\n \n For bug/feauture requests please contact mweiss@liveramp.com. \n \n \n' + '*'*30 + '\n SIMPLEST use without fields option: python LogFile.py <file> will process all fields in the log and place the processed log and a header file in ./processed/ \n \n SIMPLEST use with fields option: python LogFile.py <file> -f <fieldname1> <fieldname2> will place the processed log (with header at the top) in ./processed/ \n'+ '*'*30 , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input_file", type=str, help='name of logfile')
    parser.add_argument("-of", "--output_folder", type=str, default='./', help='path to output directory defaults to ./  this script will make a folder in here for outputs so you need to have write access')
    parser.add_argument("-is", "--init_split", type=str, default='ElmoLogEntry(', help='marks the beginning of a line , default is "ElmoLogEntry("')

    parser.add_argument("-es", "--end_split", type=str, default=')>)', help='marks the end of a line , default is ")>)"')
    parser.add_argument("-f", "--fields", nargs='+', type=str, help='fields to output. Default will produce all fields in file, NB file will not be rectangular as fields are added as they are found. Specifiying fields will include a header line in the file. Not specifying fields will output a headers file in the output directory')
    
    args = parser.parse_args()

    lf = LogFile(args.input_file, args.output_folder, args.init_split, args.end_split, args.fields)
if __name__ == '__main__':
    main()
