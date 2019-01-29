# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
from FunctionAnalyzerRadare import RadareFunctionAnalyzer
import json
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os
import random
import signal
import sqlite3
from tqdm import tqdm
from networkx.readwrite import json_graph


class DatabaseFactory:

    def __init__(self, db_name, root_path):
        self.db_name = db_name
        self.root_path = root_path

    @staticmethod
    def worker(item):
        DatabaseFactory.analyze_file(item)
        return 0

    @staticmethod
    def extract_function(graph_analyzer):
        return graph_analyzer.extractAll()

    @staticmethod
    def to_jsongraph(graph):
        return json.dumps(json_graph.adjacency_data(graph))

    @staticmethod
    def insert_in_db(db_name, pool_sem, func, filename, function_name):
        path = filename.split(os.sep)
        if len(path) < 4:
            return
        pool_sem.acquire()
        conn = sqlite3.connect(db_name)
        cfg = DatabaseFactory.to_jsongraph(func["cfg"])
        cur = conn.cursor()
        cur.execute('''INSERT INTO functions VALUES (?,?,?,?,?,?,?)''', (None,            # id
                                                                         path[-4],      # project
                                                                         path[-3],      # compiler
                                                                         path[-2],      # optimization
                                                                         path[-1],      # file_name
                                                                         function_name, # function_name
                                                                         cfg))

        inserted_id = cur.lastrowid
        acfg = DatabaseFactory.to_jsongraph(func["acfg"])
        lstm_cfg = DatabaseFactory.to_jsongraph(func["lstm_cfg"])

        cur.execute('''INSERT INTO acfg VALUES (?,?)''', (inserted_id, acfg))
        conn.commit()
        cur.execute('''INSERT INTO lstm_cfg VALUES (?,?)''', (inserted_id, lstm_cfg))
        conn.commit()

        conn.close()
        pool_sem.release()

    @staticmethod
    def analyze_file(item):
        global pool_sem
        os.setpgrp()

        filename = item[0]
        db = item[1]
        use_symbol = item[2]

        analyzer = RadareFunctionAnalyzer(filename, use_symbol)
        p = ThreadPool(1)
        res = p.apply_async(analyzer.analyze)

        try:
            result = res.get(120)
        except multiprocessing.TimeoutError:
                print("Aborting due to timeout:" + str(filename))
                print('Try to modify the timeout value in DatabaseFactory instruction  result = res.get(TIMEOUT)')
                os.killpg(0, signal.SIGKILL)
        except Exception:
                print("Aborting due to error:" + str(filename))
                os.killpg(0, signal.SIGKILL)

        for func in result:
            DatabaseFactory.insert_in_db(db, pool_sem, result[func], filename, func)

        analyzer.close()

        return 0

    # Create the db where data are stored
    def create_db(self):
        print('Database creation...')
        conn = sqlite3.connect(self.db_name)
        conn.execute(''' CREATE TABLE  IF NOT EXISTS functions (id INTEGER PRIMARY KEY, 
                                                                project text, 
                                                                compiler text, 
                                                                optimization text, 
                                                                file_name text, 
                                                                function_name text,
                                                                cfg text)''')

        conn.execute('''CREATE TABLE  IF NOT EXISTS acfg  (id INTEGER PRIMARY KEY, acfg text)''')
        conn.execute('''CREATE TABLE  IF NOT EXISTS lstm_cfg  (id INTEGER PRIMARY KEY, lstm_cfg text)''')

        conn.commit()
        conn.close()

    # Scan the root directory to find all the file to analyze,
    # query also the db for already analyzed files.
    def scan_for_file(self, start):
        file_list = []
        # Scan recursively all the subdirectory
        directories = os.listdir(start)
        for item in directories:
            item = os.path.join(start,item)
            if os.path.isdir(item):
                file_list.extend(self.scan_for_file(item + os.sep))
            elif os.path.isfile(item) and item.endswith('.o'):
                file_list.append(item)
        return file_list

    # Looks for already existing files in the database
    # It returns a list of files that are not in the database
    def remove_override(self, file_list):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        q = cur.execute('''SELECT project, compiler, optimization, file_name FROM functions''')
        names = q.fetchall()
        names = [os.path.join(self.root_path, n[0], n[1], n[2], n[3]) for n in names]
        names = set(names)
        # If some files is already in the db remove it from the file list
        if len(names) > 0:
            print(str(len(names)) + ' Already in the database')
        cleaned_file_list = []
        for f in file_list:
            if not(f in names):
                cleaned_file_list.append(f)

        return cleaned_file_list

    # root function to create the db
    def build_db(self, use_symbol):
        global pool_sem

        pool_sem = multiprocessing.BoundedSemaphore(value=1)

        self.create_db()
        file_list = self.scan_for_file(self.root_path)

        print('Found ' + str(len(file_list)) + ' during the scan')
        file_list = self.remove_override(file_list)
        print('Find ' + str(len(file_list)) + ' files to analyze')
        random.shuffle(file_list)

        t_args = [(f, self.db_name, use_symbol) for f in file_list]

        # Start a parallel pool to analyze files
        p = Pool(processes=None, maxtasksperchild=20)
        for _ in tqdm(p.imap_unordered(DatabaseFactory.worker, t_args), total=len(file_list)):
            pass

        p.close()
        p.join()


