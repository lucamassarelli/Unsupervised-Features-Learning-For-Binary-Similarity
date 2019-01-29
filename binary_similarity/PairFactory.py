from binary_similarity.embedder import Embedder
import json
from multiprocessing import Queue
import networkx as nx
from networkx import json_graph
import numpy as np
from scipy import sparse
import sqlite3
from threading import Thread
from binary_similarity.utils import __padAndFilterLSTM as padAndFilterLSTM
from binary_similarity.utils import __padAndFilter as padAndFilter
from binary_similarity.InstructionsConverter import InstructionsConverter
from binary_similarity.FunctionNormalizer import FunctionNormalizer


class DatasetGenerator:

    def get_dataset(self, epoch_number):
        pass


class PairFactory(DatasetGenerator):

    def __init__(self, db_name, feature_type, dataset_type, json_asm2id, max_instructions, max_num_vertices):
        self.db_name = db_name
        self.feature_type = feature_type
        self.dataset_type = dataset_type
        self.max_instructions = max_instructions
        self.max_num_vertices = max_num_vertices
        self.batch_dim = 0
        self.num_pairs = 0
        self.num_batches = 0
        self.converter = InstructionsConverter(json_asm2id)
        self.normalizer = FunctionNormalizer(self.max_instructions)

    def get_data_from_cfg(self, cfg):
        adj = sparse.csr_matrix([1,1])
        lenghts = []
        node_matrix = []

        try:
            adj = nx.adjacency_matrix(cfg)
            nodes = cfg.nodes(data=True)
            for i, n in enumerate(nodes):
                filtered = self.converter.convert_to_ids(n[1]['features'])
                lenghts.append(len(filtered))
                node_matrix.append(self.normalizer.normalize(filtered)[0])
        except:
            pass
        return adj, node_matrix, lenghts

    def remove_bad_acfg_node(self, g):
        nodeToRemove = []
        for n in g.nodes(data=True):
            f = n[1]['features']
            if len(f.keys()) == 0:
                nodeToRemove.append(n[0])
        for n in nodeToRemove:
            g.remove_node(n)
        return g

    def get_node_matrix(self, nodes):
        num_node = len(nodes)
        node_matrix = np.zeros([num_node, 8])
        for i, n in enumerate(nodes):
            f = n[1]['features']
            if isinstance(f['constant'], list):
                node_matrix[i, 0] = len(f['constant'])
            else:
                node_matrix[i, 0] = f['constant']
            if isinstance(f['string'], list):
                node_matrix[i, 1] = len(f['string'])
            else:
                node_matrix[i, 1] = f['string']
            node_matrix[i, 2] = f['transfer']
            node_matrix[i, 3] = f['call']
            node_matrix[i, 4] = f['instruction']
            node_matrix[i, 5] = f['arith']
            node_matrix[i, 6] = f['offspring']
            node_matrix[i, 7] = f['betweenness']
        return node_matrix

    def get_data_from_acfg(self, g):
        g = self.remove_bad_acfg_node(g)
        if len(g.nodes) > 0:
            adj = nx.adjacency_matrix(g)
            node_matrix = self.get_node_matrix(g.nodes(data=True))
        else:
            adj = sparse.bsr_matrix(np.zeros([1, 1]))
            node_matrix = np.zeros([1, 8])
        lenght = 8
        return adj, node_matrix, lenght

    def async_chunker(self, epoch, number_of_pairs, shuffle=True):
        self.num_pairs = 0

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        q = cur.execute("SELECT true_pair, false_pair from " + self.dataset_type + " WHERE id=?", (int(epoch),))
        true_pairs_id, false_pairs_id = q.fetchone()
        true_pairs_id = json.loads(true_pairs_id)
        false_pairs_id = json.loads(false_pairs_id)

        assert len(true_pairs_id) == len(false_pairs_id)
        data_len = len(true_pairs_id)

        print("Data Len: " + str(data_len))
        conn.close()

        n_chunk = int(data_len / (number_of_pairs/2)) - 1
        self.num_batches = n_chunk

        q = Queue(maxsize=50)

        t = Thread(target=self.async_create_pairs, args=(epoch, n_chunk, number_of_pairs, q))
        t.start()

        for i in range(0, n_chunk):
            yield self.async_get_dataset(i, n_chunk, number_of_pairs, q, shuffle)

    def get_pair_from_db(self, epoch_number, chunk, number_of_pairs):

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()

        pairs = []
        labels = []
        lenghts = []

        q = cur.execute("SELECT true_pair, false_pair from " + self.dataset_type + " WHERE id=?", (int(epoch_number),))
        true_pairs_id, false_pairs_id = q.fetchone()

        true_pairs_id = json.loads(true_pairs_id)
        false_pairs_id = json.loads(false_pairs_id)

        data_len = len(true_pairs_id)

        i = 0

        while i < number_of_pairs:
            if chunk * int(number_of_pairs/2) + i > data_len:
                break

            p = true_pairs_id[chunk * int(number_of_pairs/2) + i]
            q0 = cur.execute("SELECT " + self.feature_type + " FROM " + self.feature_type + " WHERE id=?", (p[0],))
            if self.feature_type == 'acfg':
                adj0, node0, lenghts0 = self.get_data_from_acfg(json_graph.adjacency_graph(json.loads(q0.fetchone()[0])))
            elif self.feature_type == 'lstm_cfg':
                adj0, node0, lenghts0 = self.get_data_from_cfg(json_graph.adjacency_graph(json.loads(q0.fetchone()[0])))

            q1 = cur.execute("SELECT " + self.feature_type + " FROM " + self.feature_type + " WHERE id=?", (p[1],))
            if self.feature_type == 'acfg':
                adj1, node1, lenghts1 = self.get_data_from_acfg(json_graph.adjacency_graph(json.loads(q0.fetchone()[0])))
            elif self.feature_type == 'lstm_cfg':
                adj1, node1, lenghts1 = self.get_data_from_cfg(json_graph.adjacency_graph(json.loads(q1.fetchone()[0])))

            pairs.append(((adj0, node0), (adj1, node1)))
            lenghts.append([lenghts0, lenghts1])
            labels.append(+1)

            p = false_pairs_id[chunk * int(number_of_pairs/2) + i]
            q0 = cur.execute("SELECT " + self.feature_type + " FROM " + self.feature_type + " WHERE id=?", (p[0],))
            if self.feature_type == 'acfg':
                adj0, node0,lenghts0 = self.get_data_from_acfg(json_graph.adjacency_graph(json.loads(q0.fetchone()[0])))
            elif self.feature_type == 'lstm_cfg':
                adj0, node0, lenghts0 = self.get_data_from_cfg(json_graph.adjacency_graph(json.loads(q0.fetchone()[0])))

            q1 = cur.execute("SELECT " + self.feature_type + " FROM " + self.feature_type + " WHERE id=?", (p[1],))
            if self.feature_type == 'acfg':
                adj1, node1, lenghts1 = self.get_data_from_acfg(json_graph.adjacency_graph(json.loads(q0.fetchone()[0])))
            elif self.feature_type == 'lstm_cfg':
                adj1, node1, lenghts1 = self.get_data_from_cfg(json_graph.adjacency_graph(json.loads(q1.fetchone()[0])))

            pairs.append(((adj0, node0), (adj1, node1)))
            lenghts.append([lenghts0, lenghts1])
            labels.append(-1)

            i += 2
        if self.feature_type == 'acfg':
            pairs, labels, output_len = padAndFilter(pairs, labels, self.max_num_vertices)
        elif self.feature_type == 'lstm_cfg':
            pairs, labels, output_len = padAndFilterLSTM(pairs, labels, lenghts, self.max_num_vertices)
        return pairs, labels, output_len

    def async_create_pairs(self, epoch, n_chunk, number_of_pairs, q):
        for i in range(0, n_chunk):
            pairs, y_, lenghts = self.get_pair_from_db(epoch, i, number_of_pairs)
            q.put((pairs, y_, lenghts), block=True)

    def async_get_dataset(self, chunk, n_chunk, number_of_pairs, q, shuffle):

        item = q.get()
        pairs = item[0]
        y_ = item[1]
        lenghts = item[2]

        assert (len(pairs) == len(y_))
        n_samples = len(pairs)
        self.batch_dim = n_samples
        self.num_pairs += n_samples

        # Unpack the list
        graph1, graph2 = zip(*pairs)
        len1, len2 = zip(*lenghts)
        adj1, nodes1 = zip(*graph1)
        adj2, nodes2 = zip(*graph2)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(n_samples))
            adj1 = np.array(adj1)[shuffle_indices]
            nodes1 = np.array(nodes1)[shuffle_indices]
            adj2 = np.array(adj2)[shuffle_indices]
            nodes2 = np.array(nodes2)[shuffle_indices]
            y_ = np.array(y_)[shuffle_indices]

        for i in range(0, n_samples, number_of_pairs):
            upper_bound = min(i + number_of_pairs, n_samples)

            ret_adj1 = adj1[i:upper_bound]
            ret_nodes1 = nodes1[i:upper_bound]
            ret_len1=len1[i:upper_bound]
            ret_adj2 = adj2[i:upper_bound]
            ret_nodes2 = nodes2[i:upper_bound]
            ret_len2 = len2[i:upper_bound]
            ret_y = y_[i:upper_bound]

            return ret_adj1, ret_nodes1, ret_adj2, ret_nodes2, ret_y, ret_len1, ret_len2