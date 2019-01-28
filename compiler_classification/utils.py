# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import numpy as np
import itertools
import pickle as pkl
import sys
from scipy.sparse import csr_matrix

porco_dio = 0
all_zeros = 0

def __loadPickle(filename):
    with open(filename, 'rb') as handle:
        b = pkl.load(handle)
    pairs = []
    labels = []
    for x in b:
        for label, pair in x.items():
            pairs.append(pair)
            labels.append(label)
    return pairs, labels

def check_if_val_in_test():

    val_pairs, val_labels = __loadPickle("partial_data/validationData.pkl")
    test_pairs, test_labels = __loadPickle("partial_data/testData.pkl")
    eq=0
    i=0
    for x in val_pairs:
        sys.stdout.write(
            "\r\tpercentage {}".format(float(i)/float(len(val_pairs))))
        i+=1
        for y in test_pairs:
            if np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1]):
                print("dentro"+str(eq))
                eq+=1

    return eq


# Filtra le istruzioni nel pickle e le sostistuisce con gli IDs
# contenuti in embedder, faccio il padding a max_instruction o a filter.
# e ritorno una lista di pairs e una lista di labels

def __convertInstructions(pairs, labels, embedder, filter=None):
    new_pairs=[]
    lenghts=[]
    for p in pairs:
        g0 = p[0]
        nodes0 = g0[1]
        g0_new = [0]*2
        g0_new[0] = g0[0]
        nodes0_new = []
        lenp0 = []
        for x in nodes0:
            filter_instructions=embedder.toIds(x)

            filter_instructions.extend([embedder.getPaddingIndex()]*(filter-len(filter_instructions)))
            if filter:
                filter_instructions=filter_instructions[0:filter]
                lenp0.append(min(len(filter_instructions),filter))
            else:
                lenp0.append(len(filter_instructions))
            nodes0_new.append(filter_instructions)

        g0_new[1] = nodes0_new

        new_pairs.append([g0_new])
        lenghts.append([lenp0])
        assert(len(nodes0_new) == g0[0].shape[0])

    return new_pairs, labels, lenghts

def convertDataToLstm(input_train_pairs, input_train_labels, input_val_pairs, input_val_labels, input_test_pairs,
                     input_test_labels, embedder,max_inst=None):
    msg = "converting...\n\ttrain:{}\n\tval:{}\n\ttest:{}\n".format(len(input_train_labels),
                                                                             len(input_val_labels),
                                                                             len(input_test_labels))

    # 1. train
    train_pairs, train_labels,train_len = __convertInstructions(input_train_pairs, input_train_labels, embedder,max_inst)

    # 2. validation
    val_pairs, val_labels, val_len = __convertInstructions(input_val_pairs, input_val_labels, embedder,max_inst)

    # 3. test
    test_pairs, test_labels, test_len = __convertInstructions(input_test_pairs, input_test_labels, embedder,max_inst)


    print(msg)

    return train_pairs, train_labels,train_len, val_pairs, val_labels,val_len, test_pairs, test_labels, test_len




def loadData(train_filename, val_filename=None, test_filename=None, val_percentage=0.1, test_percentage=0.1):
    # 1. train set
    train_pairs, train_labels = __loadPickle(train_filename)
    n = len(train_labels)

    # 2. validation set
    if val_filename:
        val_pairs, val_labels = __loadPickle(val_filename)
    else:
        # take last val_percentage as validation set
        cut = int(n * val_percentage)
        val_pairs = train_pairs[-cut:]
        val_labels = train_labels[-cut:]
        train_pairs = train_pairs[:-cut]
        train_labels = train_labels[:-cut]

    # 3. test set
    if test_filename:
        test_pairs, test_labels = __loadPickle(test_filename)
    else:
        # take last test_percentage as test set
        cut = int(n * test_percentage)
        test_pairs = train_pairs[-cut:]
        test_labels = train_labels[-cut:]
        train_pairs = train_pairs[:-cut]
        train_labels = train_labels[:-cut]

    # print("train: {}".format(train_labels))
    # print("val: {}".format(val_labels))
    # print("test: {}".format(test_labels))

    return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels


def datasetStats(train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels):
    msg = ""
    max_num_vertices = 0
    # test Statistics to be sure that are the same of Table2 in the paper
    msg += "  Statistics:\n"
    assert (len(train_pairs) == len(train_labels))
    assert (len(val_pairs) == len(val_labels))
    assert (len(test_pairs) == len(test_labels))
    msg += "\tsize train: {}\n".format(len(train_labels))
    msg += "\tsize val: {}\n".format(len(val_labels))
    msg += "\tsize test: {}\n".format(len(test_labels))
    # check average number of vertices and edges
    sum_num_vertices = 0
    sum_num_edges = 0
    forest_graph=0
    sum_no_edges=0
    num_graphs = 0
    num_vertices_list = []
    for pair in itertools.chain(train_pairs, val_pairs, test_pairs):
        for graph in pair:
            adj = graph[0]  # matrice_di_adicenza
            # print("{} - {}".format(graph[0].shape,graph[1].shape))
            num_vertices = adj.shape[0]
            if num_vertices > max_num_vertices:
                max_num_vertices = num_vertices
            num_edges = adj.count_nonzero()
            sum_num_vertices += num_vertices
            sum_num_edges += num_edges
            num_vertices_list.append(num_vertices)
            num_graphs += 1
            if (num_vertices > 1) and num_edges==0:
                forest_graph+=1
            if num_edges==0:
                sum_no_edges+=1

    # histogram number of vertices
    # import matplotlib.pyplot as plt
    # plt.hist(num_vertices_list, bins='auto')
    # plt.title("num_vertices histogram")
    # plt.show()
    # sys.exit(-1)

    msg += "\tnum_graphs: {}\n".format(num_graphs)
    msg += "\taverage number of vertices: {}\n".format(sum_num_vertices / num_graphs)
    msg += "\taverage number of edges: {}\n".format((sum_num_edges / num_graphs) / 2)  # symmetric matrix
    msg += "\tnumber of graphs without edges but with 2 or more nodes:{}\n".format(forest_graph)
    msg += "\tnumber of graphs without edges and with 1 node:{}\n".format(sum_no_edges)
    # tets labels
    labels = set()
    for value in itertools.chain(train_labels, val_labels, test_labels):
        labels.add(value)
    msg += "\tnum_labels: {}\n".format(len(labels))
    nodes = set()
    num_graphs = 0
    for pair in itertools.chain(train_pairs, val_pairs, test_pairs):
        for graph in pair:
            list_nodes = graph[1]  # matrice_dei_nodi
            for node in list_nodes:
                value = np.where(node == 1)
                if (np.shape(value[0])[0] > 0):
                    nodes.add(value[0][0])
    # print("{} -> {}".format(node,value)
    msg += "\tnum_nodes: {}\n".format(len(nodes))
    msg += "\tmax_num_vertices: {}\n".format(max_num_vertices)
    return max_num_vertices, msg


def __padAndFilter(input_pairs, input_labels,input_len, max_num_vertices, min_num_vertices):

    output_pairs = []
    output_labels = []
    output_len=[]
    for pair, label,lens in zip(input_pairs, input_labels,input_len):
        g1 = pair[0]

        # graph 1
        adj1 = g1[0]
        nodes1 = g1[1]

        #print("Max vertex"+str(max_num_vertices))
        if len(nodes1) <= max_num_vertices:
            # graph 1
            pad_lenght1 = max_num_vertices - len(nodes1)
            new_node1 = np.pad(nodes1, [(0, pad_lenght1), (0, 0)], mode='constant')
            pad_lenght1 = max_num_vertices - adj1.shape[0]
            # pass to dense for padding
            adj1_dense = np.pad(adj1.todense(), [(0, pad_lenght1), (0, pad_lenght1)], mode='constant')

            g1 = (adj1_dense, new_node1)
            output_pairs.append([g1])
            output_labels.append(label)

            new_lens_0 = lens[0]+[0]*(max_num_vertices-len(lens[0]))
            output_len.append([new_lens_0])


    return output_pairs, output_labels,output_len

def padAndFilterData(input_train_pairs, input_train_labels, input_val_pairs, input_val_labels, input_test_pairs,
                     input_test_labels, max_num_vertices,min_num_vertices):
    msg = "filtering...\n  before\n\ttrain:{}\n\tval:{}\n\ttest:{}\n".format(len(input_train_labels),
                                                                             len(input_val_labels),
                                                                             len(input_test_labels))

    # 1. train
    train_pairs, train_labels = __padAndFilter(input_train_pairs, input_train_labels, max_num_vertices,min_num_vertices)

    # 2. validation
    val_pairs, val_labels = __padAndFilter(input_val_pairs, input_val_labels, max_num_vertices,min_num_vertices)

    # 3. test
    test_pairs, test_labels = __padAndFilter(input_test_pairs, input_test_labels, max_num_vertices,min_num_vertices)


    msg += "  after\n\ttrain:{}\n\tval:{}\n\ttest:{}".format(len(train_labels), len(val_labels), len(test_labels))

    print(msg)

    return train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels





def padAndFilterDataLSTM(input_train_pairs, input_train_labels, input_train_len,input_val_pairs, input_val_labels,
                     input_val_len,
                     input_test_pairs,
                     input_test_labels,
                     input_test_len,
                     max_num_vertices,min_num_vertices):
    msg = "filtering...\n  before\n\ttrain:{}\n\tval:{}\n\ttest:{}\n".format(len(input_train_labels),
                                                                             len(input_val_labels),
                                                                             len(input_test_labels))

    # 1. train
    train_pairs, train_labels,train_len = __padAndFilterLSTM(input_train_pairs, input_train_labels,input_train_len, max_num_vertices,min_num_vertices)

    # 2. validation
    val_pairs, val_labels, val_len = __padAndFilterLSTM(input_val_pairs, input_val_labels,input_val_len, max_num_vertices,min_num_vertices)

    # 3. test
    test_pairs, test_labels, test_len = __padAndFilterLSTM(input_test_pairs, input_test_labels,input_test_len, max_num_vertices,min_num_vertices)


    msg += "  after\n\ttrain:{}\n\tval:{}\n\ttest:{}".format(len(train_labels), len(val_labels), len(test_labels))

    print(msg)

    return train_pairs, train_labels,train_len, val_pairs, val_labels,val_len, test_pairs, test_labels,test_len


if __name__ == '__main__':
    data = loadData(
        train_filename ="partial_data/trainData.pkl" ,
        val_filename = "partial_data/validationData.pkl",
        test_filename = "partial_data/testData.pkl")
    max_num_vertices, msg = datasetStats(*data)
    print(msg)