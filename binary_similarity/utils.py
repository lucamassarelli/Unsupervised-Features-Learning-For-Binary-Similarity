import numpy as np
from scipy.sparse import csr_matrix
# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#


def __padAndFilter(input_pairs, input_labels, max_num_vertices):

    output_pairs = []
    output_labels = []
    output_len = []

    for pair, label in zip(input_pairs, input_labels):
        g1 = pair[0]
        g2 = pair[1]

        # graph 1
        adj1 = g1[0]
        nodes1 = g1[1]

        # graph 2
        adj2 = g2[0]
        nodes2 = g2[1]

        if (len(nodes1) <= max_num_vertices) and (len(nodes2) <= max_num_vertices):
            pad_lenght1 = max_num_vertices - len(nodes1)
            new_node1 = np.pad(nodes1, [(0, pad_lenght1), (0, 0)], mode='constant')
            pad_lenght1 = max_num_vertices - adj1.shape[0]
            adj1_dense = np.pad(adj1.todense(), [(0, pad_lenght1), (0, pad_lenght1)], mode='constant')
            g1 = (adj1_dense, new_node1)
            pad_lenght2 = max_num_vertices - len(nodes2)
            new_node2 = np.pad(nodes2, [(0, pad_lenght2), (0, 0)], mode='constant')
            pad_lenght2 = max_num_vertices - adj2.shape[0]
            adj2_dense = np.pad(adj2.todense(), [(0, pad_lenght2), (0, pad_lenght2)], mode='constant')
            g2 = (adj2_dense, new_node2)
            output_pairs.append([g1, g2])
            output_labels.append(label)
            output_len.append([8,8])

    return output_pairs, output_labels, output_len

def __padAndFilterLSTM(input_pairs, input_labels, input_len, max_num_vertices):


    output_pairs = []
    output_labels = []
    output_len=[]

    for pair, label, lens in zip(input_pairs, input_labels, input_len):

        try:

            g1 = pair[0]
            g2 = pair[1]

            # graph 1
            adj1 = g1[0]
            nodes1 = g1[1]

            # graph 2
            adj2 = g2[0]
            nodes2 = g2[1]
            if (len(nodes1) <= max_num_vertices) and (len(nodes2) <= max_num_vertices):

                pad_lenght1 = max_num_vertices - len(nodes1)
                new_node1 = np.pad(nodes1, [(0, pad_lenght1), (0, 0)], mode='constant')

                pad_lenght1 = max_num_vertices - adj1.shape[0]
                adj1_dense = np.pad(adj1.todense(), [(0, pad_lenght1), (0, pad_lenght1)], mode='constant')
                g1 = (adj1_dense, new_node1)

                pad_lenght2 = max_num_vertices - len(nodes2)
                new_node2 = np.pad(nodes2, [(0, pad_lenght2), (0, 0)], mode='constant')
                pad_lenght2 = max_num_vertices - adj2.shape[0]
                adj2_dense = np.pad(adj2.todense(), [(0, pad_lenght2), (0, pad_lenght2)], mode='constant')
                g2 = (adj2_dense, new_node2)

                output_pairs.append([g1, g2])
                output_labels.append(label)
                new_lens_0 = lens[0]+[0]*(max_num_vertices-len(lens[0]))
                new_lens_1 = lens[1]+[0]*(max_num_vertices-len(lens[1]))
                output_len.append([new_lens_0, new_lens_1])

        except:
            pass

    return output_pairs, output_labels, output_len
