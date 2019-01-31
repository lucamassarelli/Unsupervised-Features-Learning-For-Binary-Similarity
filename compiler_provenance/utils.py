# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import numpy as np


def __padAndFilter(input_pairs, input_labels, input_len, max_num_vertices):

    output_pairs = []
    output_labels = []
    output_len = []

    for pair, label, lens in zip(input_pairs, input_labels, input_len):
        try:
            g1 = pair[0]

            # graph 1
            adj1 = g1[0]
            nodes1 = g1[1]

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

                new_lens_0 = lens + [0]*(max_num_vertices-len(lens))
                output_len.append([new_lens_0])
        except:
            pass

    return output_pairs, output_labels, output_len

