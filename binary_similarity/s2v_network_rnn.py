# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import tensorflow as tf

class NetworkLSTM:

    def __init__(self,
                 features_size,
                 embedding_size,
                 max_lv,
                 T_iterations,
                 learning_rate,
                 l2_reg_lambda,
                 batch_size,
                 max_instructions,
                 max_nodes,
                 rnn_depth,
                 rnn_kind,
                 embedding_matrix,
                 trainable_embeddings
                 ):
        print("Features size"+str(features_size))
        self.features_size = features_size
        self.embedding_size = embedding_size
        self.max_lv = max_lv
        self.T_iterations = T_iterations
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.RRN_HIDDEN = features_size
        self.batch_size = batch_size
        self.max_instructions = max_instructions
        self.max_nodes = max_nodes
        self.rnn_depth = rnn_depth
        self.rnn_kind=rnn_kind
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.generateGraphClassificationNetwork()

    def extract_axis_1(self, data, ind):
        """
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """
        ind=tf.nn.relu(ind-1)
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)

        return res

    def lstmFeatures(self, input_x, lengths):

        flattened_inputs=tf.reshape(input_x,[-1,tf.shape(input_x)[2]],name="Flattening")

        flattened_lenghts = tf.reshape(lengths, [-1])
        max = tf.reduce_max(flattened_lenghts)
        flattened_inputs=flattened_inputs[:,:max]
        flattened_embedded = tf.nn.embedding_lookup(self.instruction_embeddings_t, flattened_inputs)

        zeros = tf.zeros(tf.shape(flattened_lenghts)[0], dtype=tf.int32)
        mask = tf.not_equal(flattened_lenghts, zeros)
        int_mask = tf.cast(mask, tf.int32)
        fake_output = tf.zeros([self.features_size], dtype=tf.float32)
        partitions = tf.dynamic_partition(flattened_embedded, int_mask, 2)
        real_nodes=partitions[1]
        real_lenghts=tf.boolean_mask(flattened_lenghts,mask)
        fake_zero = tf.tile([fake_output], [tf.shape(flattened_embedded)[0] - tf.shape(partitions[1])[0], 1])

        if self.rnn_kind==0:
            rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in ([self.features_size] * self.rnn_depth)]
        else:
            rnn_layers = [tf.nn.rnn_cell.GRUCell(size) for size in ([self.features_size] * self.rnn_depth)]
        cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        rnn_outputs, _ = tf.nn.dynamic_rnn(cell, real_nodes, sequence_length=real_lenghts,
                                           dtype=tf.float32,
                                           time_major=False,
                                           parallel_iterations=88)

        last_outputs = self.extract_axis_1(rnn_outputs, real_lenghts)

        condition_indices = tf.dynamic_partition(
            tf.range(tf.shape(flattened_embedded)[0]), int_mask, 2)
        last_outputs = tf.dynamic_stitch(condition_indices, [fake_zero, last_outputs])

        gather_output2 = tf.reshape(last_outputs,
                                      [-1,tf.shape(input_x)[1],self.features_size], name="Deflattening")

        output = tf.identity(gather_output2, name="LSTMOutput")
        output=tf.nn.l2_normalize(output)
        return output

    def meanField(self, input_x, input_adj, name):

        W1_tiled = tf.tile(tf.expand_dims(self.W1, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W1_tiled")
        W2_tiled = tf.tile(tf.expand_dims(self.W2, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W2_tiled")

        CONV_PARAMS_tiled = []
        for lv in range(self.max_lv):
            CONV_PARAMS_tiled.append(tf.tile(tf.expand_dims(self.CONV_PARAMS[lv], 0), [tf.shape(input_x)[0], 1, 1],
                                             name=name + "_CONV_PARAMS_tiled_" + str(lv)))

        w1xv = tf.matmul(input_x, W1_tiled, name=name + "_w1xv")
        l = tf.matmul(input_adj, w1xv, name=name + '_l_iteration' + str(1))
        out = w1xv
        for i in range(self.T_iterations - 1):
            ol = l
            lv = self.max_lv - 1
            while lv >= 0:
                with tf.name_scope('cell_' + str(lv)) as scope:
                    node_linear = tf.matmul(ol, CONV_PARAMS_tiled[lv], name=name + '_conv_params_' + str(lv))
                    if lv > 0:
                        ol = tf.nn.relu(node_linear, name=name + '_relu_' + str(lv))
                    else:
                        ol = node_linear
                lv -= 1

            out = tf.nn.tanh(w1xv + ol, name=name + "_mu_iteration" + str(i + 2))
            l = tf.matmul(input_adj, out, name=name + '_l_iteration' + str(i + 2))

        fi = tf.expand_dims(tf.reduce_sum(out, axis=1, name=name + "_y_potential_reduce_sum"), axis=1,
                            name=name + "_y_potential_expand_dims")

        graph_embedding = tf.matmul(fi, W2_tiled, name=name + '_graph_embedding')
        return graph_embedding

    def generateGraphClassificationNetwork(self):
        print("Features size:"+str(self.features_size))

        self.instruction_embeddings_t = tf.Variable(initial_value=tf.constant(self.embedding_matrix),
                                                    trainable=self.trainable_embeddings,
                                                    name="instruction_embedding", dtype=tf.float32)

        self.x_1 = tf.placeholder(tf.int32, [None, None, None], name="x_1")
        self.adj_1 = tf.placeholder(tf.float32, [None, None, None], name="adj_1")
        self.lenghts_1 = tf.placeholder(tf.int32, [None,None], name='lenghts_1')
        self.x_2 = tf.placeholder(tf.int32, [None, None, None], name="x_2")
        self.adj_2 = tf.placeholder(tf.float32, [None, None, None], name="adj_2")
        self.lenghts_2 = tf.placeholder(tf.int32, [None,None], name='lenghts_2')
        self.y = tf.placeholder(tf.float32, [None], name='y_')

        self.norms = []

        l2_loss = tf.constant(0.0)

        # 1. parameters for MeanField
        with tf.name_scope('parameters_MeanField'):

            self.W1 = tf.Variable(tf.truncated_normal([self.features_size, self.embedding_size], stddev=0.1), name="W1")
            self.norms.append(tf.norm(self.W1))

            self.CONV_PARAMS = []
            for lv in range(self.max_lv):
                v = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                name="CONV_PARAMS_" + str(lv))
                self.CONV_PARAMS.append(v)
                self.norms.append(tf.norm(v))

            self.W2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                  name="W2")
            self.norms.append(tf.norm(self.W2))

        with tf.name_scope('LSTMExtraction1'):
            with tf.variable_scope('lstm1'):
                self.x_1_after_lstm = self.lstmFeatures(self.x_1, self.lenghts_1)
        with tf.name_scope('LSTMExtraction2'):
            with tf.variable_scope('lstm2'):
                self.x2_after_lstm = self.lstmFeatures(self.x_2, self.lenghts_2)

        with tf.name_scope('MeanField1'):
            self.graph_embedding_1 = tf.nn.l2_normalize(
                tf.squeeze(self.meanField(self.x_1_after_lstm, self.adj_1, "MeanField1"), axis=1), axis=1,
                name="embedding1")

        with tf.name_scope('MeanField2'):
            self.graph_embedding_2 = tf.nn.l2_normalize(
                tf.squeeze(self.meanField(self.x2_after_lstm, self.adj_2, "MeanField2"), axis=1), axis=1,
                name="embedding2")

        with tf.name_scope('Siamese'):
            self.cos_similarity = tf.reduce_sum(tf.multiply(self.graph_embedding_1, self.graph_embedding_2), axis=1,
                                                name="cosSimilarity")

        # Regularization
        with tf.name_scope("Regularization"):
            l2_loss += tf.nn.l2_loss(self.W1)
            for lv in range(self.max_lv):
                l2_loss += tf.nn.l2_loss(self.CONV_PARAMS[lv])
            l2_loss += tf.nn.l2_loss(self.W2)

        # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):

            self.loss = tf.reduce_sum(tf.squared_difference(self.cos_similarity, self.y), name="loss")
            self.regularized_loss = self.loss + self.l2_reg_lambda * l2_loss

        # Train step
        with tf.name_scope("Train_Step"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.regularized_loss)
