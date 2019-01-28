import numpy as np
import tensorflow as tf

# structure2vec
# DE-MF : discriminative embedding using Mean Field
# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

class Network:

    def __init__(self,
        features_size,
        embedding_size,
        max_lv,
        T_iterations,
        learning_rate,
        l2_reg_lambda
    ):
        self.features_size = features_size
        self.embedding_size = embedding_size
        self.max_lv = max_lv
        self.T_iterations = T_iterations
        self.learning_rate=learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.generateGraphClassificationNetwork()

    def meanField(self, input_x, input_adj, name):

        W1_tiled = tf.tile(tf.expand_dims(self.W1,0), [tf.shape(input_x)[0],1,1], name=name + "_W1_tiled")
        W2_tiled = tf.tile(tf.expand_dims(self.W2,0), [tf.shape(input_x)[0],1,1], name=name + "_W2_tiled")

        CONV_PARAMS_tiled = []
        for lv in range(self.max_lv):
            CONV_PARAMS_tiled.append(tf.tile(tf.expand_dims(self.CONV_PARAMS[lv],0), [tf.shape(input_x)[0],1,1], name=name + "_CONV_PARAMS_tiled_" + str(lv)))

        w1xv = tf.matmul(input_x, W1_tiled, name=name + "_w1xv")
        l = tf.matmul(input_adj, w1xv, name=name + '_l_iteration' + str(1))
        out=w1xv
        for i in range(self.T_iterations-1):
            ol = l
            lv = self.max_lv -1
            while lv >= 0 :
                with tf.name_scope('cell_' + str(lv)) as scope:
                    node_linear = tf.matmul(ol, CONV_PARAMS_tiled[lv], name=name + '_conv_params_' + str(lv))
                    if lv > 0:
                        ol = tf.nn.relu(node_linear, name=name + '_relu_' + str(lv))
                    else:
                        ol = node_linear
                lv -= 1
            out = tf.nn.tanh(w1xv + ol, name=name + "_mu_iteration" + str(i + 2))
            l = tf.matmul(input_adj, out, name=name + '_l_iteration' + str(i + 2))

        fi = tf.expand_dims(tf.reduce_sum(out, axis=1, name=name + "_y_potential_reduce_sum"), axis=1, name=name + "_y_potential_expand_dims")
        
        graph_embedding = tf.matmul(fi, W2_tiled, name=name + '_graph_embedding')
        return graph_embedding

        
    def generateGraphClassificationNetwork(self):

        # Placeholders for input, output
        self.x_1 = tf.placeholder(tf.float32,[None, None,self.features_size], name = "x_1") # Vettore del nodo in input 1
        self.adj_1 = tf.placeholder(tf.float32,[None, None, None],name="adj_1") # Matrice di adiacenza 1
        self.x_2 = tf.placeholder(tf.float32,[None, None,self.features_size], name = "x_2") # Vettore del nodo in input 2
        self.adj_2 = tf.placeholder(tf.float32,[None, None, None],name="adj_2") # Matrice di adiacenza 2
        self.y = tf.placeholder(tf.float32, [None], name='y_')

        self.lenghts_1 = tf.placeholder(tf.float32, [None], name="len1")
        self.lenghts_2 = tf.placeholder(tf.float32, [None], name="len2")

        self.norms = []

        l2_loss = tf.constant(0.0)

        # -------------------------------
        #   1. MEAN FIELD COMPONENT
        # -------------------------------

        #1. parameters for MeanField
        with tf.name_scope('parameters_MeanField'):

            # W1 is a [d,p] matrix, and p is the embedding size as explained above
            self.W1 = tf.Variable(tf.truncated_normal([self.features_size,self.embedding_size], stddev=0.1), name="W1")
            self.norms.append(tf.norm(self.W1))

            # CONV_PARAMSi (i=1,...,n) is a [p,p] matrix. We refer to n as the embedding depth (self.max_lv)
            self.CONV_PARAMS = []
            for lv in range(self.max_lv):
                v = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1), name="CONV_PARAMS_"+str(lv))
                self.CONV_PARAMS.append(v)
                self.norms.append(tf.norm(v))

            # W2 is another [p,p] matrix to transform the embedding vector
            self.W2 =  tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1), name="W2")
            self.norms.append(tf.norm(self.W2))
        
        # Mean Field
        with tf.name_scope('MeanField1'):
            self.graph_embedding_1 = tf.nn.l2_normalize(tf.squeeze(self.meanField(self.x_1,self.adj_1,"MeanField1"), axis=1), axis=1,name="embedding1") # tf.nn.l2_normalize(tf.squeeze(self.meanField(self.x_1,self.adj_1), axis=1), 1)

        with tf.name_scope('MeanField2'):
            self.graph_embedding_2 = tf.nn.l2_normalize(tf.squeeze(self.meanField(self.x_2,self.adj_2,"MeanField2"), axis=1), axis=1,name="embedding2") # tf.nn.l2_normalize(tf.squeeze(self.meanField(self.x_2,self.adj_2), axis=1), 1)

        with tf.name_scope('Siamese'):
             self.cos_similarity = tf.reduce_sum(tf.multiply(self.graph_embedding_1, self.graph_embedding_2), axis=1,name="cosSimilarity")
        
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
