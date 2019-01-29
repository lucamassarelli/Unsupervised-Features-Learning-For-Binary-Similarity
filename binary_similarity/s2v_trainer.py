# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

from binary_similarity.s2v_network_arith_mean import NetworkLSTM as ArithMeanNetwork
from binary_similarity.s2v_network_rnn import NetworkLSTM as RnnNetwork
from binary_similarity.s2v_network_attention_mean import NetworkLSTM as WeightedMeanNetwork
from binary_similarity.s2v_network import Network as AnnotationsNetwork

from binary_similarity.PairFactory import PairFactory

import tensorflow as tf
import random
import sys, os
import numpy as np
from sklearn import metrics
import matplotlib

import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class S2VTrainer:

    def __init__(self, flags, embedding_matrix):
        self.embedding_size = flags.embedding_size
        self.max_lv = flags.max_lv
        self.num_epochs = flags.num_epochs
        self.learning_rate = flags.learning_rate
        self.l2_reg_lambda = flags.l2_reg_lambda
        self.num_checkpoints = flags.num_checkpoints
        self.logdir = flags.logdir
        self.logger = flags.logger
        self.T_iterations = flags.T_iterations
        self.seed = flags.seed
        self.batch_size = flags.batch_size
        self.max_instructions = flags.max_instructions
        self.rnn_depth = flags.rnn_depth
        self.rnn_kind = flags.rnn_kind
        self.max_nodes = flags.MAX_NUM_VERTICES
        self.embeddings_matrix = embedding_matrix
        self.session = None
        self.db_name = flags.db_name
        self.feature_type = flags.feature_type
        self.json_asm2id = flags.json_asm2id
        self.trainable_embeddings = flags.trainable_embeddings
        self.network_type = flags.network_type
        self.cross_val = flags.cross_val

        random.seed(self.seed)
        np.random.seed(self.seed)

        print(self.db_name)

    def loadmodel(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as g:
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )
            sess = tf.Session(config=session_conf)

            # Sets the graph-level random seed.
            tf.set_random_seed(self.seed)

            self.createNetwork()
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
            checkpoint_dir = os.path.abspath(os.path.join(self.logdir, "checkpoints"))
            saver.restore(sess, os.path.join(checkpoint_dir, "model"))
            self.session=sess

        return

    def createNetwork(self):

        if self.network_type == "Annotations":
            self.features_size = 8
        else:
            self.features_size = np.shape(self.embeddings_matrix)[1]

        if self.network_type == "Arith_Mean":

            self.network = ArithMeanNetwork(
                features_size=self.features_size,
                embedding_size=self.embedding_size,
                max_lv=self.max_lv,
                T_iterations=self.T_iterations,
                learning_rate=self.learning_rate,
                l2_reg_lambda=self.l2_reg_lambda,
                batch_size=self.batch_size,
                max_instructions=self.max_instructions,
                max_nodes=self.max_nodes,
                rnn_depth=self.rnn_depth,
                rnn_kind=self.rnn_kind,
                embedding_matrix=self.embeddings_matrix,
                trainable_embeddings=self.trainable_embeddings
            )

        if self.network_type == "RNN":

            self.network = RnnNetwork(
                features_size=self.features_size,
                embedding_size=self.embedding_size,
                max_lv=self.max_lv,
                T_iterations=self.T_iterations,
                learning_rate=self.learning_rate,
                l2_reg_lambda=self.l2_reg_lambda,
                batch_size=self.batch_size,
                max_instructions = self.max_instructions,
                max_nodes = self.max_nodes,
                rnn_depth = self.rnn_depth,
                rnn_kind=self.rnn_kind,
                embedding_matrix=self.embeddings_matrix,
                trainable_embeddings=self.trainable_embeddings
            )

        if self.network_type == "Attention_Mean":

            self.network = WeightedMeanNetwork(
                features_size=self.features_size,
                embedding_size=self.embedding_size,
                max_lv=self.max_lv,
                T_iterations=self.T_iterations,
                learning_rate=self.learning_rate,
                l2_reg_lambda=self.l2_reg_lambda,
                batch_size=self.batch_size,
                max_instructions = self.max_instructions,
                max_nodes = self.max_nodes,
                rnn_depth = self.rnn_depth,
                rnn_kind=self.rnn_kind,
                embedding_matrix=self.embeddings_matrix,
                trainable_embeddings=self.trainable_embeddings
            )

        if self.network_type == "Annotations":

            self.network = AnnotationsNetwork(
                features_size=self.features_size,
                embedding_size=self.embedding_size,
                max_lv=self.max_lv,
                T_iterations=self.T_iterations,
                learning_rate=self.learning_rate,
                l2_reg_lambda=self.l2_reg_lambda,
            )

    def validate_loaded_model(self):

        print("Validating model")

        f = open(self.embedder_file, 'rb')
        self.embedder = pickle.load(f, encoding='latin1')
        f.close()

        sess = self.session

        p_validation = PairFactory(self.db_name, self.feature_type, 'validation_pairs', self.embedder,
                                                self.max_instructions, self.max_nodes,self.functions)


        val_loss = 0
        val_y = []
        val_pred = []

        for adj1_batch, nodes1_batch, adj2_batch, nodes2_batch, y_batch, len1_batch, len2_batch in p_validation.async_chunker(0, self.batch_size):
            feed_dict = {
                self.network.x_1: nodes1_batch,
                self.network.adj_1: adj1_batch,
                self.network.lenghts_1: len1_batch,
                self.network.x_2: nodes2_batch,
                self.network.adj_2: adj2_batch,
                self.network.lenghts_2: len2_batch,
                self.network.y: y_batch,
            }

            loss, similarities = sess.run(
                [self.network.loss, self.network.cos_similarity], feed_dict=feed_dict)
            val_loss += loss * p_validation.batch_dim
            val_y.extend(y_batch)
            val_pred.extend(similarities.tolist())
        val_loss /= p_validation.num_pairs
        val_fpr, val_tpr, val_thresholds = metrics.roc_curve(val_y, val_pred, pos_label=1)
        val_auc = metrics.auc(val_fpr, val_tpr)

        sys.stdout.write("\r\t val_auc {:g}".format(val_auc))
        sys.stdout.flush()


    def test_loaded_model(self):

        print("Validating model")

        f = open(self.embedder_file, 'rb')
        self.embedder = pickle.load(f, encoding='latin1')
        f.close()

        sess = self.session

        if self.feature_type == "lstm_cfg":
            p_test = PairFactory(self.db_name, self.feature_type, 'test_pairs', self.embedder,
                                                self.max_instructions, self.max_nodes,self.functions)

        val_loss = 0
        val_y = []
        val_pred = []

        for adj1_batch, nodes1_batch, adj2_batch, nodes2_batch, y_batch, len1_batch, len2_batch in p_test.async_chunker(0, self.batch_size):
            feed_dict = {
                self.network.x_1: nodes1_batch,
                self.network.adj_1: adj1_batch,
                self.network.lenghts_1: len1_batch,
                self.network.x_2: nodes2_batch,
                self.network.adj_2: adj2_batch,
                self.network.lenghts_2: len2_batch,
                self.network.y: y_batch,
            }

            loss, similarities = sess.run(
                [self.network.loss, self.network.cos_similarity], feed_dict=feed_dict)
            val_loss += loss * p_test.batch_dim
            val_y.extend(y_batch)
            val_pred.extend(similarities.tolist())
        val_loss /= p_test.num_pairs
        val_fpr, val_tpr, val_thresholds = metrics.roc_curve(val_y, val_pred, pos_label=1)
        val_auc = metrics.auc(val_fpr, val_tpr)

        sys.stdout.write("\r\t val_auc {:g}".format(val_auc))
        sys.stdout.flush()

    def read_weight(self):
        a = self.session.run(self.session.graph.get_tensor_by_name('LSTMExtraction1/lstm1/W0:0'))
        plt.bar(range(0, 150), a[0])
        plt.show()
        plt.savefig('/home/massarelli/weight.pdf')


    def train(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as g:
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )
            sess = tf.Session(config=session_conf)

            # Sets the graph-level random seed.
            tf.set_random_seed(self.seed)

            self.createNetwork()

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # TensorBoard
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", self.network.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(self.logdir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            val_summary_op = tf.summary.merge([loss_summary])
            val_summary_dir = os.path.join(self.logdir, "summaries", "validation")
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary])
            test_summary_dir = os.path.join(self.logdir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(self.logdir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

            BEST_VAL_AUC = 0
            stat_file = open(str(self.logdir) + "/epoch_stats.tsv", "w")
            stat_file.write("#epoch\ttrain_loss\tval_loss\tval_auc\ttest_loss\ttest_auc\n")

            p_train = PairFactory(self.db_name, self.feature_type, 'train_pairs', self.json_asm2id,
                                  self.max_instructions, self.max_nodes)

            p_validation = PairFactory(self.db_name, self.feature_type, 'validation_pairs', self.json_asm2id,
                                  self.max_instructions, self.max_nodes)

            p_test = PairFactory(self.db_name, self.feature_type, 'test_pairs', self.json_asm2id,
                                  self.max_instructions, self.max_nodes)

            step = 0
            for epoch in range(0, self.num_epochs):
                epoch_msg = ""
                epoch_msg += "  epoch: {}\n".format(epoch)

                epoch_loss = 0

                # ----------------------#
                #         TRAIN	       #
                # ----------------------#
                n_batch=0
                for adj1_batch, nodes1_batch, adj2_batch, nodes2_batch, y_batch, len1_batch, len2_batch in p_train.async_chunker(epoch%25, self.batch_size, shuffle=True):

                    feed_dict = {
                        self.network.x_1: nodes1_batch,
                        self.network.adj_1: adj1_batch,
                        self.network.lenghts_1: len1_batch,
                        self.network.x_2: nodes2_batch,
                        self.network.adj_2: adj2_batch,
                        self.network.lenghts_2: len2_batch,
                        self.network.y: y_batch,
                    }
                    summaries, _, loss, norms, ge1, ge2, cs = sess.run(
                        [train_summary_op, self.network.train_step, self.network.loss, self.network.norms,
                         self.network.graph_embedding_1, self.network.graph_embedding_2, self.network.cos_similarity],
                        feed_dict=feed_dict)

                    sys.stdout.write(
                        "\r\t Pairs in current batch {}, training batch {} / {} -- {} ".format(len(len1_batch),n_batch, p_train.num_batches, float(n_batch)/p_train.num_batches))
                    sys.stdout.flush()
                    n_batch = n_batch+1

                    # tensorboard
                    train_summary_writer.add_summary(summaries, step)
                    epoch_loss += loss * p_train.batch_dim  # ???
                    step += 1

                # recap epoch
                epoch_loss /= p_train.num_pairs
                epoch_msg += "\ttrain_loss: {}\n".format(epoch_loss)

                # ----------------------#
                #      VALIDATION	   #
                # ----------------------#
                val_loss = 0
                epoch_msg += "\n"
                val_y = []
                val_pred = []
                for adj1_batch, nodes1_batch,  adj2_batch, nodes2_batch, y_batch, len1_batch, len2_batch in p_validation.async_chunker(0, self.batch_size*2):

                    feed_dict = {
                        self.network.x_1: nodes1_batch,
                        self.network.adj_1: adj1_batch,
                        self.network.lenghts_1: len1_batch,
                        self.network.x_2: nodes2_batch,
                        self.network.adj_2: adj2_batch,
                        self.network.lenghts_2: len2_batch,
                        self.network.y: y_batch,
                    }

                    summaries, loss, similarities = sess.run(
                        [val_summary_op, self.network.loss, self.network.cos_similarity], feed_dict=feed_dict)
                    val_loss += loss * p_validation.batch_dim
                    val_summary_writer.add_summary(summaries, step)
                    val_y.extend(y_batch)
                    val_pred.extend(similarities.tolist())
                val_loss /= p_validation.num_pairs
                val_pred=np.nan_to_num(val_pred)
                val_fpr, val_tpr, val_thresholds = metrics.roc_curve(val_y, val_pred, pos_label=1)
                val_auc = metrics.auc(val_fpr, val_tpr)
                epoch_msg += "\tval_loss : {}\n\tval_auc : {}\n".format(val_loss, val_auc)

                sys.stdout.write(
                    "\r\tepoch {} / {}, loss {:g}, val_auc {:g}, norms {}".format(epoch, self.num_epochs, epoch_loss,
                                                                                  val_auc, norms))
                sys.stdout.flush()

                # execute test only if validation auc increased
                test_loss = "-"
                test_auc = "-"

                if val_auc > BEST_VAL_AUC and self.cross_val:
                    BEST_VAL_AUC =  val_auc
                    saver.save(sess, checkpoint_prefix)
                    print("\nNEW BEST_VAL_AUC: {} !\n".format(BEST_VAL_AUC))
                    # write ROC raw data
                    with open(str(self.logdir) + "/best_val_roc.tsv", "w") as the_file:
                        the_file.write("#thresholds\ttpr\tfpr\n")
                        for t, tpr, fpr in zip(val_thresholds, val_tpr, val_fpr):
                            the_file.write("{}\t{}\t{}\n".format(t, tpr, fpr))

                if val_auc > BEST_VAL_AUC and not self.cross_val:
                    BEST_VAL_AUC = val_auc

                    epoch_msg += "\tNEW BEST_VAL_AUC: {} !\n".format(BEST_VAL_AUC)

                    # save best model
                    saver.save(sess, checkpoint_prefix)

                    # ----------------------#
                    #         TEST  	    #
                    # ----------------------#

                    # TEST
                    test_loss = 0
                    epoch_msg += "\n"
                    test_y = []
                    test_pred = []
                    for adj1_batch, nodes1_batch, adj2_batch, nodes2_batch, y_batch, len1_batch, len2_batch in \
                            p_test.async_chunker(0, self.batch_size*2):

                        feed_dict = {
                            self.network.x_1: nodes1_batch,
                            self.network.adj_1: adj1_batch,
                            self.network.lenghts_1: len1_batch,
                            self.network.x_2: nodes2_batch,
                            self.network.adj_2: adj2_batch,
                            self.network.lenghts_2: len2_batch,
                            self.network.y: y_batch,
                        }

                        summaries, loss, similarities = sess.run(
                            [test_summary_op, self.network.loss, self.network.cos_similarity], feed_dict=feed_dict)
                        test_loss += loss * p_test.batch_dim
                        test_summary_writer.add_summary(summaries, step)
                        test_y.extend(y_batch)
                        test_pred.extend(similarities.tolist())
                    test_loss /= p_test.num_pairs
                    test_pred = np.nan_to_num(test_pred)
                    test_fpr, test_tpr, test_thresholds = metrics.roc_curve(test_y, test_pred, pos_label=1)

                    # write ROC raw data
                    with open(str(self.logdir) + "/best_test_roc.tsv", "w") as the_file:
                        the_file.write("#thresholds\ttpr\tfpr\n")
                        for t, tpr, fpr in zip(test_thresholds, test_tpr, test_fpr):
                            the_file.write("{}\t{}\t{}\n".format(t, tpr, fpr))

                    test_auc = metrics.auc(test_fpr, test_tpr)
                    epoch_msg += "\ttest_loss : {}\n\ttest_auc : {}\n".format(test_loss, test_auc)
                    fig = plt.figure()
                    plt.title('Receiver Operating Characteristic')
                    plt.plot(test_fpr, test_tpr, 'b', label='AUC = %0.2f' % test_auc)
                    fig.savefig(str(self.logdir) + "/best_test_roc.png")
                    print(
                        "\nNEW BEST_VAL_AUC: {} !\n\ttest_loss : {}\n\ttest_auc : {}\n".format(BEST_VAL_AUC, test_loss,
                                                                                               test_auc))
                    plt.close(fig)

                stat_file.write(
                    "{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch, epoch_loss, val_loss, val_auc, test_loss, test_auc))
                self.logger.info("\n{}\n".format(epoch_msg))
            stat_file.close()
            sess.close()
            return BEST_VAL_AUC