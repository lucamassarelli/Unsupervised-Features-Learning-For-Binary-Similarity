# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#


from compiler_provenance.s2v_classification_network_arith_mean import NetworkLSTM as arithMeanNetwork
from compiler_provenance.s2v_classification_network_rnn import NetworkLSTM as rrnFastMeanNetwork
from compiler_provenance.s2v_classification_network_annotations import Network as annotationNetwork
from compiler_provenance.s2v_classification_network_attention_mean import Network as weightedMeanNetwork

from compiler_provenance.FunctionFactory import PairFactory as FunctionFactory

import tensorflow as tf
import random
import sys, os
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib
import sqlite3
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import tqdm

class S2VTrainerLSTM:

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
        self.dense_layer_size = flags.dense_layer_size
        self.flags = flags
        self.functions = False

        if flags.class_kind == "CMP" or flags.class_kind=="FML":
            query_str="SELECT DISTINCT  compiler FROM functions"
        elif flags.class_kind == "CMPOPT":
            query_str = "SELECT DISTINCT  compiler,optimization FROM functions"
        elif flags.class_kind == "OPT":
            query_str = "SELECT DISTINCT  optimization FROM functions"

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        print("Looking in db for classes")
        q = cur.execute(query_str)
        q_compilers = q.fetchall()
        #q_compilers = [c[0] for c in compilers]
        compilers = []

        for c in q_compilers:
            if flags.class_kind == "CMPOPT":
                 compiler = c[0] + '-' + c[1]
            elif flags.class_kind == "FML":
                compiler = str(c[0]).split('-')[0]

            else:
                 compiler = c[0]
            compilers.append(compiler)

        print(compilers)


        compilers = list(set(compilers))
        conn.close()

        self.encoder = LabelEncoder()
        self.encoder.fit(compilers)
        self.num_classes = len(self.encoder.classes_)

        print("Num classes = " + str(self.num_classes))

        random.seed(self.seed)
        np.random.seed(self.seed)

        print(self.db_name)

    def plot_confusion_matrix(self, cm, classes, normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

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
            self.session = sess
        return


    def createNetwork(self):
        self.features_size = np.shape(self.embeddings_matrix)[1]
        if self.network_type == "Arith_Mean":

            self.network = arithMeanNetwork(
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
                trainable_embeddings=self.trainable_embeddings,
                num_classes=self.num_classes,
                dense_layer_size=self.dense_layer_size
            )

        if self.network_type == "RNN":

            self.network = rrnFastMeanNetwork(
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
                trainable_embeddings=self.trainable_embeddings,
                dense_layer_size=self.dense_layer_size,
                num_classes=self.num_classes
            )

        if self.network_type == "Attention_Mean":

            self.network = weightedMeanNetwork(
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
                trainable_embeddings=self.trainable_embeddings,
                dense_layer_size=self.dense_layer_size,
                num_classes = self.num_classes
            )

        if self.network_type == "Annotations":
            self.features_size = 8
            self.network = annotationNetwork(
                features_size=self.features_size,
                embedding_size=self.embedding_size,
                max_lv=self.max_lv,
                T_iterations=self.T_iterations,
                learning_rate=self.learning_rate,
                l2_reg_lambda=self.l2_reg_lambda,
                dense_layer_size=self.dense_layer_size,
                num_classes=self.num_classes
            )

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

            print("Network created")

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

            BEST_ACCURACY = 0
            stat_file = open(str(self.logdir) + "/epoch_stats.tsv", "w")
            stat_file.write("#epoch\ttrain_loss\tval_loss\tval_auc\ttest_loss\ttest_auc\n")

            print("Creating functions factories...")
            sys.stdout.flush()

            p_train = FunctionFactory(self.db_name, self.feature_type, 'train', self.json_asm2id, self.max_instructions, self.max_nodes, self.encoder,self.batch_size,self.flags)
            p_validation = FunctionFactory(self.db_name, self.feature_type, 'validation', self.json_asm2id, self.max_instructions, self.max_nodes, self.encoder,self.batch_size,self.flags)
            p_test = FunctionFactory(self.db_name, self.feature_type, 'test', self.json_asm2id,self.max_instructions, self.max_nodes, self.encoder,self.batch_size,self.flags)

            print("Starting train!")
            sys.stdout.flush()

            step = 0
            for epoch in range(0, self.num_epochs):
                epoch_msg = ""
                epoch_msg += "  epoch: {}\n".format(epoch)

                epoch_loss =  0

                # ----------------------#
                #         TRAIN	       #
                # ----------------------#
                n_batch=0
                for adj_batch, nodes_batch, y_batch, len_batch in tqdm.tqdm(p_train.async_chunker(epoch%25, self.batch_size, shuffle=True), total=p_train.num_batches):

                    assert len(adj_batch)

                    feed_dict = {
                        self.network.x: nodes_batch,
                        self.network.adj: adj_batch,
                        self.network.lenghts: len_batch,
                        self.network.y: y_batch,
                    }

                    summaries, _, loss, norms = sess.run(
                        [train_summary_op, self.network.train_step, self.network.loss, self.network.norms],
                        feed_dict=feed_dict)

                    n_batch=n_batch+1

                    # tensorboard
                    train_summary_writer.add_summary(summaries, step)
                    epoch_loss += loss * p_train.batch_dim  # ???
                    step += 1

                # recap epoch
                epoch_loss /= p_train.num_pairs

                # ----------------------#
                #      VALIDATION	   #
                # ----------------------#
                val_loss = 0
                epoch_msg += "\n"
                val_y = []
                val_pred = []
                print("Validating")
                for adj_batch, nodes_batch, y_batch, len_batch in tqdm.tqdm(p_validation.async_chunker(0, self.batch_size),total=p_validation.num_batches):
                    feed_dict = {
                        self.network.x: nodes_batch,
                        self.network.adj: adj_batch,
                        self.network.lenghts: len_batch,
                        self.network.y: y_batch,
                    }

                    summaries, loss, pred_probab, pred_classes = sess.run(
                        [val_summary_op, self.network.loss, self.network.pred_probab, self.network.pred_classes], feed_dict=feed_dict)
                    val_loss += loss * p_validation.batch_dim
                    val_summary_writer.add_summary(summaries, step)
                    val_y.extend(y_batch)
                    val_pred.extend(pred_classes)
                val_loss /= p_validation.num_pairs

                val_accuracy = metrics.accuracy_score(val_y, val_pred)

                val_report = metrics.classification_report(val_y, val_pred, target_names=self.encoder.classes_)

                tmp = val_report.split("\n")
                val_report = ""
                for l in tmp:
                    val_report += "\t\t" + l + "\n"

                stri = "\tepoch {} / {}, train loss {:g}, val loss {:g}, val accuracy {:g}\n".format(epoch, self.num_epochs, epoch_loss, val_loss, val_accuracy)
                
                epoch_msg += stri

                sys.stdout.write(stri)

                sys.stdout.flush()

                # execute test only if validation auc increased
                test_loss = "-"
                test_auc = "-"

                if val_accuracy > BEST_ACCURACY and self.cross_val:
                    BEST_ACCURACY =  val_accuracy
                    saver.save(sess, checkpoint_prefix)
                    print("\nNEW BEST_VAL_ACCURACY: {} !\n".format(BEST_ACCURACY))

                if val_accuracy > BEST_ACCURACY and not self.cross_val:
                    BEST_ACCURACY = val_accuracy

                    sys.stdout.write("\t" + "-*"*40 + "\n")

                    stri = "\tNEW BEST_ACCURACY: {} !\n\tVal Classification Report: \n {} \n".format(BEST_ACCURACY, val_report)

                    epoch_msg += stri
                    sys.stdout.write(stri)

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
                    print("Testing")
                    for adj_batch, nodes_batch, y_batch, len_batch in tqdm.tqdm(p_test.async_chunker(0, self.batch_size),total=p_test.num_batches):

                        feed_dict = {
                            self.network.x: nodes_batch,
                            self.network.adj: adj_batch,
                            self.network.lenghts: len_batch,
                            self.network.y: y_batch,
                        }

                        summaries, loss, pred_probab, pred_classes = sess.run(
                            [val_summary_op, self.network.loss, self.network.pred_probab, self.network.pred_classes],
                            feed_dict=feed_dict)
                        test_loss += loss * p_test.batch_dim
                        test_summary_writer.add_summary(summaries, step)
                        test_y.extend(y_batch)
                        test_pred.extend(pred_classes)
                    test_loss /= p_test.num_pairs

                    test_accuracy = metrics.accuracy_score(test_y, test_pred)

                    test_report = metrics.classification_report(test_y, test_pred, target_names=self.encoder.classes_)

                    tmp = test_report.split("\n")
                    test_report = ""
                    for l in tmp:
                        test_report += "\t\t" + l + "\n"

                    # Compute confusion matrix
                    cnf_matrix = metrics.confusion_matrix(test_y,  test_pred)
                    np.set_printoptions(precision=2)
                    np.savetxt(str(self.logdir) + "/best_test_confusion_matrix.csv", cnf_matrix, delimiter=',')

                    fig=plt.figure()
                    self.plot_confusion_matrix(cnf_matrix, self.encoder.classes_)
                    plt.savefig(str(self.logdir) + "/best_test_confusion_matrix.png")
                    plt.close(fig)
                   
                    tmp = str(cnf_matrix).split('\n')
                    scnf = ""
                    for l in tmp:
                        scnf += "\t\t" + l + "\n"

                    stri = "\tTest_loss : {}\n\tTest Accuracy: {}\n\tTest Classification Report:\n {} \tTest Confusion Matrix : \n {} \n".format(test_loss, test_accuracy, test_report, scnf)
                    epoch_msg += stri

                    sys.stdout.write(stri)

                    sys.stdout.write("\t" + "-*"*40 + "\n")

                stat_file.write(
                    "{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch, epoch_loss, val_loss, val_accuracy, test_loss, test_accuracy))
                self.logger.info("\n *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n {} \n".format(epoch_msg))
            stat_file.close()
            sess.close()
            return BEST_ACCURACY
