# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import argparse
import time
import sys, os
import logging

def getLogger(logfile):
    logger = logging.getLogger(__name__)
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger, hdlr

class Flags:

    def __init__(self):
        parser = argparse.ArgumentParser(description=' cryptoarb.')

        parser.add_argument("-o", "--output", dest="output_file", help="output directory for logging and models", required=False)
        parser.add_argument("-e", "--embedder", dest="embedder_file", help="file with the embedder for the instructions",required=False)
        parser.add_argument("-n", "--dbName", dest="db_name", help="Name of the database", required=False)
        parser.add_argument("-ld","--load_dir", dest="load_dir", help="Load the model from directory load_dir", required=False)
        parser.add_argument("-nn","--network_type", help="network type: Arith_Mean, Weighted_Mean, RNN, CCS", required=True, dest="network_type")
        parser.add_argument("-r", "--random", help="if present the network use random embedder", default=False, action="store_true", dest="random_embedding", required=False)
        parser.add_argument("-te","--trainable_embedding", help="if present the network consider the embedding as trainable", action="store_true", dest="trainable_embeddings", default=False)
        parser.add_argument("-cv","--cross_val", help="if present the training is done with cross validiation", default=False, action="store_true", dest="cross_val")
        parser.add_argument("-cl", "--classification_kind", help="classification kind: Compiler, Compiler+Opt, Opt",default="Compiler", required=False, dest="classification_kind")

        args = parser.parse_args()
        self.network_type = args.network_type

        if self.network_type == "Annotations":
            self.feature_type = 'acfg'
        elif self.network_type in ["Arith_Mean", "Weighted_Mean", "RNN","Attention","RNN_SINGLE"]:
            self.feature_type = 'lstm_cfg'
        else:
            print("ERROR NETWORK NOT FOUND")
            exit(0)

        if args.classification_kind == "Family":
            self.class_kind="FML"
        elif args.classification_kind == "Compiler":
            self.class_kind="CMP"
        elif args.classification_kind == "Compiler+Opt":
            self.class_kind="CMPOPT"
        elif args.classification_kind == "Opt":
            self.class_kind = "OPT"
        else:
            print("Classification option unkown")
            exit(0)

        # mode = mean_field
        self.batch_size = 250           # minibatch size (-1 = whole dataset)
        self.num_epochs = 50            # number of epochs
        self.embedding_size = 64        # dimension of latent layers
        self.learning_rate = 0.001      # init learning_rate
        self.max_lv = 2                 # embedd depth
        self.T_iterations= 2            # max rounds of message passing
        self.l2_reg_lambda = 0          # 0.002 #0.002 # regularization coefficient
        self.num_checkpoints = 1        # max number of checkpoints
        self.out_dir = args.output_file # directory for logging
        self.db_name = args.db_name
        self.load_dir=str(args.load_dir)
        self.random_embedding = args.random_embedding
        self.trainable_embeddings = args.trainable_embeddings
        self.cross_val = args.cross_val
        self.cross_val_fold = 5
        self.dense_layer_size = 200

        self.rnn_depth = 1              # depth of the rnn
        self.max_instructions = 50      # number of instructions
        self.rnn_kind = 1               #kind of rnn cell 0: lstm cell 1: GRU cell


        self.seed = 2 # random seed

        # create logdir and logger
        self.reset_logdir()

        self.embedder_file=args.embedder_file

        self.MAX_NUM_VERTICES = 150
        self.MIN_NUM_VERTICES = 1

    def reset_logdir(self):
        # create logdir
        timestamp = str(int(time.time()))
        self.logdir = os.path.abspath(os.path.join(self.out_dir, "runs", timestamp))   
        os.makedirs(self.logdir, exist_ok=True)   

        # create logger
        self.log_file = str(self.logdir)+'/console.log'
        self.logger, self.hdlr = getLogger(self.log_file)

        # create symlink for last_run
        sym_path_logdir = str(self.out_dir)+"/last_run"
        try:
            os.unlink(sym_path_logdir)   
        except:
            pass
        try:            
            os.symlink(self.logdir, sym_path_logdir)
        except:
            print("\nfailed to create symlink!\n")

    def close_log(self):
        self.hdlr.close()
        self.logger.removeHandler(self.hdlr)
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def __str__(self):
        msg = ""
        msg +="\n  Parameters:\n"
        msg +="\tNetwork_Type: {}\n".format(self.network_type)
        msg +="\tRandom embedding: {}\n".format(self.random_embedding)
        msg +="\tTrainable embedding: {}\n".format(self.trainable_embeddings)
        msg +="\tFeature Type: {}\n".format(self.feature_type)
        msg +="\tlogdir: {}\n".format(self.logdir)
        msg +="\tbatch_size: {}\n".format(self.batch_size)
        msg +="\tnum_epochs: {}\n".format(self.num_epochs)
        msg +="\tembedding_size: {}\n".format(self.embedding_size)
        msg +="\tlearning_rate: {}\n".format(self.learning_rate)
        msg +="\tmax_lv: {}\n".format(self.max_lv)
        msg +="\tT_iterations: {}\n".format(self.T_iterations)
        msg +="\tl2_reg_lambda: {}\n".format(self.l2_reg_lambda)
        msg +="\tnum_checkpoints: {}\n".format(self.num_checkpoints)
        msg +="\tseed: {}\n".format(self.seed)
        msg +="\tMAX_NUM_VERTICES: {}\n".format(self.MAX_NUM_VERTICES)
        msg +="\tMax Instructions per cfg node: {}\n".format(self.max_instructions)
        msg +="\tDense Layer Size: {}\n".format(self.dense_layer_size)
        msg += "\tClasses kind: {}\n".format(self.class_kind)
        if self.network_type == "RNN":
            msg += "\tRNN type (0, lstm; 1, GRU): {}\n".format(self.rnn_kind)
            msg += "\tRNN Depth: {}\n".format(self.rnn_depth)
        if self.network_type == "Attention":
            msg += "\tRNN type (0, lstm; 1, GRU): {}\n".format(self.rnn_kind)
            msg += "\tRNN Depth: {}\n".format(self.rnn_depth)
            msg += "\tAttention hops: {}\n".format(self.attention_hops)
            msg += "\tAttention depth: {}\n".format(self.attention_detph)
        return msg
