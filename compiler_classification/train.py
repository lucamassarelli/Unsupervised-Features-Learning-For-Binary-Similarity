# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

from s2v_trainer import S2VTrainerLSTM
from parameters import Flags
import sys
sys.path.append('../')
from embedding_util.embedder import Embedder
import pickle
import numpy as np


def run_test():
    flags = Flags()
    flags.logger.info("\n{}\n".format(flags))

    print(str(flags))

    file_embedder=flags.embedder_file

    file = open(file_embedder,'rb')
    embedder = pickle.load(file, encoding='latin1')
    file.close()

    embedding_matrix = embedder.orderedMatrix()
    if flags.random_embedding:
        embedding_matrix = np.random.rand(*np.shape(embedding_matrix)).astype(np.float32)
        embedding_matrix[0, :] = np.zeros(np.shape(embedding_matrix)[1]).astype(np.float32)

    if flags.cross_val:
        print("STARTING CROSS VALIDATION")
        res = []
        mean = 0
        for i in range(0, flags.cross_val_fold):
            print("CROSS VALIDATION STARTING FOLD: " + str(i))
            if i > 0:
                flags.close_log()
                flags.reset_logdir()
                del flags
                flags = Flags()
                flags.logger.info("\n{}\n".format(flags))

            flags.logger.info("Starting cross validation fold: {}".format(i))

            flags.db_name = flags.db_name + "_val_" + str(i+1) + ".db"
            flags.logger.info("Cross validation db name: {}".format(flags.db_name))

            trainer = S2VTrainerLSTM(flags, embedding_matrix)
            best_val_auc = trainer.train()

            mean += best_val_auc
            res.append(best_val_auc)

            flags.logger.info("Cross validation fold {} finished best auc: {}".format(i, best_val_auc))
            print("FINISH FOLD: " + str(i) + " BEST VAL AUC: " + str(best_val_auc))

        print("CROSS VALIDATION ENDED")
        print("Result: " + str(res))
        print("")

        flags.logger.info("Cross validation finished results: {}".format(res))
        flags.logger.info(" mean: {}".format(mean / flags.cross_val_fold))
        flags.close_log()

        flags.close_log()

    else:
        trainer = S2VTrainerLSTM(flags, embedding_matrix)
        trainer.train()
        flags.close_log()


if __name__ == '__main__':
    run_test()
