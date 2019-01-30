# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import argparse
from dataset_creation import DatabaseFactory, DataSplitter

def debug_msg():
    msg =  " DATABASE UTILITY"
    msg += "-------------------------------------------------\n"
    msg += "This program is an utility to save data into an sqlite database with SAFE \n\n"
    msg += "There are three main command: \n"
    msg += "BUILD:  It create a db with two tables: functions, filtered_functions. \n"
    msg += "        In the first table there are all the functions extracted from the executable with their hex code.\n"
    msg += "        In the second table functions are converted to i2v representation. \n"
    msg += "SPLIT:  Data are splitted into train validation and test set. " \
           "        Then it generate the pairs for the training of the network.\n"
    msg += "EMBEDD: Generate the embeddings of each function in the database using a trained SAFE model\n\n"
    msg += "If you want to train the network use build + split"
    msg += "If you want to create a knowledge base for the binary code search engine use build + embedd"
    msg += "This program has been written by the SAFE team.\n"
    msg += "-------------------------------------------------"
    return msg


def build_configuration(db_name, root_dir, use_symbols):
    msg = "Database creation options: \n"
    msg += " - Database Name: {} \n".format(db_name)
    msg += " - Root dir: {} \n".format(root_dir)
    msg += " - Use symbols: {} \n".format(use_symbols)
    return msg


def split_configuration(db_name, val_split, test_split, epochs):
    msg = "Splitting options: \n"
    msg += " - Database Name: {} \n".format(db_name)
    msg += " - Validation Size: {} \n".format(val_split)
    msg += " - Test Size: {} \n".format(test_split)
    msg += " - Epochs: {} \n".format(epochs)
    return msg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=debug_msg)

    parser.add_argument("-db", "--db", help="Name of the database to create", required=True)

    parser.add_argument("-b", "--build", help="Build db disassebling executables",   action="store_true")
    parser.add_argument("-s", "--split", help="Perform data splitting for training", action="store_true")

    parser.add_argument("-dir", "--dir",     help="Root path of the directory to scan")
    parser.add_argument("-sym", "--symbols", help="Use it if you want to use symbols", action="store_true")

    parser.add_argument("-test", "--test_size", help="Test set size [0-1]",            type=float, default=0.2)
    parser.add_argument("-val",  "--val_size",  help="Validation set size [0-1]",      type=float, default=0.2)
    parser.add_argument("-epo",  "--epochs",    help="# Epochs to generate pairs for", type=int,    default=25)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        print(debug_msg())
        exit(0)

    if args.build:
        print("Disassemblying files and creating dataset")
        print(build_configuration(args.db, args.dir, args.symbols))
        factory = DatabaseFactory.DatabaseFactory(args.db, args.dir)
        factory.build_db(args.symbols)

    if args.split:
        print("Splitting data and generating epoch pairs")
        print(split_configuration(args.db, args.val_size, args.test_size, args.epochs))
        splitter = DataSplitter.DataSplitter(args.db)
        splitter.split_data(args.val_size, args.test_size)
        splitter.create_pairs(args.epochs)

    exit(0)
