#!/bin/sh

# Type of the network to use

NETWORK_TYPE="Attention_Mean"
# NETWORK_TYPE="Arith_Mean"
# NETWORK_TYPE="RNN"
# NETWORK_TYPE="Annotations"

# What to classify:
CLASSIFICATION_KIND="Family"      # Compiler Family
# CLASSIFICATION_KIND="Compiler"      # Compiler Family + Version
# CLASSIFICATION_KIND="Compiler+Opt"   # Compiler Familt + Version + Optimization
# CLASSIFICATION_KIND="Opt"      # Optimization


# Root path for the experiment
MODEL_PATH=experiments/

# Path to the sqlite db with diassembled functions
DB_PATH=../data/restricted_compilers_dataset.db

# Path to embedding matrix
EMBEDDING_MATRIX=../data/i2v/embedding_matrix.npy

# Path to instruction2id dictionary
INS2ID=../data/i2v/word2id.json

# Add this argument to train.py to use random instructions embeddings
RANDOM_EMBEDDINGS="-r"

# Add this argument to train.py to use trainable instructions embeddings
TRAINABLE_EMBEDDINGS="-te"

python3 train.py --o $MODEL_PATH -n $DB_PATH -nn $NETWORK_TYPE -e $EMBEDDING_MATRIX -j $INS2ID -cl $CLASSIFICATION_KIND

