#!/bin/sh

# Type of the network to use

NETWORK_TYPE="Attention_Mean"
#NETWORK_TYPE="Arith_Mean"
#NETWORK_TYPE="RNN"
#NETWORK_TYPE="Annotations"

# Root path for binary similarity task
BASE_PATH="binary_similarity/"

# Root path for the experiment
EXPERIMENT_PATH=$BASE_PATH/experiments/

# Path for the model
MODEL_PATH=$EXPERIMENT_PATH/out

# Path to the sqlite db with diassembled functions
DB_PATH=../data/openSSL_data.db

# Path to embedding matrix
EMBEDDING_MATRIX=../data/i2v/embedding_matrix.npy

# Path to instruction2id dictionary
INS2ID=../data/i2v/word2id.json

# Add this argument to train.py to use random instructions embeddings
RANDOM_EMBEDDINGS="-r"

# Add this argument to train.py to use trainable instructions embeddings
TRAINABLE_EMBEDDINGS="-te"

python3 train.py --o $MODEL_PATH -n $DB_PATH -nn $NETWORK_TYPE -e $EMBEDDING_MATRIX -j $INS2ID
