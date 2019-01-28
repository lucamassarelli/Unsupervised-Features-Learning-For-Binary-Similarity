#!/bin/sh

# Type of the network to use

NETWORK_TYPE="Attention_Mean"
#NETWORK_TYPE="Arith_Mean"
#NETWORK_TYPE="RNN"
#NETWORK_TYPE="Annotations"

# Root path for binary similarity task
BASE_PATH="../binary_similarity"

# Root path for the experiment
EXPERIMENT_PATH=$BASE_PATH/experiments/

# Path for the model
MODEL_PATH=$EXPERIMENT_PATH/out

# Path to the sqlite db with diassembled functions
DB_PATH=$BASE_PATH/data/openSSL_data.db

# Path to pickle embedder
EMBEDDER=$BASE_PATH/word2vec/embedder.pkl

# Add this argument to train.py to use random instructions embeddings
RANDOM_EMBEDDINGS="-r"

# Add this argument to train.py to use trainable instructions embeddings
TRAINABLE_EMBEDDINGS="-te"

python3 train.py --o $OUT_PATH -n $DB_PATH -nn $NETWORK_TYPE -e $EMBEDDER
