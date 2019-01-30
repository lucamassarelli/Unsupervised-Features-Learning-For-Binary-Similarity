# Investigating Graph Embedding Neural Networks with Unsupervised Features Extraction for Binary Analysis
This repository contains the code to reproduce the experiment of the paper accepted at the Workshop on Binary 
Analysis Research (BAR) 2019.

## Tasks

You can use the code to solve two different tasks:

- Binary Similarity with functions embeddings.
- Compiler Classification

## Downloading the dataset

## Create your own dataset

Following this steps you will be able to create your own dataset!

- Install radare2 on your system.

- Put the executable you want to add to your dataset inside a directory three as follow:

'''
dataset_root/
             \
              \--project/
                         \--compiler
                                    \--optimization
                                                   \executables
'''                                               

For example you will ends up with a three like:

'''
my_dataset/
           \
            \--openSSL/
                      \--gcc-3
                              \--O1
                                   \executables
                              \--O0
                                   \executables
            \--binutil/
                      \--gcc-3
                              \--O1
                                   \executables
                      \--gcc-5      
                              \--O1
                                   \executables
'''
                          
- Once you have your executable in the correct path just launch:

'''
python dataset_creation/ExperimentUtil.py -db name_of_the_db -b --dir dataset_root [-s (if you want to use debug symbols)]
'''

- To split your dataset in train validation and test you can use the following command:

'''
python dataset_creation/ExperimentUtil.py -db name_of_the_db -s
'''

## Reproducing the experiment

### Binary Similarity



## Citation
If you include this code inside your project please cite:

Massarelli L., Di Luna G. A., Petroni F., Querzon L., Baldoni R. 
Investigating Graph Embedding Neural Networks with Unsupervised Features Extraction for Binary Analysis. 
To Appear in: In Symposium on Network and Distributed System Security (NDSS), Workshop on Binary Analysis Research. 2019.

 