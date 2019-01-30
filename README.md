# Investigating Graph Embedding Neural Networks with Unsupervised Features Extraction for Binary Analysis
This repository contains the code to reproduce the experiment of the paper accepted at the Workshop on Binary 
Analysis Research (BAR) 2019.

## Tasks

You can use the code to solve two different tasks:

- Binary Similarity with functions embeddings.
- Compiler Classification

## Reproducing the experiment

Following this step you will be able to reproduce the experiments of the paper!

### Downloading the dataset
First you need to download the datasets.
We release the three datasets used in the paper:

- **OpenSSL_dataset**: It includes two version of OpenSSL libraries compiled for X86 and ARM with 
    gcc with optimizations from 0 to 3. It has been used for binary similarity task.
    To download it:
    
    ```
    ```

   
- **Restricted_Compiler_Dataset**: It includes different projects compiled for X86 gcc-3, gcc-5, 
    clang-3.9 with optimizations from 0 to 3. It has been used for compiler classification. To download it:
    
    ```
    ```
    
- **Compiler_Dataset**: It includes different projects compiled for X86 different compilers (see the paper) with 
optimizations from 0 to 3. It has been used for compiler classification. This dataset is very huge,
you need 30 GB of space to download it. To download it:

    ```
    ```


### Downloading word2vec model for asm

Before to run the experiment you need to download also the word2vec model for asm.
It consists of two file, the embedding matrix and the word2id file. The latter that assigns to 
each instruction an id. The id correspond to the relative row of the instruction inside the
embedding matrix.


### Binary Similarity

To train the network for binary similarity task go into binary similarity folder and look at the file
train.sh.

Here you can change different parameter, like network architecture, path for saving the trained model, 
the databases you want to use for the training, and the embedding matrix for asm instructions.
By default the script uses the data downloaded before.

If you want to change the hyperparameter of the network take a look at parameters.py file!

To start the training just run:

```
export PYTHONPATH=path-to-repository
cd binary_similarity
chmod +x train.sh
./train.sh
```

### Compiler classification

Like in the previous case just run:

```
export PYTHONPATH=path-to-repository
cd compiler_classification
chmod +x train.sh
./train.sh
```

## Creating your own dataset

Following this steps you will be able to create your own dataset!

- Install radare2 on your system.

- Put the executable you want to add to your dataset inside a directory three as follow:

```
dataset_root/
             \
              \--project/
                         \--compiler
                                    \--optimization
                                                   \executables
```                                              

For example you will ends up with a three like:

```
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
```
                          
- Once you have your executable in the correct path just launch:

```
python dataset_creation/ExperimentUtil.py -db name_of_the_db -b --dir dataset_root [-s (if you want to use debug symbols)]
```

- To split your dataset in train validation and test you can use the following command:

```
python dataset_creation/ExperimentUtil.py -db name_of_the_db -s
```




## Citation
If you use this repository or datasets for your project please cite:

Massarelli L., Di Luna G. A., Petroni F., Querzon L., Baldoni R. 
Investigating Graph Embedding Neural Networks with Unsupervised Features Extraction for Binary Analysis. 
To Appear in: In Symposium on Network and Distributed System Security (NDSS), Workshop on Binary Analysis Research. 2019.

 