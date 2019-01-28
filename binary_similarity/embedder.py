#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle

class Embedder:
    embeddings = {}
    word2id = {}

    def __init__(self, embeddings, word2id):
        self.embeddings=embeddings
        self.word2id=word2id

    #prende una stringa e ritorna l'embedding in un vettore numpy
    def lookup(self,string):
        if not( string in self.word2id):
            string="UNK"
        id = self.word2id[string]
        return self.embeddings[id]

    # fa l'unione con un altro embedders
    # cambia gli id di ogni embedd per uniformarli
    #
    def merge(self,embedder):
        union_e={}
        union_w2i={}

        for i,x in enumerate(self.word2id):
            union_w2i["X_"+str(x)]=i
            union_e[i]=self.embeddings[self.word2id[x]]
        off=len(union_w2i)
        for i,x in enumerate(embedder.word2id):
            union_w2i['A_'+str(x)]=(int(i)+off)
            union_e[(int(i)+off)] = embedder.embeddings[embedder.word2id[x]]
        self.word2id=union_w2i
        self.embeddings=union_e
        return Embedder(union_e,union_w2i)



    #returns an np.matrix of the embeddings ordered by ids
    def orderedMatrix(self):
        print("Genearting ordered Matrix...")
        ## controllare l'ordine
        ##appendo l'embedding fittizzio del padding, un vettore di 0 all'inizio della matrice
        x=[[0]*len( next(iter(self.embeddings.values())))]
        x=x+[self.embeddings[x] for x in sorted(self.word2id.values())]
        return x

    def is_number(s,number):
        try:
            float(number)
            return True
        except ValueError:
            return False

    def getPaddingIndex(self):
        # ritorno l'indice del vettore fittizio di padding, che è la prima riga della matrice restituita da orderedMatrix
        return 0

    #gets an array of words (i.e. elements that have to be embedded) and returns an
    #arrays of corresponding IDs. E.g.: toIds(["mov_rax,rbx","push_rbp"]) -> [1,2]
    # s.t. self.embedding(1) -> embedding vector of "mov_rax,rbx".
    def toIds(self,array):
        ret_array=[]
        for x in array:
            if x in self.word2id:
                ret_array.append(self.word2id[x]+1)
            elif 'X_' in x:
                ret_array.append(self.word2id['X_UNK']+1)
            elif 'A_' in x:
                ret_array.append(self.word2id['A_UNK']+1)
            else:
                if self.is_number(x):
                   print("Padding")
                   #ritorno l'indice del vettore fittizio di padding, che è la prima riga
                   ret_array.append(0)
                else:
                    print("There is a problem "+str(x)+" does not appear to be an asm or arm instruction")
                    ret_array.append(self.word2id['X_UNK']+1)
        return ret_array


    def savePlaintext(self,filname="embedder.emb"):
        with open(filname,'wb') as file:
            for id in self.word2id:
                file.write(str(id) + ":" + str(np.savetxt(self.embeddings[id]))+ "\n")

    def loadPlaintext(self, filename):
            _emedding={}
            _ids={}
            with open(filename, 'rb') as the_file:
                content = the_file.readlines()
                for i,x in enumerate(content):
                    id=x.split(":")[0]
                    em=np.loadtxt(x.split(":")[1])
                    #np cast em to array
                    _ids[i]=id
                    _emedding[i]=em
            return Embedder(_emedding,_ids)

if __name__=='__main__':
    nop=0
    with open('x86_filtered_embedder_new.pkl','rb') as x86:
        x86_e=pickle.load(x86)
        with open('arm_filtered_embedder_new.pkl','rb') as arm:
            arm_e=pickle.load(arm)
            merge=x86_e.merge(arm_e)
            with open('merge_100_embedder.pkl','wb') as m:
                pickle.dump(merge,m)
