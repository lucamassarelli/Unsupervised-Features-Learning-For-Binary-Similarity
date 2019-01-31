# SAFE TEAM
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import argparse
import os
import sys
from subprocess import call

class Downloader:

    def __init__(self):
        parser = argparse.ArgumentParser(description='SAFE downloader')

        parser.add_argument("-i2v", "--i2v", dest="i2v", help="Download the i2v dictionary and embedding matrix",
                            action="store_true",
                            required=False)

        parser.add_argument("-op", "--openSSL", dest="openSSL",
                            help="Download OpenSSL dataset",
                            action="store_true",
                            required=False)

        parser.add_argument("-rc", "--restricted_compiler", dest="restricted_compiler",
                            help="Download the restricted compiler dataset",
                            action="store_true",
                            required=False)

        parser.add_argument("-c", "--compiler", dest="compiler",
                            help="Download the compiler dataset. Be careful, it is very huge ( 30 GB ).",
                            action="store_true",
                            required=False)

        args = parser.parse_args()

        self.i2v = args.i2v
        self.openSSL = args.openSSL
        self.restricted_compiler = args.restricted_compiler
        self.compiler = args.compiler

        if not (self.i2v, self.openSSL or self.restricted_compiler or self.compiler):
            parser.print_help(sys.__stdout__)

        self.url_i2v = "https://drive.google.com/file/d/1ndKVrot5lBPklGGFn-olEt-rCtzjv69z/view?usp=sharing"
        self.url_openSSL = "https://drive.google.com/file/d/1NnC4qCtZUDdb32Yfeq2toa94jvCKTBxZ/view?usp=sharing"
        self.url_restricted_compiler = "https://drive.google.com/file/d/15VUJ3iwj5VHCqAXiUcr4zJgVWSCbaU_d/view?usp=sharing"
        self.url_compiler = "https://drive.google.com/file/d/1fEr9N97fTsAS2NXYpYI3GRTxadaJwhTe/view?usp=sharing"

        self.base_path = "data"
        self.path_i2v = os.path.join(self.base_path, "")
        self.path_openSSL = os.path.join(self.base_path, "")
        self.path_restricted_compiler = os.path.join(self.base_path, "")
        self.path_compiler = os.path.join(self.base_path, "")

        self.i2v_compress_name='i2v.tar.bz2'
        self.openSSL_compress_name='openSSL_dataset.tar.bz2'
        self.restricted_compiler_compress_name='restricted_compiler_dataset.tar.bz2'
        self.compiler_compress_name = 'compiler_dataset.bz2'


    @staticmethod
    def download_file(id,path):
        try:
            print("Downloading from "+ str(id) +" into "+str(path))
            call(['./godown.pl',id,path])
        except Exception as e:
            print("Error downloading file at url:" + str(id))
            print(e)

    @staticmethod
    def decompress_file(file_src,file_path):
        try:
            call(['tar','-xvf',file_src,'-C',file_path])
        except Exception as e:
            print("Error decompressing file:" + str(file_src))
            print('you need tar command e b2zip support')
            print(e)

    def download(self):
        print('Making the godown.pl script executable, thanks:'+str('https://github.com/circulosmeos/gdown.pl'))
        call(['chmod', '+x','godown.pl'])
        print("SAFE --- downloading models")

        if self.i2v:
            print("Downloading i2v model.... in the folder data/i2v/")
            if not os.path.exists(self.path_i2v):
                os.makedirs(self.path_i2v)
            Downloader.download_file(self.url_i2v, os.path.join(self.path_i2v,self.i2v_compress_name))
            print("Decompressing i2v model and placing in " + str(self.path_i2v))
            Downloader.decompress_file(os.path.join(self.path_i2v,self.i2v_compress_name),self.path_i2v)

        if self.openSSL:
            print("Downloading the OpenSSL dataset... in the folder data")
            if not os.path.exists(self.path_openSSL):
                os.makedirs(self.path_openSSL)
            Downloader.download_file(self.url_openSSL, os.path.join(self.path_openSSL, self.openSSL_compress_name))
            print("Decompressing OpenSSL dataset and placing in " + str(self.path_openSSL))
            Downloader.decompress_file(os.path.join(self.path_openSSL, self.openSSL_compress_name), self.path_openSSL)

        if self.restricted_compiler:
            print("Downloading the restricted compiler dataset... in the folder data")
            if not os.path.exists(self.path_restricted_compiler):
                os.makedirs(self.path_restricted_compiler)
            Downloader.download_file(self.url_restricted_compiler, os.path.join(self.path_restricted_compiler,self.restricted_compiler_compress_name))
            print("Decompressing restricted compiler dataset and placing in " + str(self.path_restricted_compiler))
            Downloader.decompress_file(os.path.join(self.path_restricted_compiler, self.restricted_compiler_compress_name), self.path_restricted_compiler)

        if self.compiler:
            print("Downloading the compiler dataset... in the folder data")
            if not os.path.exists(self.path_compiler):
                os.makedirs(self.path_compiler)
            Downloader.download_file(self.url_compiler, os.path.join(self.path_compiler,self.compiler_compress_name))
            print("Decompressing restricted compiler dataset and placing in " + str(self.path_compiler))
            Downloader.decompress_file(os.path.join(self.path_compiler, self.compiler_compress_name), self.path_compiler)


if __name__=='__main__':
    a = Downloader()
    a.download()