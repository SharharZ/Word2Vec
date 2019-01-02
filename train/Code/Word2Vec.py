#!/usr/bin/python
# -*- coding:utf8 -*-
import os
from MySentences import MySentences
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import warnings
import logging
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class Word2Vec:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        print("This is a word2vec training program!!\n")
        print("corpus path：", self.data_path)
        print("output path:", self.output_path)

    @staticmethod
    def word2vec_train_onefile(data_path, output_path):
        result = "train.model"
        print("corpusPath:  ", data_path)
        sents = word2vec.LineSentence(data_path)
        print("Training.......")
        # sg=0 cbow; sg=1 skipgram; hs=0 negative-sampling; hs=1 hierarchical-softmax
        model = word2vec.Word2Vec(sents, size=300, window=5, min_count=1, workers=1, sg=0, hs=0)
        print("Taining finished!!!!\n")
        model.save(output_path+result, ignore=[])
        print("\nmodel finished!!\n")

    @staticmethod
    def word2vec_train(sentences, output_path):
        result = "train.model"
        print("Training.......")
        # 获取日志信息
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
        # sg=0 cbow; sg=1 skipgram; hs=0 negative-sampling; hs=1 hierarchical-softmax
        model = word2vec.Word2Vec(sentences, seed=1, size=200, window=30, min_count=1, workers=1, sg=1, hs=1, iter=50)
        print("Taining finished!!!!\n")
        model.save(output_path+result, ignore=[])
        print("\nmodel finished!!\n")

    @staticmethod
    def train_more(data_path, output_path):
        base_model_name = "train.model"
        incremental_model = "incremental_train.model"
        model = KeyedVectors.load(output_path + base_model_name)
        for root, dirs, files in os.walk(data_path):
            for eachfiles in files:
                data_path = os.path.join(root, eachfiles).replace("\\", "/")
                print("AddCorpusPath:  ", data_path)
                more_sentences = word2vec.LineSentence(data_path)
                print("Incremental Training.......")
                model.train(more_sentences, total_examples=model.corpus_count, epochs=2)
                print("Incremental Training has finished!\n")
                model.save(output_path+incremental_model, ignore=[])

    @staticmethod
    def model_to_bin(path):
        model_name = "train.model"
        bin_name = "train.bin"
        # model convert to bin
        print("Model to bin......")
        model = KeyedVectors.load(path+model_name)
        print("Done loading model file!")
        # 以一种C语言可以解析的形式存储词向量
        model.wv.save_word2vec_format(path+bin_name, binary=True)
        print("Model to bin, Conversion to complete!!\n")

    @staticmethod
    def bin_to_txt(path):
        bin_name = "train.bin"
        txt_name = "train.txt"
        # bin to txt
        print("bin to txt......")
        model = KeyedVectors.load_word2vec_format(path+bin_name, binary=True)
        print("Done loading bin file!")
        f = open(path+txt_name, 'w', encoding='utf8')
        vocab = model.vocab
        f.write(str(len(vocab)) + " " + str(model.vector_size) + "\n\n")
        for item in vocab:
            vector = []
            for dimension in model[item]:
                vector.append(str(dimension))
            vector_str = ",".join(vector)
            line = item + " " + vector_str
            f.write(line + "\n\n")
        f.close()
        print("Bin to txt, Conversion to complete!!\n ")

    def incremental_train(self):
        self.train_more(self.data_path, self.output_path)
        self.model_to_bin(self.output_path)
        self.bin_to_txt(self.output_path)

    def main_train(self):
        sentences = MySentences(self.data_path)
        self.word2vec_train(sentences, self.output_path)
        # self.word2vec_train1(self.data_path, self.output_path)
        self.model_to_bin(self.output_path)
        self.bin_to_txt(self.output_path)
        print("\nTask complete!!!")
