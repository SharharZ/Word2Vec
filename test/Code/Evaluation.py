#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import logging
import pandas as pd
from EnPreProcess import EnPreProcess
from gensim.models.keyedvectors import KeyedVectors
from txtsim import Phrase2VecByMean
import time

FORMAT = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=FORMAT, level=logging.CRITICAL)
logger = logging.getLogger('evaluation')


class Evaluation:

    def __init__(self, test_path, model_path, result_path):
        print("Testing begin!!")
        print('\n')
        self.test_path = test_path
        self.model_path = model_path
        self.result_path = result_path

    @staticmethod
    def read_csv(test_path, sheet_name):
        logger.info("Loading test data")
        for root, dirs, files in os.walk(test_path):
            for eachfiles in files:
                data_path = os.path.join(root, eachfiles).replace("\\", "/")
                data = pd.read_excel(data_path, sheet_name=sheet_name)
        return data

    @staticmethod
    def _normalized(data):
        sents = EnPreProcess.sen_token(data)  # 分句
        # tagged_line=self.pos_tagger(sents)#暂不启用词性标注
        clean_lines = [EnPreProcess.clean_lines(line) for line in sents]  # 清洗句子，去掉标点符号，数字
        # print(clean_lines)
        words = [EnPreProcess.word_tokener(cl) for cl in clean_lines]  # 分词
        # checked_words=self.word_check(words)#暂不启用拼写检查
        clean_words = EnPreProcess.clean_words(words)  # 转化为小写并且去掉停用词
        # print(clean_words)
        # stem_words = self.stemmer_words(clean_words)#提取词干
        lemmatize_words = EnPreProcess.lemmatize_words(clean_words)  # 词性还原
        # print(lemmatize_words)
        str_lines = EnPreProcess.words_to_str(lemmatize_words)  # 重新合成为句子
        str_line = ' '.join(phrase for phrase in str_lines if phrase != '\n')
        return str_line

    @staticmethod
    def evaluate(test_data_df, prcstructures_df, model, topn):
        mg_desc_column = 'Material Group Description'
        ps_desc_column = 'Procurement Structure Description'
        structure_desc_column = 'Description'
        correct_count = 0
        for i in range(len(test_data_df)):
            result = model.calculate_similarity(test_data_df.loc[i, mg_desc_column],
                                                prcstructures_df.loc[:, structure_desc_column])
            for n in range(topn):
                if test_data_df.loc[i, ps_desc_column] == result[n]['target_doc']:
                    correct_count += 1
                    break
        return correct_count

    def evaluate_w2v(self, test_data_df, prcstructures_df, model_file, binary):
        top_n = [1, 3, 5]
        material_group_count = len(test_data_df)
        logger.info("Loading model file {}".format(model_file))
        st = time.clock()
        w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=binary)
        # w2v_model.init_sims(True)
        
        models = list()
        models.append(Phrase2VecByMean(w2v_model))
        # models.append(WmdTxtSim(w2v_model))

        # index = random.randint(0, material_group_count-1)
        # example = (test_data_df.loc[index, mg_desc_column], test_data_df[index, ps_desc_column])
        # # print("example: " + str(example))
        # models.append(AnalogySim(w2v_model, example))
        filename = model_file.split('/')[-1]
        ed = time.clock()
        print("Loading model time used: %s seconds" % (ed - st))
        print("\n")

        st1 = time.clock()
        for model in models:
            model_name = type(model).__name__
            logger.debug("Evaluating {} with {} ".format(filename, model_name))
            for topn in top_n:
                correct_count = self.evaluate(test_data_df, prcstructures_df, model, topn)
                accuracy = correct_count * 1.0 / material_group_count
                print("Top {} {}\t{}\t{}/{}\t{}".format(topn, model_name, filename, correct_count, material_group_count,
                                                        accuracy))
            print("\n")
        ed1 = time.clock()
        print("Computing accuracy time used: %s seconds" % (ed1 - st1))
        print("\n")

    def evaluation_main(self):
        bin_model_dir = "models/bin"
        txt_model_dir = "models/txt"
        test_sheet = 'Test_Set'
        mg_desc_column = 'Material Group Description'
        ps_desc_column = 'Procurement Structure Description'
        # ps_code_column = 'Procurement Structure Code'
        procurement_structure_sheet = 'Procurement_Structure'
        structure_desc_column = 'Description'
        # structure_code_column = 'Structure Code'
        test_data_df = self.read_csv(self.test_path, test_sheet)
        prcstructures_df = self.read_csv(self.test_path, procurement_structure_sheet)
        material_group_count = len(test_data_df)
        prcstructure_count = len(prcstructures_df)
        logger.debug(
         "Totally {} Material Groups and {} Procurement Structures".format(material_group_count, prcstructure_count))

        st = time.clock()
        logger.info('Pre-processing start...')
        for i in range(material_group_count):
            test_data_df.loc[i, mg_desc_column] = self._normalized(test_data_df.loc[i, mg_desc_column])
            test_data_df.loc[i, ps_desc_column] = self._normalized(test_data_df.loc[i, ps_desc_column])

        for i in range(prcstructure_count):
            prcstructures_df.loc[i, structure_desc_column] = self._normalized(prcstructures_df.loc[i, structure_desc_column])
            logger.debug(prcstructures_df.loc[i, structure_desc_column])
        logger.info('Pre-processing ends...')
        ed = time.clock()
        print("Preprocess data time used: %s seconds" % (ed - st))
        print("\n")

        if self.model_path == bin_model_dir:
            for model_file in os.listdir(self.model_path):
                self.evaluate_w2v(test_data_df, prcstructures_df, self.model_path + "/" + model_file, True)
        elif self.model_path == txt_model_dir:
            for model_file in os.listdir(self.model_path):
                self.evaluate_w2v(test_data_df, prcstructures_df, self.model_path + "/" + model_file, False)