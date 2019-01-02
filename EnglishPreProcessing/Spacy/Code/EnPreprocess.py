#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import re
import spacy
import logging


class EnPreprocess:
    def __init__(self, data_path, output_path):
        print("This is a English corpus preprocess program!!")
        print('\n')
        self.data_path = data_path
        self.output_path = output_path

    @staticmethod
    def file_read(file_path):  # 读取内容
        with open(file_path, encoding='UTF-8', errors='ignore') as f:
            raw = f.read()
            f.close()

        return raw

    def write_result(self, result, data_path, output_path):
        txt_path = os.path.basename(data_path)
        cwd = os.getcwd()
        path = re.sub(str(data_path).split('/')[-1], '', data_path)
        path = re.sub(str(path).split('/')[0], output_path, path)
        dir_path = cwd + '/' + re.sub('//', '/', path)
        self.mkdir(dir_path)

        with open(dir_path+txt_path, 'w', encoding='utf8') as f:
            for i in result:
                if i != '\n':
                    f.write(str(i) + ' ')
                else:
                    f.write(str(i))

    @staticmethod
    def mkdir(path):
        # 去除首位空格
        path = path.strip()
        # 去除尾部 \ 符号
        path = path.rstrip("\\")
        # 判断路径是否存在
        # 存在    True
        # 不存在  False
        is_exists = os.path.exists(path)

        # 判断结果
        if not is_exists:
            # 如果不存在则创建目录
            # print(path + '  create successfully!')
            print('\n')
            # 创建目录操作函数
            os.makedirs(path)
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            # print(path+'  The directory exists!!')
            print('\n')
            return False

    @staticmethod
    def words_to_str(clean_lines):  # 转换成字符串
        str_line = []
        for lines in clean_lines:
            lines += '\n'
            for line in lines:
                str_line.append(line)
        return str_line

    def en_pre_main(self):
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
        nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
        stopwords = '[’!"#$%&\'•●®()*+,-./:;<=>?@[\\]^_`{|}~]+'
        punctuations = ['be']
        clean_words = []
        for root, dirs, files in os.walk(self.data_path):
            for eachfiles in files:
                data_path = os.path.join(root, eachfiles).replace("\\", "/")
                print("CopusPath:", data_path)
                raw = self.file_read(data_path).strip()
                doc = nlp(raw)
                for tokens in doc.sents:
                    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
                    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
                    clean_words.append(tokens)
                    print(tokens)
                str_line = self.words_to_str(clean_words)  # 重新合成为句子
                self.write_result(str_line, data_path, self.output_path)
                print(data_path + "     Prepcocess OK!!")
                print('\n')
        print("All files finished!!!")
