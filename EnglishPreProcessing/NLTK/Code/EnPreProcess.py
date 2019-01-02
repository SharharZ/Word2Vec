#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class EnPreProcess:
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
    def get_html(url):
        page = urllib.request.urlopen(url)
        html = page.read().decode("utf-8")
        soup = bs4.BeautifulSoup(html, 'html.parser')
        soup.prettify()
        tds = soup.find_all('td')

        td_content = []
        for i in tds:
            # print(i.text) #这里取标签span的内容
            td_content.append(i.text)
        # print(td_content)
        return html

    @staticmethod
    def sen_token(raw):  # 分割成句子
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_tokenizer.tokenize(raw)
        return sents

    @staticmethod
    def pos_tagger(sents):
        tagged_line = [nltk.pos_tag(sent) for sent in sents]
        return tagged_line

    @staticmethod
    def word_tokener(sent):  # 将单句字符串分割成词
        words_in_str = nltk.word_tokenize(sent)
        return words_in_str

    @staticmethod
    def word_check(words):  # 拼写检查
        d = enchant.Dict("en_US")
        checked_words = ()
        for word in words:
            if not d.check(word):
                d.suggest(word)
                word = raw_input()
            checked_words = (checked_words, '05')
        return checked_words

    @staticmethod
    def clean_lines(line):  # 去除数字和ACSCII符号
        a = '[’!"#$%&\'•●®()*+,-./:;<=>?@[\\]^_`{|}~]+'
        line = re.sub(a, " ", line)
        line1 = re.sub('\d', " ", line)  # 数字
        line2 = re.sub('\n', "", line1)  # 去掉换行符
        clean_line = re.sub('[^a-zA-Z]', ' ', line2).replace("\n", '')

        return clean_line

    @staticmethod
    def clean_words(words_in_str):  # 小写化，去掉停用词
        clean_words = []
        stop_words= stopwords.words('english')
        stop_words.append('cid')
        stop_words.append('\n')

        for words in words_in_str:
            # clean_words += [[w.lower() for w in words if w.lower() not in stopwords.words('english') and 3<=len(w)]]

            clean_words += [[w.lower() for w in words if w.lower() not in stop_words and 3 <= len(w)]]
            # clean_words += [[w.lower() for w in words if len(w)>=3]]

        return clean_words

    @staticmethod
    def stemmer_words(clean_words_list):  # 提取词干
        stem_words = []
        # porter = nltk.PorterStemmer()
        # stem_words = [porter.stem(t) for t in clean_words_list]

        # for words in clean_words_list:
        # stem_words += [[wn.morphy(w) for w in words]]
        s = nltk.stem.SnowballStemmer('english')
        stem_words += [[s.stem(t) for t in clean_words_list]]

        return stem_words

    @staticmethod
    def lemmatize_words(clean_words_list):  # 词性还原
        le_words = []
        l = WordNetLemmatizer()
        for words in clean_words_list:
            le_words += [l.lemmatize(t) for t in words]
            le_words.append('\n')
        return le_words

    @staticmethod
    def words_to_str(stem_words):  # 转换成字符串
        # num = 0
        # str_line = []
        # for words in stem_words:
        #     for w in words:
        #         num += 1
        #         if num > 20:
        #             str_line.append('\n')
        #             num = 0
        #         else:
        #             str_line.append(w)

        str_line = []
        str_line += [w for w in stem_words]
        return str_line

    def en_pre_main(self):
        for root, dirs, files in os.walk(self.data_path):
            for eachfiles in files:
                data_path = os.path.join(root, eachfiles).replace("\\", "/")
                print("CopusPath:", data_path)
                raw = self.file_read(data_path).strip()
                sents = self.sen_token(raw)  # 分句
                # tagged_line=self.pos_tagger(sents)#暂不启用词性标注
                clean_lines = [self.clean_lines(line) for line in sents]  # 清洗句子，去掉标点符号，数字
                # print(clean_lines)
                words = [self.word_tokener(cl) for cl in clean_lines]  # 分词
                # checked_words=self.word_check(words)#暂不启用拼写检查
                clean_words = self.clean_words(words)  # 转化为小写并且去掉停用词
                # print(clean_words)
                # stem_words = self.stemmer_words(clean_words)#提取词干
                lemmatize_words = self.lemmatize_words(clean_words)  # 词性还原
                # print(lemmatize_words)
                str_line = self.words_to_str(lemmatize_words)  # 重新合成为句子
                self.write_result(str_line, data_path, self.output_path)
                print(data_path + "     Prepcocess OK!!")
                print('\n')
        print("All files finished!!!")
