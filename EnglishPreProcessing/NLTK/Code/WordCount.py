import os
from collections import defaultdict


def word_count(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for eachfiles in files:
                data_path = os.path.join(root, eachfiles).replace("\\", "/")

                print('Counting words in {}'.format(eachfiles))
                with open(data_path) as fin:
                    total_count = 0
                    vocabulary = defaultdict(int)
                    for line in fin:
                        words = line.split()
                        total_count += len(words)
                        for word in words:
                            vocabulary[word] = vocabulary[word] + 1
                    with open(data_dir + 'Count_' + eachfiles, 'wt', encoding='utf-8') as fout:
                        fout.write('Total words: {}\nVocabulary size: {}\n'.format(total_count, len(vocabulary)))
                        sorted_list = sorted(vocabulary, key=vocabulary.get, reverse=True)
                        for word in sorted_list:
                            fout.write("{} {}\n".format(word, vocabulary[word]))

if __name__ == '__main__':
    path = 'output/'
    word_count(path)
