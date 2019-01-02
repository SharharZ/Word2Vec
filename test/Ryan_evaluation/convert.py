import os

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors


def golve2w2v(src_dir, dst_dir):
    for file in os.listdir(src_dir):
        print("Converting {}".format(file))
        glove2word2vec(src_dir + "\\" + file, dst_dir + "\\" + file)


def w2v_txt2bin(src_file, dst_file):
    model = KeyedVectors.load_word2vec_format(src_file, binary=False)
    model.save_word2vec_format(dst_file, binary=True)


def w2v_bin2txt(src_file, dst_file):
    model = KeyedVectors.load_word2vec_format(src_file, binary=True)
    model.save_word2vec_format(dst_file, binary=False)


if __name__ == '__main__':
    # golve2w2v("C:\\Users\\two\\Desktop\\May_HackFest\\evaluation\\Jeff", "models\\txt")
    # w2v_bin2txt('models/txt/fake_model.txt', 'models/bin/fake_model.bin')
    
    for file in os.listdir('models/txt'):
        if not file.startswith('fake'):
            w2v_txt2bin('models/txt/' + file, 'models/bin/' + file)
