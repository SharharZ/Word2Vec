import time
from Word2Vec import Word2Vec

if __name__ == '__main__':
    start = time.clock()
    data_path = "corpus/"
    output_path = "model/"
    w2v = Word2Vec(data_path, output_path)
    w2v.main_train()
    end = time.clock()
    print("Training time used: %s seconds" % (end-start))
    print("Training time used: %s minutes" % ((end-start)/60))
