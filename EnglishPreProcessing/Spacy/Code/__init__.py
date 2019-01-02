import time
from EnPreprocess import EnPreprocess

if __name__ == '__main__':
    start = time.clock()
    data_path = "data/"
    output_path = "output/"
    epp = EnPreprocess(data_path, output_path)
    epp.en_pre_main()
    end = time.clock()
    print("Testing time used: %s seconds" % (end-start))
    print("Testing time used: %s minutes" % ((end-start)/60))
