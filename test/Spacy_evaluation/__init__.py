import time
from Evaluation import Evaluation

if __name__ == '__main__':
    start = time.clock()
    test_path = "data/"
    model_path = "models/bin"
    result_path = "result/"
    ev = Evaluation(test_path, model_path, result_path)
    ev.evaluation_main()
    end = time.clock()
    print("Testing time used: %s seconds" % (end-start))
    print("Testing time used: %s minutes" % ((end-start)/60))
