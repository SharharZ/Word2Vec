import glob
import logging
import os
import time
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors

from txtsim import Phrase2VecByMean, WmdTxtSim, AnalogySim


FORMAT = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=FORMAT, level=logging.CRITICAL)
logger = logging.getLogger('evaluation')


FILENAME = 'data/PS_MG_with_Demo_TestSet.xlsx'

BIN_MODEL_DIR = 'models/bin'
TXT_MODEL_DIR = 'models/txt'

TEST_SHEET = 'Test_Set'
MG_DESC_COLUMN = 'Material Group Description'
PS_DESC_COLUMN = 'Procurement Structure Description'
PS_CODE_COLUMN = 'Procurement Structure Code'
PROCUREMENT_STRUCTURE_SHEET = 'Procurement_Structure'
STRUCTURE_DESC_COLUMN = 'Description'
STRUCTURE_CODE_COLUMN = 'Structure Code'

STOP_WORDS = ['/', '=', '+', '*', ',']
TOP_N = [1, 3, 5]

IGNOR_FAKE_MODEL = True


def _normalized(str):
    for word in STOP_WORDS:
        str = str.replace(word, ' ')
    return ' '.join(str.lower().split())


def evaluate(test_data_df, prcstructures_df, model, topn):
    correct_count = 0
    for i in range(len(test_data_df)):
        result = model.calculate_similarity(test_data_df.loc[i, MG_DESC_COLUMN], prcstructures_df.loc[:, STRUCTURE_DESC_COLUMN])
        for n in range(topn):
            if test_data_df.loc[i, PS_DESC_COLUMN] == result[n]['target_doc']:
                correct_count += 1
                break
    return correct_count


def evaluate_w2v(test_data_df, prcstructures_df, model_file, binary):
    if IGNOR_FAKE_MODEL and 'fake_model' in model_file:
        return
    material_group_count = len(test_data_df)
    logger.info("Loading model file {}".format(model_file))
    st = time.clock()
    w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=binary)
    # w2v_model.init_sims(True)

    models = []
    models.append(Phrase2VecByMean(w2v_model))
    # models.append(WmdTxtSim(w2v_model))

    # index = random.randint(0, material_group_count-1)
    # example = (test_data_df.loc[index, MG_DESC_COLUMN], test_data_df.loc[index, PS_DESC_COLUMN])
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
        for topn in TOP_N:
            correct_count = evaluate(test_data_df, prcstructures_df, model, topn)
            accuracy = correct_count * 1.0 / material_group_count        
            print("Top {} {}\t{}\t{}/{}\t{}".format(topn, model_name, filename, correct_count, material_group_count, accuracy))
        print("\n")
    ed1 = time.clock()
    print("Computing accuracy time used: %s seconds" % (ed1 - st1))
    print("\n")


def main():
    logger.info("Loading test data")
    test_data_df = pd.read_excel(FILENAME, sheet_name=TEST_SHEET)
    prcstructures_df = pd.read_excel(FILENAME, sheet_name=PROCUREMENT_STRUCTURE_SHEET)

    material_group_count = len(test_data_df)
    prcstructure_count = len(prcstructures_df)
    logger.debug("Totally {} Material Groups and {} Procurement Structures".format(material_group_count, prcstructure_count))

    st = time.clock()
    logger.info('Pre-processing start...')
    for i in range(material_group_count):
        test_data_df.loc[i, MG_DESC_COLUMN] = _normalized(test_data_df.loc[i, MG_DESC_COLUMN])
        test_data_df.loc[i, PS_DESC_COLUMN] = _normalized(test_data_df.loc[i, PS_DESC_COLUMN])

    for i in range(prcstructure_count):
        prcstructures_df.loc[i, STRUCTURE_DESC_COLUMN] = _normalized(prcstructures_df.loc[i, STRUCTURE_DESC_COLUMN])
        logger.debug(prcstructures_df.loc[i, STRUCTURE_DESC_COLUMN])
    logger.info('Pre-processing ends...')
    ed = time.clock()
    print("Preprocess data time used: %s seconds" % (ed - st))
    print("\n")

    for model_file in os.listdir(BIN_MODEL_DIR):
        evaluate_w2v(test_data_df, prcstructures_df, BIN_MODEL_DIR + "/" + model_file, True)

    for model_file in os.listdir(TXT_MODEL_DIR):
        evaluate_w2v(test_data_df, prcstructures_df, TXT_MODEL_DIR + "/" + model_file, False)


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print("Testing time used: %s seconds" % (end - start))
    print("Testing time used: %s minutes" % ((end - start) / 60))
