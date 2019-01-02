import logging
from numpy import array, dot
from gensim import matutils  # utility fnc for pickling, common scipy operations etc

logger = logging.getLogger('txtsim')


class BaseTxtSim(object):
    """Docstring for class TxtSim"""

    def __init__(self, w2v_model):
        super(BaseTxtSim, self).__init__()
        self.w2v_model = w2v_model
        self._nonexist_docs = set()

    def _calculate_score(self, src_words, target_words):
        return 0.0

    def calculate_similarity(self, source_doc, target_docs=[]):
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        results = []
        for target_doc in target_docs:
            src_words = self._filter_nonexist(source_doc.split())
            target_words = self._filter_nonexist(target_doc.split())

            sim_score = self._calculate_score(src_words, target_words)
            results.append({
                'score': sim_score,
                'target_doc': target_doc
            })

        ''' Sort results by score in desc order '''
        results.sort(key=lambda k: k['score'], reverse=True)

        if len(results) > 0:
            logger.debug("result for [{0}]: {1}".format(source_doc, results[0]))

        return results

    def _filter_nonexist(self, words):
        result = []
        for word in words:
            try:
                self.w2v_model[word]
                result.append(word)
            except KeyError:
                pass

        if len(result) == 0 and ' '.join(words) not in self._nonexist_docs:
            self._nonexist_docs.add(' '.join(words))
            logger.debug('[{0}] is invalid'.format(' '.join(words)))
        return result


class Phrase2VecByMean(BaseTxtSim):
    def __init__(self, w2v_model):
        super(Phrase2VecByMean, self).__init__(w2v_model)
        pass

    def _calculate_score(self, src_words, target_words):
        if len(src_words) > 0 and len(target_words) > 0:
            return self.w2v_model.n_similarity(
                src_words, target_words)
        return 0.0


class WmdTxtSim(BaseTxtSim):
    def __init__(self, w2v_model):
        super(WmdTxtSim, self).__init__(w2v_model)

    def _calculate_score(self, src_words, target_words):
        if len(src_words) > 0 and len(target_words) > 0:
            return self.w2v_model.wmdistance(
                src_words, target_words) * -1.0
        return -10000000.0


class AnalogySim(BaseTxtSim):
    def __init__(self, w2v_model, example):
        super(AnalogySim, self).__init__(w2v_model)
        self.example = example
        src_words = self._filter_nonexist(example[0])
        dst_words = self._filter_nonexist(example[1])

        self.delta = None
        if len(src_words) > 0 and len(dst_words) > 0:
            src_vec = self._get_vector_by_mean(src_words)
            dst_vec = self._get_vector_by_mean(dst_words)
            self.delta = dst_vec - src_vec

    def _calculate_score(self, src_words, target_words):
        if len(src_words) <= 0 or len(target_words) <= 0:
            return 0.0
        v1 = self._get_vector_by_mean(src_words)
        v2 = self._get_vector_by_mean(target_words)
        if self.delta is not None:
            v1 = v1 + self.delta
        return self._consine_similarity(v1, v2)

    def _get_vector_by_mean(self, words):
        result = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                result.append(vec)
            except KeyError:
                pass
        return array(result).mean(axis=0)

    @staticmethod
    def _consine_similarity(v1, v2):
        return dot(matutils.unitvec(v1), matutils.unitvec(v2))
