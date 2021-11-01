from gensim.models import KeyedVectors
from gensim.corpora.dictionary import Dictionary
from tornado.log import app_log


class Word2VecModel(object):

    def __init__(self):
        self._index_dict = None
        self._word_vectors = None
        self._model = None

    def create_dictionaries(self):
        """
        # create a dictionary of words and phrases,return the index of each word,vector of words,and index of words corresponding to each sentence
            Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries
        """
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(self._model.vocab.keys(),
                            allow_update=True)
        # the index of a word which have word vector is not 0
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        # integrate all the corresponding word vectors into the word vector matrix
        w2vec = {word: self._model[word] for word in w2indx.keys()}
        # return index, word vector matrix and the sentence with an unifying length and indexed
        return w2indx, w2vec

    def load(self, path):
        """
        Load word2vec model
        :param path: model path
        :return: w2v model
        """
        try:
            if isinstance(path, list):
                path = path[0]

            self._model = KeyedVectors.load_word2vec_format(path)
            if self._model is None:
                raise Exception('No Word2vec data provided...')

            # index, word vector matrix and the sentence with an unifying length and indexed based on the trained model
            self._index_dict, self._word_vectors = self.create_dictionaries()
        except Exception as e:
            app_log.error(e)

    def get(self):
        return self._index_dict, self._word_vectors
