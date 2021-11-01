import config
from utils import get_celery_sentiment_app, create_lock_file, get_city_dict
from sentiment.models.word2vec import Word2VecModel

_word2vec = None


def _get_word2vec():
    global _word2vec
    if _word2vec:
        return _word2vec

    _word2vec = Word2VecModel()
    _word2vec.load(config.word2vec_model_path)

    return _word2vec


celery_app = get_celery_sentiment_app(config)

word2vec_model = _get_word2vec()

city_dict = get_city_dict()

create_lock_file(config.write_lock_filepath)

print("initialized Finished")



