import os
from rw_flock import RWFlocker
from utils import get_logger

_cwd_path = os.getcwd()

_fasttext_model_path = [os.path.join(_cwd_path, "resources/fasttext_model.bin")]

_lstm_model_path = [
                    os.path.join(_cwd_path, "resources/lstm_model.yml"),
                    os.path.join(_cwd_path, "resources/lstm_model.h5")
                   ]

_textcnn_model_path = [
                       os.path.join(_cwd_path, "resources/textcnn_model.yml"),
                       os.path.join(_cwd_path, "resources/textcnn_model.h5")
                      ]


_model_paths = {
    "Fasttext": _fasttext_model_path,
    "LSTM": _lstm_model_path,
    "TEXTCNN": _textcnn_model_path
}


_preferred_model = "TEXTCNN"


logger = get_logger("sentiment-commit-model")


def _execute():

    model_path = _model_paths.get(_preferred_model)
    if model_path is None:
        return

    try:
        RWFlocker.lock(RWFlocker.WRITE)
        for path in model_path:
            p = path.rfind('.')
            main_path = path
            postfix = ''
            if p > 0:
                main_path = main_path[: p]
                postfix = path[p + 1:]

            new_path = main_path + '_updated'
            if len(postfix) > 0:
                new_path += '.' + postfix

            logger.info("start to commit {}".format(new_path))

            if os.path.exists(new_path):
                os.remove(path)
                os.rename(new_path, path)

        RWFlocker.unlock()
    except Exception as e:
        logger.error(e)
    finally:
        RWFlocker.unlock()


if __name__ == "__main__":
    _execute()

