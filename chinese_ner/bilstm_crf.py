import os
import re
import pickle
import numpy as np
import config
from config import ner_model_config
from chinese_ner.batch import BatchGenerator
import tensorflow as tf
from chinese_ner import utils
from chinese_ner import logger as app_log


class Model(object):

    def __init__(self):
        self.sess = None
        self.initialized = self._init_train()
        if not self.initialized:
            app_log.error("Fail to initialize Bilstm-ctf model")

    def __del__(self):
        tf.reset_default_graph()
        self.close()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def has_trained(self):
        return self.sess is not None

    def train(self):
        if not self.initialized:
            self.initialized = self._init_train()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            self._train(sess, saver)

        self.init_predict()

    def predict(self, text):
        if self.sess is None:
            app_log.info('Model not found, please train your model first')
            return []

        max_len = ner_model_config["max_len"]

        text = re.split(u'[，。！？、‘’“”（）]', text)
        text_id = []
        for sen in text:
            word_id = []
            for word in sen:
                if word in self.word2id:
                    word_id.append(self.word2id[word])
                else:
                    word_id.append(self.word2id["unknown"])
            text_id.append(utils.padding(word_id))

        zero_padding = []
        zero_padding.extend([0] * max_len)
        text_id.extend([zero_padding] * (self.batch_size - len(text_id)))
        feed_dict = {self.input_data: text_id}
        predict = self.sess.run([self.viterbi_sequence], feed_dict)
        return self._get_entity(text, predict[0])

    def extract_nt(self, text):
        entities = self.predict(text)
        nt_results = []
        for entity in entities:
            entity = entity.split(':')
            if len(entity) == 1:
                continue
            if entity[0] == "nt" and isinstance(entity[1], str):
                nt_results.append(entity[1].strip())

        return nt_results

    def extract_ns(self, text):
        entities = self.predict(text)
        ns_results = []
        for entity in entities:
            entity = entity.split(':')
            if entity[0] == "ns":
                ns_results.append(entity[1].strip())

        return ns_results

    def init_predict(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(config.ner_model_path)
        if ckpt is None:
            app_log.info('Model not found, please train your model first')
            return None
        else:
            path = ckpt.model_checkpoint_path
            app_log.info('loading pre-trained model from {}.....'.format(path))
            saver.restore(self.sess, path)

    def _init_train(self):
        if not os.path.exists(config.ner_train_data_pkl_path):
            app_log.error("{} does not exist".format(config.ner_train_data_pkl_path))
            return False

        try:
            with open(config.ner_train_data_pkl_path, 'rb') as inp:
                self.word2id = pickle.load(inp)
                self.id2word = pickle.load(inp)
                self.tag2id = pickle.load(inp)
                self.id2tag = pickle.load(inp)
                x_train = pickle.load(inp)
                y_train = pickle.load(inp)
                x_test = pickle.load(inp)
                y_test = pickle.load(inp)
                x_valid = pickle.load(inp)
                y_valid = pickle.load(inp)

            app_log.info("train len: {}".format(len(x_train)))
            app_log.info("test len: {}".format(len(x_test)))
            app_log.info("word2id len: {}".format(len(self.word2id)))
            app_log.info('Creating the data generator ...')

            self.data_train = BatchGenerator(x_train, y_train, shuffle=True)
            self.data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
            self.data_test = BatchGenerator(x_test, y_test, shuffle=False)

            self.sen_len = len(x_train[0])
            self.embedding_size = len(self.word2id) + 1
            self.tag_size = len(self.tag2id)

            if ner_model_config["pretrained"]:
                self.embedding_pretrained, self.pretrained = self._use_character_vectors()
            else:
                self.pretrained = False

            self.lr = ner_model_config["lr"]
            self.batch_size = ner_model_config["batch_size"]
            self.embedding_dim = ner_model_config["embedding_dim"]
            self.dropout_keep = ner_model_config["dropout_keep"]

            self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.sen_len], name="input_data")
            self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.sen_len], name="labels")
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.embedding_size,
                                                                           self.embedding_dim],
                                                        name="embedding_placeholder")

            with tf.variable_scope("bilstm_crf", reuse=tf.AUTO_REUSE) as scope:
                self._build_net()

            app_log.info('Finished creating the data generator.')
            return True
        except Exception as e:
            app_log.error("Fail to intialize train process: {}".format(e))
            return False

    def _use_character_vectors(self):
        if not os.path.exists(config.chinese_character_vector_path):
            app_log.error("{} does not exist".format(config.chinese_character_vector_path))
            return None, False

        app_log.info("use chinese character pretrained embedding")

        try:
            word2vec = {}
            with open(config.chinese_character_vector_path, 'r') as input_data:
                for line in input_data.readlines():
                    word2vec[line.split()[0].strip()] = map(eval, line.split()[1:])

            unknown_pre = []
            unknown_pre.extend([1] * ner_model_config["embedding_dim"])

            embedding_pre = [unknown_pre]

            for word in self.word2id:
                if word in word2vec:
                    embedding_pre.append(word2vec[word])
                else:
                    embedding_pre.append(unknown_pre)

            embedding_pre = np.asarray(embedding_pre)
            return embedding_pre, True
        except Exception as e:
            app_log.error("Fail to read chinese character vectors: {}".format(e))
            return None, False

    def _calculate(self, x, y):
        results = []
        entity = []

        try:
            for i in range(len(x)):  # for every sentence
                for j in range(len(x[i])):  # for every word
                    if x[i][j] == 0 or y[i][j] == 0:
                        continue
                    if self.id2tag[y[i][j]][0] == 'B':
                        entity = [self.id2word[x[i][j]] + '/' + self.id2tag[y[i][j]]]
                    elif self.id2tag[y[i][j]][0] == 'M' and len(entity) != 0 \
                            and entity[-1].split('/')[1][1:] == self.id2tag[y[i][j]][1:]:
                        entity.append(self.id2word[x[i][j]] + '/' + self.id2tag[y[i][j]])
                    elif self.id2tag[y[i][j]][0] == 'E' and len(entity) != 0 \
                            and entity[-1].split('/')[1][1:] == self.id2tag[y[i][j]][1:]:
                        entity.append(self.id2word[x[i][j]] + '/' + self.id2tag[y[i][j]])
                        entity.append(str(i))
                        entity.append(str(j))
                        results.append(entity)
                        entity = []
                    else:
                        entity = []
            return results
        except Exception as e:
            app_log.error("Fail to parse model results：{}".format(e))
            return None

    def _get_entity(self, x, y):
        entity = ""
        results = []
        for i in range(len(x)):  # for every sentence
            for j in range(len(x[0])):  # for every word
                if y[i][j] == 0:
                    continue
                if self.id2tag[y[i][j]][0] == 'B':
                    entity = self.id2tag[y[i][j]][2:] + ':' + x[i][j]
                elif self.id2tag[y[i][j]][0] == 'M' and len(entity) != 0:
                    entity += x[i][j]
                elif self.id2tag[y[i][j]][0] == 'E' and len(entity) != 0:
                    entity += x[i][j]
                    results.append(entity)
                    entity = ""
                else:
                    entity = ""
        return results

    def _evaluate(self, sess, data_generator, batch_num):
        pred_entries = []
        true_entries = []
        for batch in range(batch_num):
            x_batch, y_batch = data_generator.next_batch(self.batch_size)
            feed_dict = {self.input_data: x_batch, self.labels: y_batch}
            predict = sess.run([self.viterbi_sequence], feed_dict)[0]

            pred_entries.extend(self._calculate(x_batch, predict))
            true_entries.extend(self._calculate(x_batch, y_batch))

        intersection = [i for i in pred_entries if i in true_entries]

        if len(intersection) > 0:
            precision = float(len(intersection)) / len(pred_entries)
            recall = float(len(intersection)) / len(true_entries)
            app_log.info("Precision: {}".format(precision))
            app_log.info("Recall: {}".format(recall))
            app_log.info("F1: {}".format((2 * precision * recall) / (precision + recall)))
        else:
            app_log.info("Precision: 0")

    def _build_net(self):
        word_embeddings = tf.get_variable("word_embeddings", [self.embedding_size, self.embedding_dim])
        if self.pretrained:
            embeddings_init = word_embeddings.assign(self.embedding_pretrained)

        input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)
        input_embedded = tf.nn.dropout(input_embedded, self.dropout_keep)

        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_dim, forget_bias=1.0, state_is_tuple=True)
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                         lstm_bw_cell,
                                                                         input_embedded,
                                                                         dtype=tf.float32,
                                                                         time_major=False,
                                                                         scope=None)

        bilstm_out = tf.concat([output_fw, output_bw], axis=2)


        # Fully connected layer.
        W = tf.get_variable(name="W", shape=[self.batch_size, 2 * self.embedding_dim, self.tag_size],
                            dtype=tf.float32)

        b = tf.get_variable(name="b", shape=[self.batch_size, self.sen_len, self.tag_size], dtype=tf.float32,
                            initializer=tf.zeros_initializer())

        bilstm_out = tf.tanh(tf.matmul(bilstm_out, W) + b)

        # Linear-CRF.
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(bilstm_out, self.labels,
                                                                                   tf.tile(np.array([self.sen_len]),
                                                                                           np.array([self.batch_size])))

        loss = tf.reduce_mean(-log_likelihood)

        # Compute the viterbi sequence and score (used for prediction and test time).
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_out, self.transition_params,
                                                                         tf.tile(np.array([self.sen_len]),
                                                                                 np.array([self.batch_size])))

        # Training ops.
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(loss)

    def _train(self, sess, saver):
        if not self.initialized:
            app_log.error("BiLSTM-CRF model uninitialized")
            return

        batch_num = int(self.data_train.y.shape[0] / self.batch_size)
        batch_num_test = int(self.data_test.y.shape[0] / self.batch_size)
        epochs = ner_model_config["epochs"]

        if not os.path.exists(config.chinese_character_vector_path):
            os.makedirs(config.chinese_character_vector_path)

        for epoch in range(epochs):
            for batch in range(batch_num):
                x_batch, y_batch = self.data_train.next_batch(self.batch_size)
                # print x_batch.shape
                feed_dict = {self.input_data: x_batch, self.labels: y_batch}
                predicts, _ = sess.run([self.viterbi_sequence, self.train_op], feed_dict)
                acc = 0
                if batch % 200 == 0:
                    for i in range(len(y_batch)):
                        for j in range(len(y_batch[i])):
                            if y_batch[i][j] == predicts[i][j]:
                                acc += 1

                    app_log.info("Accuracy is {}".format(float(acc)/(len(y_batch) * len(y_batch[0]))))

            path_name = config.ner_model_path + "model" + str(epoch) + ".ckpt"
            app_log.info(path_name)

            if epoch % 2 == 0:
                saver.save(sess, path_name)
                app_log.info("model has been saved")
                app_log.info("Train ...")
                self._evaluate(sess, self.data_train, batch_num)
                app_log.info("Test ...")
                self._evaluate(sess, self.data_test, batch_num_test)


