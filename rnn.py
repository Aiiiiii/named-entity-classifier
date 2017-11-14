"""
See README.md
References:
https://github.com/rockingdingo/deepnlp/tree/master/deepnlp/ner
https://github.com/dhwajraj/NER-RNN/blob/master/NerMulti.py
https://www.tensorflow.org/tutorials/recurrent
"""

from __future__ import print_function
import pickle
import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
import sys
import pickle as pkl
import argparse
from random import random
import numpy as np
import tensorflow as tf
import numpy as np
import os
import urllib.request
from nltk.corpus import brown


def download_data():
    """ Download labeled data.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'    
    urllib.request.urlretrieve(url, 'test.txt')
    print ('download labeled data!')


#resize data
def remove_crap(input_file):
    f = open(input_file)
    lines = f.readlines()
    l = list()
    for line in lines:
        if "-DOCSTART-" in line:
            pass
        else:
            l.append(line)
    ff = open('temp.txt', 'w')
    ff.writelines(l)
    ff.close()

def modify_data_size(output_file, resize):
    final_list = list()
    l = list()
    temp_len = 0
    count = 0
    for line in open('temp.txt', 'r'):
        if line in ['\n', '\r\n']:
            if temp_len == 0:
                l = []
            elif temp_len > resize:
                count += 1
                l = []
                temp_len = 0
            else:
                l.append(line)
                final_list.append(l)
                l = []
                temp_len = 0
        else:
            l.append(line)
            temp_len += 1
    f = open(output_file, 'w')
    for i in final_list:
        f.writelines(i)
    f.close()
    print('%d sentences cut out of %d total sentences' % (count, len(final_list)))
    os.system('rm temp.txt')

    
def get_train_data():
    emb = pickle.load(open('train_embed.pkl', 'rb'))
    tag = pickle.load(open('train_tag.pkl', 'rb'))
    print('train data loaded')
    print('train tag dim:%d'%len(tag))
    return emb, tag
def get_test_data():
    emb = pickle.load(open('test_embed.pkl', 'rb'))
    tag = pickle.load(open('test_tag.pkl', 'rb'))
    print('test data loaded')
    print('test tag dim:%d'%len(tag))
    return emb, tag


class RandomVec:
    def __init__(self, dim):
        self.dim = dim
        self.vocab = {}
        self.vec = []

    def __getitem__(self, word):
        ind = self.vocab.get(word, -1)
        if ind == -1:
            new_vec = np.array([random() for i in range(self.dim)])
            self.vocab[word] = len(self.vocab)
            self.vec.append(new_vec)
            return new_vec
        else:
            return self.vec[ind]

class WordVec:
    def __init__(self,dimension):
        print('processing corpus')
        print('training...')
        size = dimension
        self.wvec_model = Word2Vec(brown.sents(), min_count=5, size=size, window=5)
        self.rand_model = RandomVec(dimension)
        
    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.wvec_model[word]
        except KeyError:
            return self.rand_model[word]


def find_max_length(file_name):
    temp_len = 0
    max_length = 0
    for line in open(file_name):
        if line in ['\n', '\r\n']:
            if temp_len > max_length:
                max_length = temp_len
            temp_len = 0
        else:
            temp_len += 1
    return max_length


#add pos embeddings; 5 dimension
def pos(tag):
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def chunk(tag):
    one_hot = np.zeros(5)
    if 'NP' in tag:
        one_hot[0] = 1
    elif 'VP' in tag:
        one_hot[1] = 1
    elif 'PP' in tag:
        one_hot[2] = 1
    elif tag == 'O':
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])



def get_input(model, word_dim, input_file, output_embed, output_tag, sentence_length=-1):
    print('processing %s' % input_file)
    word = []
    tag = []
    sentence = []
    sentence_tag = []
    if sentence_length == -1:
        max_sentence_length = find_max_length(input_file)
    else:
        max_sentence_length = sentence_length
    sentence_length = 0
    print("max sentence length is %d" % max_sentence_length)
    for line in open(input_file):
        if line in ['\n', '\r\n']:
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.array([0] * 5))
                temp = np.array([0 for _ in range(word_dim + 11)])
                word.append(temp)
            sentence.append(word)
            sentence_tag.append(np.array(tag))
            sentence_length = 0
            word = []
            tag = []
        else:
            assert (len(line.split()) == 4)
            sentence_length += 1
            temp = model[line.split()[0]]
            assert len(temp) == word_dim
            temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
            temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
            temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
            word.append(temp)
            #print ('word:\n%s\n'%word[:5])
            t = line.split()[3]
            # 5 classes: 0-None,1-Person,2-Location,3-Organisation,4-Misc
            if t.endswith('I-PER'):
                tag.append(np.array([1, 0, 0, 0, 0]))
            elif t.endswith('I-LOC'):
                tag.append(np.array([0, 1, 0, 0, 0]))
            elif t.endswith('I-ORG'):
                tag.append(np.array([0, 0, 1, 0, 0]))
            elif t.endswith('I-MISC'):
                tag.append(np.array([0, 0, 0, 1, 0]))
            elif t.endswith('O'):
                tag.append(np.array([0, 0, 0, 0, 1]))
            else:
                print("error in input tag {%s}" % t)
                sys.exit(0)
    assert (len(sentence) == len(sentence_tag))
    pkl.dump(sentence, open(output_embed, 'wb'), protocol = 2)
    pkl.dump(sentence_tag, open(output_tag, 'wb'), protocol = 2)



class Model:
    def __init__(self, embword_dim, sentence_len, class_size, rnn_size, num_layers):
        self.input_data = tf.placeholder(tf.float32, [None, sentence_len, embword_dim])
        self.output_data = tf.placeholder(tf.float32, [None, sentence_len, class_size])

        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        fw_cell = tf.contrib.rnn.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)

        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])), dtype=tf.float32, sequence_length=self.length)
        weight, bias = self.weight_and_bias(2 * rnn_size, class_size)
        output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2 * rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, sentence_len, class_size])
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(0.003)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def f1(class_size, prediction, target, length):
    tp = np.array([0] * (class_size+1))
    fp = np.array([0] * (class_size+1))
    fn = np.array([0] * (class_size+1))
    target = np.argmax(target, 2)
    #print('target dim:\n%s'%target.shape[1])
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    unnamed_entity = class_size - 1
    for i in range(class_size):
        if i != unnamed_entity:
            tp[class_size] += tp[i]
            fp[class_size] += fp[i]
            fn[class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(class_size+1):
        precision.append(0 if (tp[i] + fp[i]) == 0 else tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(0 if (tp[i] + fn[i]) == 0 else tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(0 if (precision[i] + recall[i]) == 0 else 2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    return fscore, precision, recall


def train(batch_size, epoch, embword_dim, sentence_len, class_size, rnn_size, num_layers):
    train_inp, train_out = get_train_data()
    test_inp, test_out = get_test_data()
    model = Model(embword_dim, sentence_len, class_size, rnn_size, num_layers)
    maximum = 0
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for e in range(epoch):
            if e == epoch-1:
                fscore, precision, recall = f1(class_size, pred, test_out, length)
                ner = ['I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O']
                fo = open('output2.txt', 'w')
                fo.write('Evaluate result after training %d epoch on %d rnn hidden unites and %d layers\n'%(epoch, rnn_size, num_layers))
                fo.write('NER labels:\n%s\n'%ner)
                fo.write('f1:\n%s\n'%fscore[:5])
                fo.write('precision:\n%s\n'%precision[:5])
                fo.write('recall:\n%s\n'%recall[:5])
                fo.close()
            for ptr in range(0, len(train_inp), batch_size):
                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + batch_size],
                                          model.output_data: train_out[ptr:ptr + batch_size]})
            if e % 10 == 0:
                save_path = saver.save(sess, "model.ckpt")
                print("model saved in file: %s" % save_path)
            pred, length = sess.run([model.prediction, model.length], {model.input_data: test_inp, model.output_data: test_out}) #!!!!
            #print ("prediction:\n%s"%pred)
            #print ('length;\n%s'%length)
            fscore, precision, recall = f1(class_size, pred, test_out, length)
            print("epoch %d:" % e)
            print('test score:')
            ner = ['I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O', 'average']
            print ('%s'%ner)
            print('f1:\n%s'%fscore)
            print ('precision:\n%s'%precision)
            print ('recall:\n%s'%recall)


if __name__ == '__main__':
    #prepare data
    nltk.download('brown')
    download_data()
    maximum_sentence_len = 30
    remove_crap('train.txt')
    modify_data_size('train_resize.txt', maximum_sentence_len)
    remove_crap('test.txt')
    modify_data_size('test_resize.txt', maximum_sentence_len)
    #train the Word2vec embeddings: dimension = 50
    dimension = 50
    embed_model = WordVec(dimension)
    pkl.dump(embed_model, open('wordvec_model_' + str(dimension) + '.pkl', 'wb'), protocol = 2)
    print ('word to vector done!')
    #add other feature enbeddings to Word2vec: 
    train_file = 'train_resize.txt'
    test_file = 'test_resize.txt'
    trained_model = embed_model
    model_dimension = dimension
    get_input(trained_model, model_dimension, train_file , 'train_embed.pkl', 'train_tag.pkl')
    get_input(trained_model, model_dimension, test_file, 'test_embed.pkl', 'test_tag.pkl')
    #train model
    embword_dim = dimension + 11
    sentence_len = 30
    class_size = 5
    rnn_size = 20
    num_layers = 2
    batch_size = 64
    epoch = 120
    train(batch_size, epoch, embword_dim, sentence_len, class_size, rnn_size, num_layers)




