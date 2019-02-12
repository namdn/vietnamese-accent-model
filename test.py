from __future__ import print_function, division

import itertools
import os
import random
import re
import string
import glob

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# for reproducibility
random.seed(1337)
np.random.seed(1337)


# maximum string length to train and predict
# this is set based on our ngram length break down below
MAXLEN = 32

# minimum string length to consider
MINLEN = 3

# how many words per ngram to consider in our model
NGRAM = 5

# inverting the input generally help with accuracy
INVERT = True

# mini batch size
BATCH_SIZE = 128

# number of phrases set apart from training set to validate our model
# VALIDATION_SIZE = 100000
VALIDATION_SIZE = 100

# using g2.2xl GPU is ~5x faster than a Macbook Pro Core i5 CPU
HAS_GPU = False


PATH = 'D:/Projects/chatbot/get-data/cafef.vn/nganhang'

def read_data(path = PATH, max_file = None):
    
    texts = []
    filenames = glob.glob(f"{path}/*.chn")
    random.shuffle(filenames)
    max_file = max_file or len(filenames)
    
    for filename in filenames[:max_file]:
        print(filename)
        with open(filename, 'r', encoding='utf8') as f:
            text = f.read()
            texts.append(text)

    return texts

print('reading file...')
texts = read_data(max_file=100)

def extract_phrases(text):
    """ extract phrases, i.e. group of continuous words, from text """
    return re.findall(r'\w[\w ]+', text, re.UNICODE)


phrases = itertools.chain.from_iterable(extract_phrases(text) for text in texts)
phrases = [p.lower().strip() for p in phrases]
[print(s) for s in phrases[:5]]
print('Number of phrases:', len(phrases))



def gen_ngrams(words, n=3):
    """ gen n-grams from given phrase or list of words """
    if isinstance(words, str):
        words = re.split('\s+', words.strip())
    
    if len(words) < n:
        padded_words = words + ['\x00'] * (n - len(words))
        yield tuple(padded_words)
    else:
        for i in range(len(words) - n + 1):
            yield tuple(words[i: i+n])

ngrams = itertools.chain.from_iterable(gen_ngrams(p, NGRAM) for p in phrases)
ngrams = list(set(' '.join(t) for t in set(ngrams)))


# [print(t) for t in ngrams[:5]]
# print('Number of {}-gram: {}'.format(NGRAM, len(ngrams)))
# pd.DataFrame([len(ngram) for ngram in ngrams]).hist(bins=20)
# plt.show()



accented_chars = {
    'a': u'a á à ả ã ạ â ấ ầ ẩ ẫ ậ ă ắ ằ ẳ ẵ ặ',
    'o': u'o ó ò ỏ õ ọ ô ố ồ ổ ỗ ộ ơ ớ ờ ở ỡ ợ',
    'e': u'e é è ẻ ẽ ẹ ê ế ề ể ễ ệ',
    'u': u'u ú ù ủ ũ ụ ư ứ ừ ử ữ ự',
    'i': u'i í ì ỉ ĩ ị',
    'y': u'y ý ỳ ỷ ỹ ỵ',
    'd': u'd đ',
}

plain_char_map = {}
for c, variants in accented_chars.items():
    for v in variants.split(' '):
        plain_char_map[v] = c


def remove_accent(text):
    return u''.join(plain_char_map.get(char, char) for char in text)

# print(remove_accent(u'cô gái đến từ hôm qua'))


class CharacterCodec(object):
    def __init__(self, alphabet, maxlen):
        self.alphabet = list(sorted(set(alphabet)))
        self.index_alphabet = dict((c, i) for i, c in enumerate(self.alphabet))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.alphabet)))
        for i, c in enumerate(C[:maxlen]):
            X[i, self.index_alphabet[c]] = 1
        return X
    
    def try_encode(self, C, maxlen=None):
        try:
            return self.encode(C, maxlen)
        except KeyError:
            return None

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.alphabet[x] for x in X)


#\x00 is the padding characters
alphabet = set('\x00 _' + string.ascii_lowercase + string.digits + ''.join(plain_char_map.keys()))
codec = CharacterCodec(alphabet, MAXLEN)

# print(codec.encode(u'cô gái đến từ hôm qua'))
# print(codec.decode(codec.encode(u'cô gái đến từ hôm qua')).replace('\x00', '#'))



np.random.shuffle(ngrams)
train_size = len(ngrams) - VALIDATION_SIZE
train_set = ngrams[:train_size]
validation_set = ngrams[train_size:]

print('train size: {}'.format(len(train_set)))
print('validation size: {}'.format(len(validation_set)))


def gen_batch(it, size):
    """ batch the input iterator to iterator of list of given size"""
    for _, group in itertools.groupby(enumerate(it), lambda x: x[0] // size):
        yield list(zip(*group))[1]


def gen_stream(ngrams):
    """ generate an infinite stream of (input, output) pair from phrases """
    while True:
        for s in ngrams:
            output_s = s + '\x00' * (MAXLEN - len(s))
            input_s = remove_accent(output_s)    
            input_s = input_s[::-1] if INVERT else input_s
            input_vec = codec.try_encode(input_s)
            output_vec = codec.try_encode(output_s)
            if input_vec is not None and output_vec is not None:
                yield input_vec, output_vec


def gen_data(ngrams, batch_size=128):
    """ generate infinite X, Y array of batch_size from given phrases """
    for batch in gen_batch(gen_stream(ngrams), size=batch_size):
        # we need to return X, Y array from one batch, which is a list of (x, y) pair
        X, Y = zip(*batch)
        yield np.array(X), np.array(Y)


# next(iter(gen_data(train_set, 1)))


from keras.models import Sequential
# from keras.engine.training import slice_X
# from keras.engine.training import _slice_arrays as slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.callbacks import Callback


RNN = recurrent.LSTM
HIDDEN_SIZE = 256

rnn_consume_less = 'gpu' if HAS_GPU else 'cpu'

model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(alphabet)), consume_less=rnn_consume_less))
model.add(RepeatVector(MAXLEN))
model.add(RNN(HIDDEN_SIZE, return_sequences=True, consume_less=rnn_consume_less))
model.add(RNN(HIDDEN_SIZE, return_sequences=True, consume_less=rnn_consume_less))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


from collections import defaultdict


class EverHistory(Callback):
    """ A Keras History that isn't cleared upon training begin and know how to plot its loss and accuracy history """
    
    def __init__(self):
        self.epoch = []
        self.history = defaultdict(list)
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(len(self.epoch))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def _plot(self, name, metric):
        legend = [metric]
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch, self.history[metric], marker='.')
        
        val_metric = 'val_' + metric
        if val_metric in self.history:
            legend.append(val_metric)
            plt.plot(self.epoch, self.history[val_metric], marker='.')

        plt.title(name + ' over epochs')
        plt.xlabel('Epochs')
        plt.legend(legend, loc='best')
        plt.show()
        
    def plot_loss(self):
        self._plot('Loss', 'loss')
        
    def plot_accuracy(self):
        self._plot('Accuracy', 'acc')


history = EverHistory()





from collections import Counter


def guess(ngram):
    text = ' '.join(ngram)
    text += '\x00' * (MAXLEN - len(text))
    if INVERT:
        text = text[::-1]
    preds = model.predict_classes(np.array([codec.encode(text)]), verbose=0)
    return codec.decode(preds[0], calc_argmax=False).strip('\x00')


def add_accent(text):
    ngrams = list(gen_ngrams(text.lower(), n=NGRAM))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])

    output = ' '.join(c.most_common(1)[0][0] for c in candidates if len(c.most_common(1)))
    return output

train_generator = gen_data(train_set, batch_size=BATCH_SIZE)
validation_generator = gen_data(validation_set, batch_size=BATCH_SIZE)

# model.fit_generator(train_generator, samples_per_epoch=128000, nb_epoch=1,
#                     validation_data=validation_generator, nb_val_samples=12800,
#                     callbacks=[history])

# print(u'"{}"'.format(add_accent('co gai den tu hom qua')))

from IPython import display


test = 'co gai den tu hom qua'
correct = u'cô gái đến từ hôm qua'
outputs = []
for i in range(50):
    display.clear_output(wait=True)
    history.plot_loss()
    history.plot_accuracy()
    outputs.append(add_accent('co gai den tu hom qua'))
    for it, out in enumerate(outputs):
        is_correct = u'✅' if out.strip('#') == correct else u'❌'
        print(u'epoch {:>2}: "{}" {}'.format(it, out, is_correct))  
    
    # model.fit_generator(train_generator, samples_per_epoch=12800, nb_epoch=1,
    #                     validation_data=validation_generator, nb_val_samples=1280,
    #                     callbacks=[history])
    model.fit_generator(train_generator, samples_per_epoch=128, nb_epoch=1,
                        validation_data=validation_generator, nb_val_samples=12,
                        callbacks=[history])


