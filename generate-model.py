import numpy
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load data and simple preprocess
filename = "DT-text.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping between char-int
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

pickle.dump(char_to_int, open("index/char_to_int.json", "w"))
pickle.dump(int_to_char, open("index/int_to_char.json", "w"))

# stat
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Text Length: ", n_chars
print "Total Vocab Number: ", n_vocab

# build the dataset
seq_length = 20
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])\

# stat
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns

# reshape input to be [samples, seq length/unroll step, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the stack LSTM
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="model-tmp/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, nb_epoch=60, batch_size=64, callbacks=callbacks_list)
