import sys
import numpy
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from scipy.misc import logsumexp

# load the mapping
char_to_int = pickle.load(open("char_to_int.json", "r"))
int_to_char  = pickle.load(open("int_to_char.json", "r"))
seq_length = 20

# redefine the stack LSTM
model = Sequential()
model.add(LSTM(512, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(len(char_to_int), activation='softmax'))

# load the model parameter
filename = "model-DT.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# helper softmax to make sure sum (a)<1
def log_softmax(vec):
	return vec - logsumexp(vec)
def softmax(vec):
	return numpy.exp(log_softmax(vec))

# helper for sampling with multinomial
def sample_i(a, temp=1.0):
	a = numpy.log(a) / temp
	a = softmax(a)
	a /= (1 + 1e-5) # make sure sum (a)<1
	return numpy.argmax(numpy.random.multinomial(1,a,1))

# setup the seed and result
seed_text = "i have the best temperament"
seed_text = seed_text[0:seq_length]
pattern = [char_to_int[char] for char in seed_text]
result = ""

# generate characters
for i in range(5000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(char_to_int))
	prediction = model.predict(x, verbose=0)[0]
	index = sample_i(prediction, 0.5)
	result += int_to_char[index]
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print result[result.find('.')+1:result.rfind('.')+1]
