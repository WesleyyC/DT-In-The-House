import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from scipy.misc import logsumexp

# load ascii text and covert to lowercase
filename = "DT-text.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

# prepare the dataset of input to output pairs encoded as integers
seq_length = 20
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))

# load the network weights
filename = "model-DT.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# helper softmax
def log_softmax(vec):
	return vec - logsumexp(vec)
def softmax(vec):
	return numpy.exp(log_softmax(vec))

# helper for sampling
def sample_i(a, temp=1.0):
	a = numpy.log(a) / temp
	a = softmax(a)
	a /= (1 + 1e-5)
	return numpy.argmax(numpy.random.multinomial(1,a,1))

# pick a random seed
start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
result = ""
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]).strip(), "\""

# generate characters
for i in range(5000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)[0]
	index = sample_i(prediction, 0.5)
	result += int_to_char[index]
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

print result[result.find('.')+1:result.rfind('.')+1]
