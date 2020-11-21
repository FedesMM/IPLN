from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot, Tokenizer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, Flatten, Dense
from files_manager import load_fasttext_es_300, load_test, load_file, preprocesing
import numpy as np


def prepare_tweets(tweets, tokenizer, max_length):
	twits = preprocesing(tweets[:len(tweets), 0])
	# define class labels
	y = tweets[:, 1].astype('float32')
	
	# integer encode the documents
	encoded_twits = tokenizer.texts_to_sequences(twits)
	x = pad_sequences(encoded_twits, maxlen=max_length, padding='post')
	return (x,y)

def train(data_path):
	# define class twits
	train = load_file(data_path +'train.csv')
	val = load_file(data_path + 'val.csv')

	twits = preprocesing(train[:, 0])
	# prepare tokenizer
	t = Tokenizer()
	t.fit_on_texts(twits)
	vocab_size = len(t.word_index) + 1

	# Calculo largo maximo
	mylen = np.vectorize(len)
	lens=mylen(twits)
	max_len=max(lens)
	#TODO: Contar el twtit mas largo
	max_length = max_len
	train_data = prepare_tweets(train, t, max_length)
	val_data = prepare_tweets(val, t, max_length)

	# load the whole embedding into memory
	embeddings_index = dict()
	f = open(data_path + 'fasttext.es.300.txt', encoding="utf8")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))

	# create a weight matrix for words in training docs
	embedding_matrix = np.zeros((vocab_size, 300))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# define model
	model = Sequential()
	e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
	model.add(e)
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	# summarize the model
	print(model.summary())
	# fit the model
	model.fit(train_data[0], train_data[1], epochs=50, verbose=1, validation_data=val_data, validation_batch_size=32)
	return model, t, max_length


