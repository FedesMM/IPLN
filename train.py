from os import name
import re
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot, Tokenizer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, Flatten, Dense
from files_manager import load_fasttext_es_300, load_test, load_file, preprocesing
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

'''Recibe una lista de twits y devuelve un arreglo enteros que cuenta cuantas palabras en plurar hay en cada twit de la lista de entrada.'''
def contar_plurales(twits):
	print ("twits.size: "+str(twits.size))
	conteo_plurales=[]
	for twit in twits:	
		plurales= re.compile(r'[o|a]s(?![a-zA-Z0-9])')
		contados=len(re.findall(plurales, str(twit)))
		conteo_plurales+=[contados]
	return conteo_plurales

def prepare_tweets(tweets, tokenizer, max_length):
	# Features antes del preprocesing
	#Feature 1
	plurales=contar_plurales(tweets)
	print ("plurales1: "+str(plurales))

	twits = preprocesing(tweets[:len(tweets), 0])
	# define class labels
	y = tweets[:, 1].astype('float32')

	# integer encode the documents
	encoded_twits = tokenizer.texts_to_sequences(twits)
	x = pad_sequences(encoded_twits, maxlen=max_length, padding='post').astype('float32')
	x = np.concatenate ((x , (np.asmatrix(plurales)).T), axis=1)
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

	twits = [x.split() for x in twits]
	#print(twits)

	# Calculo largo maximo
	mylen = np.vectorize(len)
	lens=mylen(twits)
	max_len=max(lens)
	#TODO: Contar el twtit mas largo
	max_length = max_len
	train_data_x, train_data_y = prepare_tweets(train, t, max_length)
	val_data_x, val_data_y  = prepare_tweets(val, t, max_length)
	#Se agrego la feature plurales 
	max_length+=1
	print("x: "+str(train_data_x.shape)+", y: "+str(train_data_y.shape))
	print("x: "+str(val_data_x.shape)+", y: "+str(val_data_y.shape))

	#calculo features
	#max_length=max_length+1
	
	#new_train_data_x = np.concatenate ((train_data_x , (np.asmatrix(train_data_y)).T), axis=1)
	#new_val_data_x = np.concatenate ((val_data_x , (np.asmatrix(val_data_y)).T), axis=1)	
	#print(new_train_data_x.shape)
	#print(new_val_data_x.shape)

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
	#model.add(Dense(300, activation='sigmoid'))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid', name="Sigmoide"))
	#model.add(Dense(1, activation='relu', name="Relu_1"))
	#model.add(Dense(1, activation='relu', name="Relu_2"))
	#model.add(Dense(100, activation='elu', name="Elu"))
	#model.add(Dense(1, activation='elu', name="densa_3"))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	# summarize the model
	print(model.summary())

	# Set callback functions to early stop training and save the best model so far
	callbacks = [EarlyStopping(monitor='val_accuracy', patience=50),
		ModelCheckpoint(filepath='best_model.h5', monitor='val_accuracy', save_best_only=True)]
	
	print(train_data_x.shape)
	print(train_data_y.shape)
	print(val_data_x.shape)
	print(val_data_y.shape)
	# fit the model
	model.fit(train_data_x, train_data_y, epochs=100, verbose=1, callbacks=callbacks, validation_data=(val_data_x,val_data_y), validation_batch_size=32)
	return model, t, max_length


