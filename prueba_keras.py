# define documents
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot, Tokenizer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, Flatten, Dense
from files_manager import load_fasttext_es_300, load_test, preprocesing
import numpy as np

def prueba_1():
	docs = ['Well done!',
			'Good work',
			'Great effort',
			'nice work',
			'Excellent!',
			'Weak',
			'Poor effort!',
			'not good',
			'poor work',
			'Could have done better.']
	# define class labels
	labels = np.array([1,1,1,1,1,0,0,0,0,0])

	# integer encode the documents
	vocab_size = 50
	encoded_docs = [one_hot(d, vocab_size) for d in docs]
	print(encoded_docs)

	# pad documents to a max length of 4 words
	max_length = 4
	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	print(padded_docs)

	# define the model
	model = Sequential()
	model.add(Embedding(vocab_size, 8, input_length=max_length))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	# summarize the model
	print(model.summary())

	# fit the model
	model.fit(padded_docs, labels, epochs=50, verbose=0)
	# evaluate the model
	loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

def prueba_2():
	cantidad_twits=10
	# define class twits
	test = load_test()
	twits = preprocesing(test[:cantidad_twits, 0])
	print(f"\ntwiters:\n{twits}")
	# define class labels
	labels = test[:cantidad_twits, 1].astype('float32')
	print(f"\nlabels:\n{labels}")
	# prepare tokenizer
	t = Tokenizer()
	t.fit_on_texts(twits)
	vocab_size = len(t.word_index) + 1
	# integer encode the documents
	encoded_twits = t.texts_to_sequences(twits)
	print(f"\nencoded_twits:\n{encoded_twits}")
	# pad documents to a max length of 4 words
	#TODO: Contar el twtit mas largo
	max_length = 140
	padded_twits = pad_sequences(encoded_twits, maxlen=max_length, padding='post')
	print(f"\npadded_twits:\n{padded_twits}")

	# load the whole embedding into memory
	embeddings_index = dict()
	f = open('fasttext.es.300.txt')
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
	model.fit(padded_twits, labels, epochs=50, verbose=0)
	# evaluate the model
	loss, accuracy = model.evaluate(padded_twits, labels, verbose=0)
	print('Accuracy: %f' % (accuracy * 100))

prueba_2()

