import files_manager
import train

def evaluate(model, t, max_length, path, csv):

	test = files_manager.load_file(path + csv)
	test_data = train.prepare_tweets(test, t, max_length)

	# evaluate the model
	loss, accuracy = model.evaluate(x=test_data[0], y=test_data[1], verbose=0)
	print('Accuracy '+ csv +': %f' % (accuracy * 100))
	predicted = model.predict(test_data[0])
	files_manager.save_file(csv, predicted)
	#print(predicted)