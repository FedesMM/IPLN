import files_manager
import train
import numpy as np

def evaluate(model, t, max_length, path, csv):
	print ("max_length: "+str(max_length))
	test = files_manager.load_file(path + csv)
	test_data_x, test_data_y = train.prepare_tweets(test, t, max_length-1)
	#max_length+=1
	#y=(np.asmatrix(test_data_y)).T
	#test_data_x = np.concatenate ((test_data[0] , (np.asmatrix(test_data[1])).T), axis=1)	
	print(test_data_x.shape)
	#print(y.shape)

	# evaluate the model
	loss, accuracy = model.evaluate(x=test_data_x, y=test_data_y, verbose=0)
	print('Accuracy '+ csv +': %f' % (accuracy * 100))
	predicted = model.predict(test_data_x)
	files_manager.save_file(csv, predicted)
	#print(predicted)