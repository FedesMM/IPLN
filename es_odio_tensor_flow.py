import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

'''
print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
'''
def es_odio_TF(x_train, y_train, x_val, y_val, x_test, y_test, model_file_name):
    ''' Los datos tienen que estar como numpys'''

    model_file = model_file_name #"fasttext.es.300.txt"
    hub_layer = hub.KerasLayer(model_file, output_shape=[20], input_shape=[],
                               dtype=tf.string, trainable=True)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    print(model.summary())

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])



    history = model.fit(x_train,
                        y_train,
                        epochs=30,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    results = model.evaluate(x_test, y_test)
    print(f"results: {results}")

    history_dict = history.history
    print(f"history_dict.keys(): {history_dict.keys()}")

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()   # clear figure
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()