#https://www.tensorflow.org/tutorials/keras/basic_text_classification
import tensorflow as tf
from tensorflow import keras

import numpy as np
print(tf.__version__)
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#25000 of each
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# first review in array of reviews
print(train_data[0])
print(train_data[1])
print(train_data[2])

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#returns the decoded text
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(train_data[0])
decode_review(train_data[1])
decode_review(train_data[2])
#make array elements the same length so they can be fed into neural network
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(train_data[0])

# The first layer is an Embedding layer.
#This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index.
#These vectors are learned as the model trains.
#The vectors add a dimension to the output array.
#The resulting dimensions are: (batch, sequence, embedding)

#Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension.
#This allows the model to handle input of variable length, in the simplest way possible.

#3rd layer This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.

#The last layer is densely connected with a single output node.
#Using the sigmoid activation function, this value is a float between 0 and 1, representing a probability, or confidence level.

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
#build model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#configure model to use optimizer and loss function to reduce loss
#binary classifaction
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# create a validation set with training data to increase accuracy
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train the model for 40 epochs in mini batches of 512 samples
# 40 iterations over all samples in the x_train and y_train tensors
#returns a history object that contains everything that happened in training
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=25,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# returns Loss, accuracy
results = model.evaluate(test_data, test_labels)

print(results)


#Create Graph of accuracy and loss over time
#look at training with history object
history_dict = history.history
history_dict.keys()

# four metrics watched during training and validation
#dict_keys(['loss', 'val_loss', 'val_acc', 'acc'])

#create plot
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

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
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



#
