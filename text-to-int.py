import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text


from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D



text = "the hello matt is working on this"
text2 = "the time i spent doing this project"

result = keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
# becomes ['hello', 'matt', 'is', 'working', 'on', 'this']
print(result)

result2 = keras.preprocessing.text.text_to_word_sequence(text2, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
# becomes ['hello', 'matt', 'is', 'working', 'on', 'this']
print(result2)


print(keras.preprocessing.text.hashing_trick(text, 10000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~  ', lower=True, split=' '))
print(keras.preprocessing.text.hashing_trick(text2, 10000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~  ', lower=True, split=' '))
