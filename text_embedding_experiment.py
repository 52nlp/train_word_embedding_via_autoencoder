import pandas as pd
import numpy as np
# !pip install tensorflow==2.3.0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Activation, LSTM, Bidirectional, Dense, RepeatVector, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
from tensorflow.keras import optimizers
print ('earger:', tf.executing_eagerly())


num_words = 20000
maxlen = 100
embed_dim = 50
batch_size = 32


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Word-Embedding', mask_zero=False)
x = emb_layer(encoder_inputs)
state_h = Bidirectional(LSTM(128, activation='relu', name='Encoder-Last-LSTM'))(x)
encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)

decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
decoder_lstm = Bidirectional(LSTM(128, return_sequences=True, name='Decoder-LSTM-before'))
decoder_lstm_output = decoder_lstm(decoded)
decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')
decoder_outputs = decoder_dense(decoder_lstm_output)

seq2seq_Model = Model(encoder_inputs, decoder_outputs)
seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
history = seq2seq_Model.fit(x_train, np.expand_dims(x_train, -1),
          batch_size=batch_size,
          epochs=10)