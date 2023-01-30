'''
Long Short-Term Memory layer - Hochreiter 1997.

See the Keras RNN API guide for details about the usage of RNN API.

Based on available runtime hardware and constraints, this 
layer will choose different implementations (cuDNN-based or pure-TensorFlow) 
to maximize the performance. If a GPU is available and all the arguments to the 
layer meet the requirement of the cuDNN kernel (see below for details), the layer 
will use a fast cuDNN implementation.

From : https://keras.io/api/layers/recurrent_layers/lstm/
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# nombre de commentaires à utiliser pour l'apprentissage
max_features = 20000
# nombre de mots à utiliser pour l'apprentissage (coupé après 150)
maxlen = 250 
batch_size = 32

print('----- START LOADING DATA -----')
# recuperation des données
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# verifier la taille des données
print("LENGTH OF TRAINING DATA: ", len(x_train))
print("LENGTH OF TEST DATA: ", len(x_test))


print('----- END LOADING DATA -----')

"""
What is a pad sequence?

This function transforms a list (of length num_samples) 
of sequences (lists of integers) into a 2D Numpy array of 
shape (num_samples, num_timesteps). num_timesteps is either 
the maxlen argument if provided, or the length of the longest 
sequence in the list.
"""
print('----- START PADDING DATA -----')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('----- END PADDING DATA -----')

print('----- START BUILDING MODEL -----')
# Construction du modele
model = Sequential()
# on utilise une couche d'embedding pour convertir les mots en vecteurs
model.add(Embedding(max_features, 12))
# on utilise une couche LSTM pour la representation (cf supra pour les explications)
"""
Extrait de documentation Keras : 
Dropout : Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs. Default: 0. 
Reccurent Dropout : Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state. Default: 0. 
"""
model.add(LSTM(12, dropout=0.2, recurrent_dropout=0.2))
# on utilise une couche de sortie pour la classification
model.add(Dense(1, activation='sigmoid'))

# on compile le modele en utilisant l'optimiseur 'rmsprop' et la fonction de perte 'binary_crossentropy'
# la metrics est la fonction de perte qui est utilisée pour calculer l'erreur
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('----- END BUILDING MODEL -----')

print('----- START FITTING MODEL -----')
# on entraine le modele sur les données d'apprentissage
# au vu de la taille des données, on utilise un batch_size de 32
# et on fait une epoque de 2 epochs
model.fit(x_train, y_train, batch_size=batch_size, epochs=4, validation_data=(x_test, y_test))
# on détermine le score du modele sur les données de test
score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
print('----- END FITTING MODEL -----')

# on affiche le score du modele
print('Test score:', score)
print('Test accuracy:', acc)
