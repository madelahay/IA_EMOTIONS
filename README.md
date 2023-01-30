# **Projet de fin d'année : Intelligence Artificielle** 

-> Un notebook Jupyter est disponible dans le dossier "notebook" du projet pour tester celui-ci.

### **Sujet : (TP DEEP)**
- choisir un autre problème de classification et entraîner un réseau de neurones simple,
- reprendre celui sur les chiffres et étudier l'impact d'un jeu de données « faussé »   
  en apprentissage (suivant le taux d'inversion entre le chiffre 1 et 7 par exemple).

### **Ce que l'on à choisi de faire :**

Nous avons choisi de nous baser sur un dataset déjà présent dans keras, le dataset IMDB.

Celui ci contient beaucoup de types de données différentes, mais celles qui vont nous intéresser vont être :  
- Le contenu des commentaires d'un film
- L'aspect positif ou negatif du commentaire

Le modèle Kéras que nous allons créer vas avoir pour tâche de récupérer le contenu d'un commentaire sous un film,   
établire les occurences de chaque mots dans le commentaire, et en fonction du nombre d'occurences, déterminer  
si le commentaire est plutôt positif ou négatif.  

Les "commentaires" en eux même sont des array d'entiers, qui correspondent en fait aux positions de mots dans le dictionnaire  
IMDB qui fait correspondre à un index un mot. Donc un commentaire du type "Film génial" pourrait etre représenté sous la forme list([4, 8])  
(c'est un exemple, les indices ne correspondent pas).  

Le jeu d'entrainement est donc constitué de commentaires et de leur caractère (positif / negatif soit 0 ou 1), et le jeu de   
test est constitué de commentaires également, le modèle ayant pour charge de déterminer leur caractère.  

## **Introduction à LSTM et imports :**

```python
Nous avons choisi d'utiliser des couches LSTM, proposées par Keras. 
Voici un extrait de la documentation associée pour précision :

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
```

## **Variables globales du projet :**

```python
# nombre de commentaires à utiliser pour l'apprentissage
max_features = 20000
# nombre de mots à utiliser pour l'apprentissage (coupé après 150)
maxlen = 250 
batch_size = 32
```

## **Chargement des données à utiliser dans le modèle :**

```python

print('----- START LOADING DATA -----')
# recuperation des données
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# verifier la taille des données
print("LENGTH OF TRAINING DATA: ", len(x_train))
print("LENGTH OF TEST DATA: ", len(x_test))


print('----- END LOADING DATA -----')
```

imdb.load_data permet de charger les tableaux contenant les indices des mots des commentaires.   
La fonction retourne donc deux tuples, correspondant aux jeux de données d'entrainement et de test.  

Le paramètre num_words est en fait le nombre de commentaires que l'on vas récupérer, soit la taille du dataset.

## **On prépare les données pour le traitement :**


```python
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
```
Ici on coupe en effet la longueur de tous les commentaires, pour accélerer la vitesse de traitement.

En effet, nous avons remarqué au cours de nos essais que garder 500 mots du commentaire avec notre modèle pouvait  
entrainer des durées d'apprentissage allant au dela de 6min30 par epochs, ce que nous trouvions trop long pour un  
petit projet. C'est pourquoi on coupe les commentaires à 250 mots de longs, n'impactant que très peu la précision   
du modèle.

## **Construction du modèle :**

```python
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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

print('----- END BUILDING MODEL -----')
```

## **Apprentissage et tests du modèle :**

Après avoir fait tourner ce modèle, notre meilleur score était approximativement de 87% de reconnaissance sur un jeu de données de test  
inconnues du réseau de neuronnes, ce que nous avons trouvé très satisfaisant au vu du caractère assez subjectif d'un commentaire et d'une émotion humaine.

```python
print('----- START FITTING MODEL -----')
# on entraine le modele sur les données d'apprentissage
# au vu de la taille des données, on utilise un batch_size de 32
# et on fait une epoque de 2 epochs
model.fit(x_train, y_train, batch_size=batch_size, epochs=2, validation_data=(x_test, y_test))
# on détermine le score du modele sur les données de test
score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
print('----- END FITTING MODEL -----')

# on affiche le score du modele
print('Test score:', score)
print('Test accuracy:', acc)

```

Nous avons décidé donc de n'utiliser que 2 époque pour faire tourner le modèle, une époque prenant déjà 1min30,   
et le modèle convergeant déjà très vite, faire plus d'époques n'améliore pas vraiment l'accurary du modèle

### **Tests sur un couche Dense :**

Nous avons également essayé de faire tourner ce modèle avec des couches Dense, mais la qualité de la reconnaissance était  
bien moindre, d'ou l'utilisation des couches LSTM, plus faciles également à faire tourner sur n'importe quelle machine étant  
donné qu'elles sont basées sur la puissance de calcul de la machine locale. (voir notebook testingDense.ipynb)

En effet, un modèle Dense converge très rapidement, mais sur une reconnaissance à 50% environ, soit une chance sur deux, donc  
aléatoire étant donné qu'il n'y à que deux états de sortie possible. En plus de ça, la loss était très élevée.