import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
 
from keras.optimizers import RMSprop

from tqdm import tqdm

def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    expPreds = np.exp(preds)
    preds = expPreds / np.sum(expPreds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, diversity):
  startIndex = random.randint(0, len(words) - maxChainLength - 1)
  generated = ''
  sentence = words[startIndex: startIndex + maxChainLength]
  generated = sentence.copy()
  for i in tqdm(range(length), desc="Text generation", ncols=80):
    xPred = np.zeros((1, maxChainLength, len(vocabulary)))
    for t, word in enumerate(sentence):
      xPred[0, t, wordToIndices[word]] = 1.

    preds = model.predict(xPred, verbose = 0)[0]
    nextIndex = sample_index(preds, diversity)
    nextWord = indicesToWords[nextIndex]

    generated.append(nextWord)
    sentence.append(nextWord)
    sentence = sentence[1:]
  return ' '.join(generated)
 
with open('src/input.txt', 'r') as file:
  words = file.read().lower().split()

vocabulary = sorted(list(set(words)))

wordToIndices = dict((word, index) for index, word in enumerate(vocabulary))
indicesToWords = dict((index, word) for index, word in enumerate(vocabulary))

maxChainLength = 10
nextWords = []
sentences = []

for i in range(0, len(words) - maxChainLength, 1):
  sentences.append(words[i:i+maxChainLength])
  nextWords.append(words[i+maxChainLength])

X = np.zeros((len(sentences), maxChainLength, len(vocabulary)), dtype=np.bool)
Y = np.zeros((len(sentences), len(vocabulary)), dtype=np.bool)

for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence):
    X[i, t, wordToIndices[word]] = 1
  Y[i, wordToIndices[nextWords[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(maxChainLength, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(X, Y, batch_size=128, epochs=50)

generatedText = generate_text(1500, 0.2)

with open('result/gen.txt', 'w', encoding='utf-8') as f:
  f.write(generatedText)

print("Generated text saved to file")