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

print(sentences)