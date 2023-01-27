import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(tokenized_sentence):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)    

# a = "How"

# a = tokenize(a)
# print(a)

# b = ["Organize", "organizer", "organizing"]
# z = [stem(i) for i in b]
# print(z)