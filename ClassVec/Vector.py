from gensim.models import word2vec
import numpy as np
import gensim
import os

class Word2vec():

    def __init__(self):
        # if using from different directory
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        #self.model_path = os.path.join(parent_dir, "ClassVec//GoogleNews-vectors-negative300.bin")

        # if using from same directionary
        self.model_path = "GoogleNews-vectors-negative300.bin"
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        self.vocab = self.model.vocab.keys()

    def generate_word2vec(self, words):
        words = words.split(' ')
        print('Number of words  : ' + str(len(words)))
        vectors = list()
        for word in words:
            if word in self.vocab:
                vec = self.model[word]
                vectors.append(vec)
            else:
                print("Word {} not in vocab".format(word))
                vectors.append(np.zeros((300,)))
                continue
        if len(words) <= 1:
            return vectors[0]

        return sum(vectors)/len(words)

# create Label Embeddings
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path = os.path.join(parent_dir, "dataset.csv")
column_names = ['index', 'image_feature', 'text_embedding', 'class']
data = pd.read_csv(path, names=column_names, header=None, index_col=[0], skiprows=1)
print(data.head())
print(data.dtypes)

vector = Word2vec()
array1 = np.empty(shape=(0, 2))
d = set(data['class'])
print(len(d), type(d))
with open(os.path.join(parent_dir, 'Main/train_classes.txt'), 'r') as infile:
    train_classes = [str.strip(line) for line in infile]
with open(os.path.join(parent_dir, 'Main/zsl_classes.txt'), 'r') as infile:
    zsl_classes = [str.strip(line) for line in infile]
for c in d:
    if c in train_classes or c in zsl_classes:
        vec = vector.generate_word2vec(c)
        array1 = np.append(array1, np.array([[c, vec]]), axis=0)
    else:
        print("Class not in either list: ", c)

print(array1.shape, array1[0:5])
np.save('new_class_vectors', array1)

exit(1)
