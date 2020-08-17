import string
import nltk

nltk.download('averaged_perceptron_tagger')

# $ pip install "tensorflow>=1.7.0"
# $ pip install tensorflow-hub
# pip install --upgrade tensorflow-gpu
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf


class Text_Model:
    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

    def elmo_vectors(self, x):
        embeddings = self.elmo([x], signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1))

    def get_descriptors(self, filename):
        descriptors = list()
        str = ""
        file = open(filename, 'r', encoding="utf8")
        docu = file.read()
        for line in docu.split("\n"):
            # prepare translation table for removing punctuation
            table = str.maketrans('', '', string.punctuation)
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # removing punctuation from each token
            line = [word.translate(table) for word in line]
            line = line[1:]
            line = [x for x in line if x]
            # remove hanging 's' and 'a'
            line = [word for word in line if len(word) > 1]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]

            if len(line) > 0:
                descriptors.append(' '.join(line))
                str = str + descriptors[len(descriptors) - 1]

        return str
