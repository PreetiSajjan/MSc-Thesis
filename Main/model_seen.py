import numpy as np
from keras import Input
from sklearn.model_selection import train_test_split

np.random.seed(123)
import os
from keras.utils.vis_utils import plot_model

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KDTree
import pandas as pd
from keras.layers.merge import add
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical, np_utils

WORD2VECPATH = "new_class_vectors.npy"
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
column_names = ['index', 'image_feature', 'text_embedding', 'class']
DATAPATH = os.path.join(parent_dir, "finaldataset.csv")
MODELPATH = os.path.join(parent_dir, "Model/")  # "/model/"

def load_keras_model(model_path):
    with open(model_path +"model.json", 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"model.h5")
    return loaded_model


def save_keras_model(model, model_path):
    """save Keras model and its weights"""
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_json = model.to_json()
    with open(model_path + "model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_path + "model.h5")
    print("-> zsl model is saved.")
    return


def get_feature(s, sl):
    token = s.split(sl)
    token = token[:-1]
    token = [float(i) for i in token]
    array = np.array(token)
    return array


def get_dataframe():
    data = pd.read_csv(DATAPATH, names=column_names, header=None, index_col=[0], skiprows=1)
    global train_df, zsl_df
    train_df = pd.DataFrame(columns=column_names[1:])
    zsl_df = pd.DataFrame(columns=column_names[1:])

    for ind in data.index:
        df = pd.DataFrame(columns=column_names[1:])
        img = get_feature(data['image_feature'][ind], ";")
        txt = get_feature(data['text_embedding'][ind], " ")
        key = data['class'][ind]
        df = df.append({"image_feature": img,
                        "text_embedding": txt,
                        "class": key}, ignore_index=True)
        if key in train_classes:
            train_df = train_df.append(df, ignore_index=True)
        elif key in zsl_classes:
            zsl_df = zsl_df.append(df, ignore_index=True)
        del df

    print(len(train_df), train_df.head())
    print(len(zsl_df), zsl_df.head())
    np.random.shuffle(train_df.values)
    np.random.shuffle(zsl_df.values)


def load_data():
    get_dataframe()
    #X_train, Y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
    #X_test, Y_test = zsl_df.iloc[:, :-1].values, zsl_df.iloc[:, -1].values

    X, Y = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    # L2 NORMALIZE X_TRAIN
    # X_train = normalize(X_train, norm='l2')
    # X_test = normalize(X_test, norm='l2')

    label_encoder = LabelEncoder()
    label_encoder.fit(train_classes)

    # training encoding
    encoded_Y = label_encoder.transform(Y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_train_cat = np_utils.to_categorical(encoded_Y)

    # ---------------------------------- Training Placeholders ------------------------------------------------

    train_image_feature = np.zeros((len(X_train), 4096), dtype='float32')
    train_text_feature = np.zeros((len(X_train), 1024), dtype='float32')
    for index, (img, text) in enumerate(X_train):
        train_image_feature[index] = img
        train_text_feature[index] = text

    #print(train_image_feature.shape, train_image_feature, "\n", train_text_feature.shape, train_text_feature)

    # ---------------------------------- Testing Placeholders ------------------------------------------------

    test_image_feature = np.zeros((len(X_test), 4096), dtype='float32')
    test_text_feature = np.zeros((len(X_test), 1024), dtype='float32')
    for index, (img1, text1) in enumerate(X_test):
        test_image_feature[index] = img1
        test_text_feature[index] = text1

    return (train_image_feature, train_text_feature, Y_train_cat), (test_image_feature, test_text_feature, Y_test)


def custom_kernel_init(shape, dtype=None):
    class_vectors = np.load(WORD2VECPATH, allow_pickle=True)
    training_vectors = sorted([(label, vec) for (label, vec) in class_vectors if label in train_classes],
                              key=lambda x: x[0])
    classnames, vectors = zip(*training_vectors)
    print("\nClass name: ", classnames)
    vectors = np.asarray(vectors, dtype=np.float)
    vectors = vectors.T
    return vectors


def build_model():
    inputs1 = Input(shape=(4096,))
    batch1 = BatchNormalization()(inputs1)
    dr1 = Dropout(0.5)(batch1)
    fe1 = Dense(2048, activation='relu')(dr1)
    dr2 = Dropout(0.6)(fe1)
    fe2 = Dense(1500, activation='relu')(dr2)
    dr3 = Dropout(0.2)(fe2)
    fe3 = Dense(2000, activation='relu')(dr3)
    dr4 = Dropout(0.4)(fe3)
    fe4 = Dense(1500, activation='relu')(dr4)
    dr5 = Dropout(0.3)(fe4)
    fe5 = Dense(1024, activation='softmax')(dr5)

    inputs2 = Input(shape=(1024,))
    decoder1 = add([fe5, inputs2])
    #batch2 = BatchNormalization()(decoder1)
    #dr6 = Dropout(0.8)(batch2)
    decoder2 = Dense(800, activation='relu')(decoder1)
    dr7 = Dropout(0.5)(decoder2)
    decoder3 = Dense(512, activation='relu')(dr7)
    dr8 = Dropout(0.6)(decoder3)
    decoder4 = Dense(512, activation='relu')(dr8)
    dr9 = Dropout(0.4)(decoder4)
    decoder5 = Dense(400, activation='relu')(dr9)
    dr10 = Dropout(0.5)(decoder5)
    decoder6 = Dense(NUM_ATTR, activation='relu')(dr10)
    outputs = Dense(NUM_CLASS, activation='softmax', trainable=False, kernel_initializer=custom_kernel_init)(decoder6) #False
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    adam = Adam(lr=5e-5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])

    print("\n-----------------------model building is completed.-----------------------")
    model.summary()
    return model


def main():
    global train_classes
    with open('train_classes.txt', 'r') as infile:
        train_classes = [str.strip(line) for line in infile]

    global zsl_classes
    with open('zsl_classes.txt', 'r') as infile:
        zsl_classes = [str.strip(line) for line in infile]

    # SET HYPERPARAMETERS
    global NUM_CLASS, NUM_ATTR, EPOCH, BATCH_SIZE
    NUM_CLASS = 171 #156
    NUM_ATTR = 300
    BATCH_SIZE = 512
    EPOCH = 180

    # TRAINING PHASE

    (train_image_feature, train_text_feature, Y_train_cat), (
        test_image_feature, test_text_feature, Y_test) = load_data()
    model = build_model()
    #plot_model(model, to_file='train_model_plot.png', show_shapes=True, show_layer_names=True)

    history = model.fit([train_image_feature, train_text_feature], Y_train_cat,
                        epochs=EPOCH,
                        batch_size=BATCH_SIZE,
                        validation_split=0.3,
                        shuffle=True)

    print("\n-----------------------model training is completed.-----------------------")

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # CREATE AND SAVE ZSL MODEL

    inp = model.input
    out = model.layers[-2].output
    zsl_model = Model(inp, out)
    save_keras_model(zsl_model, model_path=MODELPATH)
    #plot_model(zsl_model, to_file='zsl_model_plot.png', show_shapes=True, show_layer_names=True)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # EVALUATION OF ZERO-SHOT LEARNING PERFORMANCE

    class_vectors = sorted(np.load(WORD2VECPATH, allow_pickle=True), key=lambda x: x[0])
    classnames, vectors = zip(*class_vectors)
    classnames = list(classnames)
    vectors = np.asarray(vectors, dtype=np.float)

    tree = KDTree(vectors, )
    print("\n-----------------------zero shot model building is completed.-----------------------")
    zsl_model.summary()
    pred_zsl = zsl_model.predict([test_image_feature, test_text_feature])
    print("\n\n**************prediction*******************\n", pred_zsl.shape, pred_zsl)
    top10, top5, top3, top1 = 0, 0, 0, 0
    for i, pred in enumerate(pred_zsl):
        pred = np.expand_dims(pred, axis=0)
        dist_5, index_5 = tree.query(pred, k=10)
        pred_labels = [classnames[index] for index in index_5[0]]
        true_label = Y_test[i]
        print("pred_labels: ", pred_labels, "\nActual:", true_label)
        if true_label in pred_labels:
            top10 += 1
        if true_label in pred_labels[:5]:
            top5 += 1
        if true_label in pred_labels[:3]:
            top3 += 1
        if true_label in pred_labels[0]:
            top1 += 1

    print()
    print("ZERO SHOT LEARNING SCORE")
    print("-> Top-10 Accuracy: %.2f" % (top10 / float(len(test_text_feature))))
    print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(test_text_feature))))
    print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(test_text_feature))))
    print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(test_text_feature))))
    return


if __name__ == '__main__':
    main()
