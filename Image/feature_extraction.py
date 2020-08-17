from keras import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.metrics import euclidean_distances
import os


class Image_Model:
    def __init__(self):
        self.model = self.model_fun()

    def model_fun(self):
        # load model
        model = VGG16()
        # remove the output layer
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        # summarize the model
        model.summary()

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        return model

    def get_features(self, path):
        image = load_img(path, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        features = self.model.predict(image)
        return features


def main():
    # load an image from file
    model = Image_Model()
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    reference_blackfooted_image = '/dataset/Images\\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg'
    reference_blackfooted_extracted_features = model.get_features(reference_blackfooted_image)

    blackfooted_2_image = path + '/Segmentation2/training/001.Black_footed_Albatross\\Black_Footed_Albatross_0053_796109.jpg'
    blackfooted_2_extracted_features = model.get_features(blackfooted_2_image)
    dist = euclidean_distances(reference_blackfooted_extracted_features, blackfooted_2_extracted_features)
    print("Euclidean distance between reference blackfooted and blackfooted 2: %s" % str(dist))

    blackfooted_image = path + '/Segmentation2/training/001.Black_footed_Albatross\\Black_Footed_Albatross_0050_796125.jpg'
    blackfooted_extracted_features = model.get_features(blackfooted_image)
    dist = euclidean_distances(reference_blackfooted_extracted_features, blackfooted_extracted_features)
    print("Euclidean distance between reference blackfooted and blackfooted: %s" % str(dist))

    groove_image = path + '/Segmentation2/training/004.Groove_billed_Ani\\Groove_Billed_Ani_0071_1559.jpg'
    groove_extracted_features = model.get_features(groove_image)
    dist = euclidean_distances(reference_blackfooted_extracted_features, groove_extracted_features)
    print("Euclidean distance between reference blackfooted and groove tesco: %s" % str(dist))

if __name__ == '__main__':
    main()
