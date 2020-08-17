import pandas as pd
import os
import csv
from Image.feature_extraction import Image_Model
from Text.feature_extraction import Text_Model


class Dataset:
    def __init__(self):
        self.text_dir = "dataset/CUBird_WikiArticles"
        self.image_dir = "dataset/Images"
        self.text_model = Text_Model()
        self.img_model = Image_Model()

        self.image_folders = os.listdir(self.image_dir)
        self.column_names = ['index', 'image_feature', 'text_embedding', 'class']
        self.main_dataframe = pd.DataFrame(columns=self.column_names)

    def get_embedding(self, path):
        doc = self.text_model.get_descriptors(path)
        embedding = self.text_model.elmo_vectors(doc)
        return embedding

    def get_features(self, path):
        image_feature = self.img_model.get_features(os.path.join(path))
        return image_feature

    def main(self):
        index = 0
        for text_file in os.listdir(self.text_dir):
            print(os.path.join(self.text_dir, text_file))
            key = text_file.split(".")[1]
            embedding = self.get_embedding(os.path.join(self.text_dir, text_file))

            res = [i for i in self.image_folders if key in i]
            for class_dir in res:
                path = os.path.join(self.image_dir, class_dir)
                for img_file in os.listdir(path):
                    image_feature = self.get_features(os.path.join(path, img_file))
                    df = pd.DataFrame(columns=self.column_names)
                    key = key.replace("_", " ")
					
					# convert the numpy array to string inorder not to loose data
                    img = ""
                    txt = ""
                    for item in image_feature:
                        for i in item:
                            img = img + str(i) + ";"

                    for item in embedding:
                        for i in item:
                            txt = txt + str(i) + " "

                    index += 1
                    df = df.append({"index": index,
                                    "image_feature": img,
                                    "text_embedding": txt,
                                    "class": key}, ignore_index=True)
                    self.main_dataframe = self.main_dataframe.append(df, ignore_index=True)

                print("Main: \n", self.main_dataframe)

        self.main_dataframe.to_csv('dataset.csv', encoding='utf-8')
        return self.main_dataframe


data = Dataset()
print("here")
data.main()