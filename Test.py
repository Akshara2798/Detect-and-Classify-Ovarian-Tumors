import numpy as np
import pandas as pd
import cv2
from Preprocess import *
from Segmentation import *
from Optimization import *
from Existing_Class import *
from Classification import *
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from tkinter import Tk
from Performance import *
from tkinter.filedialog import askopenfilename
from tensorflow.keras.utils import custom_object_scope

class MedicalImageProcessor:
    def __init__(self, model_path="Models/Proposed_model", seg_model_path="Segmentation Model/proposed_seg_model.h5"):
        self.model_path = model_path
        self.seg_model_path = seg_model_path
        self.custom_objects = {"HybridGGO_DBO": HybridGGO_DBO}
        self.image = None
        self.filtered_image = None
        self.segmented_image = None
        self.predicted_class = None
        self.classes = [
            'Chocolate Cyst', 'Serous Cystadenoma', 'Teratoma', 'Normal Ovary', 
            'Theca Cell Tumors', 'Simple Cyst', 'Mucinous Cystadenoma', 
            'High Grade Serous Carcinoma'
        ]
    
    def select_image(self):
        print("Select Input  .....")
        Tk().withdraw()  # Hide Tkinter window
        image_path = askopenfilename(initialdir='OTU_2d-20241021T080321Z-001/OTU_2d/images')
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (256, 256))
    
    def preprocess_image(self):
        print("\nPreprocessing ....\n")
        # Anisotropic Speckle Reducing Diffusion filter
        self.filtered_image = anisotropic_speckle_reducing_filter(self.image) / 255
        cv2.imshow("Original Image", self.image)
        cv2.imshow("Preprocessed Image", self.filtered_image)
        cv2.imwrite("Results/Original_Image.png", self.image)
        cv2.imwrite("Results/Filtered_Image.png",self.filtered_image)

    def segment_image(self):
        print("Segmentation ....\n")
        # Progressive attention attached Dual embedding layer Vision transformer (PA-DViT)
        with custom_object_scope(self.custom_objects):
            seg_model = load_model(self.seg_model_path)
        predicted_image = seg_model.predict(self.filtered_image.reshape(1, 256, 256, 3))
        print(predicted_image.shape)
        print(np.unique(predicted_image))
        
        predicted_image = np.where(predicted_image > 0.5, 1, 0).reshape(256, 256)
        print(self.filtered_image.shape)
        print(predicted_image.shape)
        print(np.unique(self.filtered_image))
        print(np.unique(predicted_image))
        self.segmented_image = (self.filtered_image * predicted_image[..., np.newaxis]) * 255
        self.segmented_image = self.segmented_image.astype("uint8")
        
        cv2.imshow("Predicted Image", predicted_image.astype("float64"))
        cv2.imwrite("Results/Predicted_Image.png", predicted_image.astype("float64"))
        cv2.imshow("Segmented Image", self.segmented_image)
        cv2.imwrite("Results/Segmented_Image.png", self.segmented_image)
    
    def classify_image(self):
        print()
        print("Classification....\n")
        # # Attention based Convolutional Hidden Markow Network (ACHMN)
        # Integrated Dung goose optimization algorithm
        with custom_object_scope(self.custom_objects):
            loaded_model = load_model(self.model_path)
        img=self.segmented_image.astype('float32')
        predicted=loaded_model.predict(img.reshape(1,img.shape[0],img.shape[1],img.shape[-1]));predicted=model.predict(self.segmented_image)
        self.predicted_class = self.classes[int(predicted)]
        print("Predicted Class:", self.predicted_class)
    
    def display_results(self):
        print("\nDisplaying Results ....\n")
        Plot()

    def process(self):
        self.select_image()
        self.preprocess_image()
        self.segment_image()
        self.classify_image()
        self.display_results()


if __name__ == "__main__":
    processor = MedicalImageProcessor()
    processor.process()




