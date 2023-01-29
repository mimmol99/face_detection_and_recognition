import os
import numpy as np
import cv2
from PIL import Image # For face recognition we will the the LBPH Face Recognizer
from utils import *

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1,neighbors=10,grid_x = 10,grid_y = 10)


def main():

    faces = []
    IDs = []

    for index,person in enumerate(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path,person)
        print(index,person,len(os.listdir(person_path)))
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path,image_name)
            facesImg = Image.open(image_path).convert('L')
            faceNP = np.array(facesImg, 'uint8')
            faces.append(faceNP)
            IDs.append(index)


    recognizer.train(faces,np.array(IDs))
    recognizer.save('./recognizer_trained.yml')
    print("Finish training")

if __name__ == "__main__":
    main()
