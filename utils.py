trained_recognizer_path = '//recognizer_trained.yml'
dataset_path = '//faces'
CAMERA_INDEX = 0

import os
import cv2
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_index_person_dict(path):
    index_person_dict = {}
    for index,person in enumerate(os.listdir(path)):
        index_person_dict[index] = person
    return index_person_dict

def image_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def get_face_from_frame(frame,face_cascade):
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(image_to_gray(frame), 1.1, 12)
    return faces
