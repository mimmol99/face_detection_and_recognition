import os
import cv2
from utils import *

#create and load the faces classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trained_recognizer_path)

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "purple": (128, 0, 128),
    "orange": (0, 165, 255),
    "pink": (0, 192, 203)
}

colors_index= {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "orange",
    5: "pink"
}


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        faces = get_face_from_frame(frame,face_cascade)
        index_person_dict = get_index_person_dict(dataset_path)

        for index,(x, y, w, h) in enumerate(faces):
            tickness = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), colors[colors_index[index]], tickness)
            gray =image_to_gray(frame)
            cv2.imshow('Camera_Gray',gray)
            cropped_gray = gray[y+tickness:y+h-tickness, x+tickness:x+w-tickness]
            cv2.imshow('Cropped_gray',cropped_gray)
            id,distance = recognizer.predict(cropped_gray)
            cv2.putText(frame, index_person_dict[id]+" "+str(distance), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the frame with the detected faces
        cv2.imshow('Camera', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
