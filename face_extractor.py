import cv2
import os
from utils import *
# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the default camera
cap = cv2.VideoCapture(0)




check_path(dataset_path)

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



name = input("name:")
check_path(os.path.join(dataset_path,name))
print("found:",len(os.listdir(os.path.join(dataset_path,name)))," images of ",name)
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 12)

    # Draw rectangles around the faces

    for index,(x, y, w, h) in enumerate(faces):
        tickness = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[colors_index[index]], tickness)

    # Display the frame with the detected faces
    cv2.imshow('Faces', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        # Save the frame to a file
        for index,(x, y, w, h) in enumerate(faces):
            
            cropped_image = frame[y+tickness:y+h-tickness, x+tickness:x+w-tickness]  # Crop the frame to only include the rectangle around the face
            cv2.imwrite(dataset_path+'/'+name+'/'+str(len(os.listdir(dataset_path+'/'+name)))+".jpg", cropped_image)
            print("Frame saved!",str(len(os.listdir(dataset_path+'/'+name))))

    # Exit the loop if the 'q' key is pressed
    if key & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

