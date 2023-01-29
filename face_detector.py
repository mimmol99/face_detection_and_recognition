import cv2
from utils import *
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

    # Open the default camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

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
            cv2.putText(frame, 'Test', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # Display the frame with the detected faces
        cv2.imshow('Faces', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
