# face_detection_and_recognition

This project enables the creation of a face dataset, training of a face recognizer, and testing the recognizer. Follow these steps to use the system.

Setup
Create Dataset
Modify utils.py with your parameters.
Run face_extractor.py.
Enter the names of the people to include in the dataset.
Look at the camera and press "s" to save the cropped face.
Press "q" to exit.
Train Recognizer
Ensure the dataset is ready.
Run recognizer_train.py to create the .yml file for the trained model.
Test Recognizer
Run recognizer_test.py.
Look at the camera to test the recognizer with one or more people.
Scripts Overview
face_detector.py: Try face detection.
face_extractor.py: Create the face dataset.
recognizer_train.py: Train the face recognizer.
recognizer_test.py: Test the trained recognizer.
By following these steps, you can create, train, and test a face recognition model efficiently.
