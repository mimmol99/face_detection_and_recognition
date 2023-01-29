# face_detection_and_recognition

Through this code is possible to create a dataset of faces using a face detector (face_detecor.py to try it and face_extractor.py to create the dataset),train a face recognizer through a dataset made of directories with labels name and relative images inside. (all automatically done by face_extractor.py).
after the dataset is ready use recognizer_train to create the yml used to load the trained model and use it.
See the recognizer test to test the recognizer.

To create the dataset:
modify utils.py with your parameters.
run "face_extractor.py".
insert the name of the people to insert in dataset.
look the camera and press "s" to save the cropped face.
press "q" to exit.

To test the recognizer:
just run the recognizer_test.py and look the camera(also more than one person at time)
